import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import io
import torch
import numpy as np
from PIL import Image
import base64
from unittest.mock import Mock, patch

# Add the parent directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app import (
    app,
    ImageProcessingConfig,
    ImagePreprocessor,
    CalibrationVisualizer,
    ImageProcessingError,
    GeoCalib  # GeoCalib import 추가
)

# Fixtures
@pytest.fixture
def test_client():
    """테스트 클라이언트 fixture"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def config():
    """테스트용 이미지 처리 설정"""
    return ImageProcessingConfig(
        target_size=(720, 1280),
        max_size=1024 * 1024,  # 1MB for testing
        allowed_extensions={'.jpg', '.jpeg', '.png'}
    )

@pytest.fixture
def preprocessor(config):
    """이미지 전처리기 fixture"""
    return ImagePreprocessor(config)

@pytest.fixture
def sample_image():
    """테스트용 샘플 이미지"""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_geocalib():
    """GeoCalib 모델 모의 객체"""
    with patch('app.GeoCalib', autospec=True) as mock:
        # Mock the model instance
        mock_instance = Mock()
        
        # Create a mock tensor that behaves like an image tensor
        mock_image_tensor = torch.zeros((3, 720, 1280))
        
        # Mock the undistort_image method
        def mock_undistort(x):
            return [mock_image_tensor]
        
        mock_camera = Mock()
        mock_camera.focal_length = torch.tensor([1.0, 1.0])
        mock_camera.principal_point = torch.tensor([0.5, 0.5])
        mock_camera.rotation = torch.tensor([[1.0, 0.0, 0.0], 
                                           [0.0, 1.0, 0.0], 
                                           [0.0, 0.0, 1.0]])
        mock_camera.undistort_image = mock_undistort
        
        mock_instance.calibrate.return_value = {
            "camera": [mock_camera],
            "gravity": [Mock()],
            "up_confidence": [torch.ones(1)],
            "latitude_confidence": [torch.ones(1)]
        }
        
        mock.return_value = mock_instance
        yield mock

class TestImageProcessingConfig:
    """ImageProcessingConfig 클래스 테스트"""
    
    def test_default_initialization(self):
        """기본 설정 초기화 테스트"""
        config = ImageProcessingConfig()
        assert config.target_size == (720, 1280)
        assert config.max_size == 10 * 1024 * 1024
        assert '.jpg' in config.allowed_extensions
        assert '.jpeg' in config.allowed_extensions
        assert '.png' in config.allowed_extensions

    def test_custom_initialization(self):
        """사용자 정의 설정 초기화 테스트"""
        custom_config = ImageProcessingConfig(
            target_size=(480, 640),
            max_size=2 * 1024 * 1024,
            allowed_extensions={'.png'}
        )
        assert custom_config.target_size == (480, 640)
        assert custom_config.max_size == 2 * 1024 * 1024
        assert custom_config.allowed_extensions == {'.png'}

class TestImagePreprocessor:
    """ImagePreprocessor 클래스 테스트"""

    def test_validate_image_size(self, preprocessor):
        """이미지 크기 검증 테스트"""
        # 크기 제한을 초과하는 이미지 생성 (2MB)
        large_image = np.ones((2000, 2000, 3), dtype=np.uint8) * 255
        img_byte_arr = io.BytesIO()
        Image.fromarray(large_image).save(img_byte_arr, format='PNG', optimize=False)
        img_data = img_byte_arr.getvalue()
        
        # 이미지 크기가 제한을 초과하는지 확인
        assert len(img_data) > preprocessor.config.max_size
        
        with pytest.raises(ImageProcessingError):
            preprocessor.validate_image(img_data, "test.png")

    def test_validate_image_format(self, preprocessor, sample_image):
        """이미지 형식 검증 테스트"""
        with pytest.raises(ImageProcessingError, match="Unsupported file type"):
            preprocessor.validate_image(sample_image.getvalue(), "test.gif")

    def test_resize_with_aspect_ratio(self, preprocessor):
        """이미지 리사이징 테스트"""
        original_image = Image.new('RGB', (1920, 1080))
        resized = preprocessor.resize_with_aspect_ratio(original_image)
        
        # 크기 제한 확인
        assert resized.size[0] <= 1280
        assert resized.size[1] <= 720
        
        # 종횡비 보존 확인
        original_ratio = 1920 / 1080
        resized_ratio = resized.size[0] / resized.size[1]
        assert abs(original_ratio - resized_ratio) < 0.01

    @patch('torch.cuda.is_available', return_value=False)
    def test_preprocess_pipeline(self, mock_cuda, preprocessor, sample_image):
        """전체 전처리 파이프라인 테스트"""
        image_tensor, original_size = preprocessor.preprocess(
            sample_image.getvalue(),
            "test.png",
            normalize=True
        )
        
        assert isinstance(image_tensor, torch.Tensor)
        assert len(image_tensor.shape) == 3  # (C, H, W)
        assert image_tensor.max() <= 1.0  # 정규화 확인
        assert image_tensor.min() >= 0.0
        assert original_size == (100, 100)

@patch('app.GeoCalib')
@patch('torch.cuda.is_available', return_value=False)
class TestAPIEndpoints:
    """API 엔드포인트 테스트"""

    def test_calibrate_endpoint_success(self, mock_cuda, mock_model, test_client, sample_image):
        """캘리브레이션 엔드포인트 성공 케이스"""
        # Mock 모델 설정
        mock_instance = Mock()
        mock_camera = Mock()
        mock_camera.focal_length = torch.tensor([1.0, 1.0])
        mock_camera.principal_point = torch.tensor([0.5, 0.5])
        mock_camera.rotation = torch.tensor([[1.0, 0.0, 0.0], 
                                           [0.0, 1.0, 0.0], 
                                           [0.0, 0.0, 1.0]])
        mock_camera.undistort_image = lambda x: x
        
        mock_instance.calibrate.return_value = {
            "camera": [mock_camera],
            "gravity": [Mock()],
            "up_confidence": [torch.ones(1)],
            "latitude_confidence": [torch.ones(1)]
        }
        mock_model.return_value = mock_instance

        response = test_client.post(
            "/calibrate",
            files={"file": ("test.png", sample_image, "image/png")},
            params={"normalize": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "camera_parameters" in data

    def test_calibrate_endpoint_invalid_file(self, mock_cuda, mock_model, test_client):
        """잘못된 파일 형식으로 요청 시 테스트"""
        response = test_client.post(
            "/calibrate",
            files={"file": ("test.txt", b"invalid data", "text/plain")},
            params={"normalize": True}
        )
        
        assert response.status_code == 400

    def test_calibrate_endpoint_missing_file(self, mock_cuda, mock_model, test_client):
        """파일 없이 요청 시 테스트"""
        response = test_client.post("/calibrate")
        assert response.status_code == 422

    def test_calibrate_endpoint_large_file(self, mock_cuda, mock_model, test_client):
        """크기 제한 초과 파일 업로드 테스트"""
        # 2MB 크기의 이미지 생성
        large_image = np.ones((2000, 2000, 3), dtype=np.uint8) * 255
        img_byte_arr = io.BytesIO()
        Image.fromarray(large_image).save(img_byte_arr, format='PNG', optimize=False)
        img_byte_arr.seek(0)
        
        response = test_client.post(
            "/calibrate",
            files={"file": ("large.png", img_byte_arr, "image/png")},
            params={"normalize": True}
        )
        
        assert response.status_code == 400
        assert "Image size exceeds" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])