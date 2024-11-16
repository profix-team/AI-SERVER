# geocalib import (로컬 폴더에서 import)
from GeoCalib.geocalib import viz2d, GeoCalib
from GeoCalib.geocalib.camera import camera_models
from GeoCalib.geocalib.gravity import Gravity
from GeoCalib.geocalib.utils import deg2rad, print_calibration

from fastapi.responses import JSONResponse
from typing import Any

class UnicodeJSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
import base64
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import ImageOps
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
TARGET_SIZE = (720, 1280)

@dataclass
class ImageProcessingConfig:
    """이미지 처리 설정을 위한 데이터 클래스"""
    target_size: Tuple[int, int] = TARGET_SIZE
    max_size: int = MAX_IMAGE_SIZE
    allowed_extensions: List[str] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ALLOWED_EXTENSIONS

class ImageProcessingError(Exception):
    """이미지 처리 관련 커스텀 예외"""
    pass

@contextmanager
def plot_context():
    """matplotlib figure 컨텍스트 매니저"""
    try:
        yield
    finally:
        plt.close('all')

class ImagePreprocessor:
    """이미지 전처리를 위한 클래스"""
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def validate_image(self, image_data: bytes, filename: str) -> None:
        """이미지 유효성 검증"""
        if len(image_data) > self.config.max_size:
            raise ImageProcessingError(f"Image size exceeds {self.config.max_size/1024/1024}MB limit")
            
        ext = Path(filename).suffix.lower()
        if ext not in self.config.allowed_extensions:
            raise ImageProcessingError(f"Unsupported file type. Allowed types: {self.config.allowed_extensions}")
            
        try:
            Image.open(io.BytesIO(image_data)).verify()
        except Exception as e:
            raise ImageProcessingError(f"Invalid image file: {str(e)}")
    
    def resize_with_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """종횡비를 유지하면서 이미지 크기 조정"""
        width, height = image.size
        aspect_ratio = width / height
        target_aspect_ratio = self.config.target_size[1] / self.config.target_size[0]
        
        if aspect_ratio > target_aspect_ratio:
            new_width = self.config.target_size[1]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.config.target_size[0]
            new_width = int(new_height * aspect_ratio)
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu')  # autocast 수정
    def preprocess(self, image_data: bytes, filename: str, normalize: bool = True) -> Tuple[torch.Tensor, tuple]:
        """전체 전처리 파이프라인"""
        try:
            self.validate_image(image_data, filename)
            
            with Image.open(io.BytesIO(image_data)) as image:
                original_size = image.size  # 원본 이미지 크기 저장
                image = ImageOps.exif_transpose(image)  # EXIF 정보에 따라 이미지 회전
                
                # 이미지 크기 조정
                image = self.resize_with_aspect_ratio(image)
                logger.info(f"Resized image size: {image.size}")
                
                # RGB로 변환
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 텐서로 변환
                to_tensor = transforms.ToTensor()
                image_tensor = to_tensor(image)
                
                # 정규화
                if normalize:
                    image_tensor = image_tensor / 255.0
                
                # 텐서 범위 확인
                logger.info(f"Tensor shape: {image_tensor.shape}")
                logger.info(f"Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
                
                return image_tensor, original_size
                
        except Exception as e:
            logger.error(f"Error in preprocess: {str(e)}")
            raise
        
class CameraCalibrationService:
    """카메라 캘리브레이션 서비스 클래스"""
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = ImagePreprocessor(config)
        self.model = self._initialize_model()
        self.visualizer = CalibrationVisualizer()

    def _initialize_model(self):
        try:
            return GeoCalib(weights="pinhole").to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize GeoCalib model: {e}")
            raise RuntimeError("Model initialization failed")

    async def process_image(self, file: UploadFile, normalize: bool = True) -> dict:
        try:
            contents = await file.read()
            image_tensor, original_size = self.preprocessor.preprocess(
                contents, file.filename, normalize
            )
            
            with torch.cuda.amp.autocast():
                results = self.model.calibrate(image_tensor, camera_model="pinhole")
            
            camera = results["camera"][0] if isinstance(results["camera"], list) else results["camera"]
            
            camera_params = {
                "focal_length": camera.focal_length.cpu().tolist(),
                "principal_point": camera.principal_point.cpu().tolist(),
                "rotation": camera.rotation.cpu().tolist(),
            }
            
            visualization_results = self.visualizer.process_visualization(image_tensor, {
                **results,
                "camera": camera
            })
            
            return {
                "status": "success",
                "original_size": {
                    "width": original_size[0],
                    "height": original_size[1]
                },
                "preprocessed_size": {
                    "width": self.config.target_size[1],
                    "height": self.config.target_size[0]
                },
                "camera_parameters": camera_params,
                **visualization_results
            }
            
        except ImageProcessingError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(status_code=503, detail="GPU memory exceeded")
        except Exception as e:
            logger.error(f"Unexpected error in process_image: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
class CalibrationVisualizer:
    """캘리브레이션 결과 시각화 클래스"""
    @staticmethod
    def fig_to_base64(fig) -> str:
        """matplotlib figure를 base64 문자열로 변환"""
        with io.BytesIO() as buf:
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    @staticmethod
    def process_visualization(image_tensor: torch.Tensor, results: Dict) -> Dict[str, str]:
        """이미지 처리 및 시각화 수행"""
        with plot_context():
            # 이미지 텐서 정규화 확인
            image_tensor = torch.clamp(image_tensor, 0, 1)  # 0-1 범위로 클리핑
            
            # 왜곡 보정된 이미지 생성
            with torch.no_grad():  # 메모리 사용 최적화
                undistorted = results["camera"][0].undistort_image(image_tensor[None])[0]
                undistorted = torch.clamp(undistorted, 0, 1)  # 결과도 0-1 범위로 클리핑
            
            # 원본/보정 이미지 비교 시각화
            fig1, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, im, title in zip(axes, [image_tensor, undistorted], ["Original", "Undistorted"]):
                # uint8로 변환하여 표시
                img_display = (im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                ax.imshow(img_display)
                ax.set_title(title)
                ax.axis("off")
            comparison_plot = CalibrationVisualizer.fig_to_base64(fig1)
            
            # 캘리브레이션 결과 시각화
            titles = ["Calibration Result", "Up Confidence", "Latitude Confidence"]
            fig2 = viz2d.plot_images([(image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)] * 3, titles)
            ax = fig2.get_axes()
            viz2d.plot_perspective_fields([results["camera"][0]], [results["gravity"][0]], axes=[ax[0]])
            viz2d.plot_confidences([results[f"{k}_confidence"][0] for k in ["up", "latitude"]], axes=ax[1:])
            plt.tight_layout()
            calibration_plot = CalibrationVisualizer.fig_to_base64(fig2)
            
            # 보정된 이미지를 base64로 변환
            with io.BytesIO() as undistorted_buffer:
                undistorted_img = Image.fromarray(
                    (undistorted.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                undistorted_img.save(undistorted_buffer, format='PNG')
                undistorted_base64 = base64.b64encode(undistorted_buffer.getvalue()).decode('utf-8')
            
            return {
                "comparison_plot": comparison_plot,
                "calibration_plot": calibration_plot,
                "undistorted_image": undistorted_base64
            }

# FastAPI 앱 설정
app = FastAPI(title="Camera Calibration API",default_response_class=UnicodeJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 객체 초기화
config = ImageProcessingConfig()
preprocessor = ImagePreprocessor(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeoCalib(weights="pinhole").to(device)
visualizer = CalibrationVisualizer()

def create_visualization(img: torch.Tensor, undistorted: torch.Tensor, results: Dict) -> Dict[str, str]:
    """시각화 데이터 생성"""
    with plot_context():
        # 원본/보정 이미지 비교
        fig1, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, im, title in zip(axes, [img, undistorted], ["Original", "Undistorted"]):
            ax.imshow(im.permute(1, 2, 0).cpu().numpy())
            ax.set_title(title)
            ax.axis("off")
        comparison_plot = fig_to_base64(fig1)
        
        # 캘리브레이션 결과 시각화
        titles = ["Calibration Result", "Up Confidence", "Latitude Confidence"]
        fig2 = viz2d.plot_images([img.permute(1, 2, 0).cpu().numpy()] * 3, titles)
        ax = fig2.get_axes()
        viz2d.plot_perspective_fields([results["camera"][0]], [results["gravity"][0]], axes=[ax[0]])
        viz2d.plot_confidences([results[f"{k}_confidence"][0] for k in ["up", "latitude"]], axes=ax[1:])
        plt.tight_layout()
        calibration_plot = fig_to_base64(fig2)
        
        # 보정된 이미지를 base64로 변환
        with io.BytesIO() as undistorted_buffer:
            undistorted_img = Image.fromarray(
                (undistorted.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            undistorted_img.save(undistorted_buffer, format='PNG')
            undistorted_base64 = base64.b64encode(undistorted_buffer.getvalue()).decode('utf-8')
        
        return {
            "comparison_plot": comparison_plot,
            "calibration_plot": calibration_plot,
            "undistorted_image": undistorted_base64
        }

def extract_camera_params(results: Dict) -> Dict:
    """카메라 파라미터 추출"""
    camera = results["camera"][0]
    camera_params = {
        "focal_length": camera.f.cpu().numpy().tolist() if hasattr(camera, 'f') else None,
        "principal_point": camera.c.cpu().numpy().tolist() if hasattr(camera, 'c') else None,
        "size": camera.size.cpu().numpy().tolist() if hasattr(camera, 'size') else None,
        "hfov": float(camera.hfov.cpu().numpy()) if hasattr(camera, 'hfov') else None,
        "vfov": float(camera.vfov.cpu().numpy()) if hasattr(camera, 'vfov') else None
    }
    return camera_params

# 서비스 초기화
@app.on_event("startup")
async def startup_event():
    app.state.config = ImageProcessingConfig()
    app.state.calibration_service = CameraCalibrationService(app.state.config)
    
def fig_to_base64(fig) -> str:
    """matplotlib figure를 base64 문자열로 변환"""
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.post("/calibrate")
async def calibrate_image(
    file: UploadFile = File(...),
    sensitivity: float = Query(0.01, description="Distortion detection sensitivity (0.001-0.1)")
) -> Dict[str, Any]:
    try:
        contents = await file.read()
        temp_file = Path("temp_image.jpg")
        
        with open(temp_file, "wb") as f:
            f.write(contents)
            
        try:
            # 각 카메라 모델별로 왜곡 분석
            model_results = {}
            distortion_scores = {}
            
            # 여러 카메라 모델로 테스트
            camera_models = ["pinhole", "simple_radial", "simple_divisional"]
            for cam_model in camera_models:
                weights = "pinhole" if cam_model == "pinhole" else "distorted"
                model_instance = GeoCalib(weights=weights).to(device)
                
                # 이미지 로드
                img = model_instance.load_image(str(temp_file)).to(device)
                results = model_instance.calibrate(img, camera_model=cam_model)
                undistorted = results["camera"].undistort_image(img[None])[0]
                
                # 왜곡 점수 계산
                distortion_score = calculate_distortion_score(img, undistorted)
                distortion_scores[cam_model] = distortion_score
                model_results[cam_model] = (results, img, undistorted)
                
                logger.info(f"Model {cam_model} distortion score: {distortion_score}")
            
            # 가장 큰 왜곡 보정 효과를 보인 모델 선택
            best_model = max(distortion_scores.items(), key=lambda x: x[1])[0]
            results, img, undistorted = model_results[best_model]
            
            # 왜곡 분석
            distortion_analysis = analyze_distortion(
                img, 
                undistorted, 
                distortion_scores[best_model],
                sensitivity
            )
            
            # 시각화
            visualization_data = create_visualization(img, undistorted, results)
            
            # 카메라 파라미터 추출
            camera_params = extract_camera_params(results)
            
            return JSONResponse(content={
                "status": "success",
                "distortion_analysis": distortion_analysis,
                "camera_model_used": best_model,
                "distortion_scores": distortion_scores,
                "camera_parameters": camera_params,
                **visualization_data
            })
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

def calculate_distortion_score(original: torch.Tensor, undistorted: torch.Tensor) -> float:
    """이미지 왜곡 점수 계산"""
    with torch.no_grad():
        # 픽셀 차이
        pixel_diff = torch.abs(original - undistorted).mean().item()
        
        # 에지 기반 왜곡 측정
        edge_diff = calculate_edge_difference(original, undistorted)
        
        # 구조적 유사성
        ssim_score = calculate_ssim(original, undistorted)
        
        # 종합 점수 계산
        distortion_score = (pixel_diff * 0.3 + edge_diff * 0.4 + (1 - ssim_score) * 0.3)
        
        return distortion_score

def calculate_edge_difference(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """에지 차이 계산"""
    # Sobel 필터 적용
    def get_edges(img):
        gray = img.mean(dim=0, keepdim=True)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device).float()
        
        edges_x = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))
        edges_y = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))
        return torch.sqrt(edges_x.pow(2) + edges_y.pow(2))
    
    edges1 = get_edges(img1)
    edges2 = get_edges(img2)
    return torch.abs(edges1 - edges2).mean().item()

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """구조적 유사성 계산"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = torch.nn.functional.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = torch.nn.functional.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def analyze_distortion(original: torch.Tensor, 
                        undistorted: torch.Tensor, 
                        distortion_score: float,
                        sensitivity: float) -> Dict[str, Any]:
    """왜곡 분석 결과 생성"""
    # 민감도에 따른 임계값 설정
    thresholds = {
        "minimal": sensitivity,
        "moderate": sensitivity * 2,
        "severe": sensitivity * 3
    }
    
    analysis = {
        "has_distortion": distortion_score > thresholds["minimal"],
        "distortion_score": float(distortion_score),
        "severity": "정상",
        "details": {}
    }
    
    # 왜곡 심각도 판단
    if distortion_score > thresholds["severe"]:
        analysis["severity"] = "심각"
    elif distortion_score > thresholds["moderate"]:
        analysis["severity"] = "중간"
    elif distortion_score > thresholds["minimal"]:
        analysis["severity"] = "경미"
        
    # 세부 분석
    analysis["details"] = {
        "pixel_difference": float(torch.abs(original - undistorted).mean().item()),
        "edge_difference": float(calculate_edge_difference(original, undistorted)),
        "structural_similarity": float(calculate_ssim(original, undistorted))
    }
    
    # 권장 사항
    analysis["recommendation"] = generate_recommendation(analysis)
    
    return analysis

def generate_recommendation(analysis: Dict[str, Any]) -> str:
    """분석 결과에 따른 권장사항 생성"""
    if not analysis["has_distortion"]:
        return "이미지가 정상적으로 촬영되었습니다."
        
    if analysis["severity"] == "심각":
        return "이미지에 심각한 왜곡이 감지되었습니다. 다른 렌즈나 카메라로 재촬영을 권장합니다."
    elif analysis["severity"] == "중간":
        return "이미지에 중간 수준의 왜곡이 있습니다. 가능하다면 더 넓은 거리에서 촬영하는 것을 권장합니다."
    else:
        return "이미지에 경미한 왜곡이 감지되었습니다. 현재 상태로도 사용 가능하나, 필요시 재촬영을 고려해보세요."
            
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)