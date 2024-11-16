import requests
from pathlib import Path
import json
import logging
from PIL import Image
import io
import base64

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_base64_image(base64_str: str, output_path: str):
    """Base64 문자열을 이미지로 저장"""
    try:
        img_data = base64.b64decode(base64_str)
        with open(output_path, 'wb') as f:
            f.write(img_data)
        logger.info(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")

def test_calibration_api():
    # API 설정
    url = "http://localhost:8000/calibrate"
    image_path = "test_images\KakaoTalk_20241111_052053487.jpg"  # 실제 이미지 경로로 변경해주세요
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 이미지 파일 확인
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        return
    
    # 이미지 크기 확인
    with Image.open(image_path) as img:
        logger.info(f"Input image size: {img.size}")
    
    # API 요청
    with open(image_path, 'rb') as f:
        files = {
            'file': (Path(image_path).name, f, 'image/jpeg')
        }
        params = {
            'normalize': True
        }
        
        try:
            logger.info("Sending request to API...")
            response = requests.post(url, files=files, params=params)
            
            logger.info(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Saving results...")
                
                # JSON 결과 저장
                with open(output_dir / "calibration_results.json", 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info("Saved calibration results to JSON")
                
                # 시각화 결과 저장 (있는 경우에만)
                for key in ['comparison_plot', 'calibration_plot', 'undistorted_image']:
                    if key in result:
                        try:
                            save_base64_image(
                                result[key], 
                                output_dir / f"{key}.png"
                            )
                            logger.info(f"Saved {key}")
                        except Exception as e:
                            logger.error(f"Error saving {key}: {e}")
                
                # 결과 출력
                logger.info("\nCamera Parameters:")
                if 'camera_parameters' in result:
                    logger.info(json.dumps(result['camera_parameters'], indent=2))
                
                if 'uncertainties' in result:
                    logger.info("\nUncertainties:")
                    logger.info(json.dumps(result['uncertainties'], indent=2))
                
                logger.info("\nResults have been saved to 'test_outputs' directory")
                
            else:
                logger.error("Error response:")
                logger.error(response.text)
                try:
                    error_detail = response.json()
                    logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    logger.error("Could not parse error response as JSON")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
        except KeyError as e:
            logger.error(f"Missing key in response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_calibration_api()