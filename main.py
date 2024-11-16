import torch
import matplotlib.pyplot as plt

from geocalib import viz2d, GeoCalib
from geocalib.camera import camera_models
from geocalib.gravity import Gravity
from geocalib.utils import deg2rad, print_calibration

# 사용할 이미지 경로 및 디바이스 설정
img_path = r"dis_image.png"  # 올바른 이미지 경로 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# @title Inference for other camera models
camera_models = {"pinhole": "Pinhole Model", "simple_radial": "Radial Distortion", "simple_divisional": "Divisional Distortion"}
print(f"Available camera models: {list(camera_models.keys())}")
camera_model = "simple_divisional"  # 원하는 카메라 모델 선택

# 카메라 모델에 따른 가중치 설정 및 모델 초기화
weights = "pinhole" if camera_model == "pinhole" else "distorted"
model = GeoCalib(weights=weights).to(device)

# 이미지 로드 및 텐서 변환 [C, H, W] 형식으로 변경
img = model.load_image(img_path).to(device)

# 캘리브레이션 수행 (선택한 카메라 모델에 맞춰)
results = model.calibrate(img, camera_model=camera_model)

# 결과를 바로 출력 (함수 따로 정의하지 않고 사용)
print_calibration(results)

# 왜곡된 이미지 보정 (예측된 카메라 파라미터로 보정)
undistorted = results["camera"].undistort_image(img[None])[0]

# 시각화: 원본 및 보정된 이미지 비교
titles = ["Original", "Undistorted"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, im, title in zip(axes, [img, undistorted], titles):
    ax.imshow(im.permute(1, 2, 0).cpu().numpy())
    ax.set_title(title)
    ax.axis("off")

# 추가적인 시각화: Calibration Result, Up Confidence, Latitude Confidence
titles = ["Calibration Result", "Up Confidence", "Latitude Confidence"]
fig = viz2d.plot_images([img.permute(1, 2, 0).cpu().numpy()] * 3, titles)
ax = fig.get_axes()
viz2d.plot_perspective_fields([results["camera"][0]], [results["gravity"][0]], axes=[ax[0]])
viz2d.plot_confidences([results[f"{k}_confidence"][0] for k in ["up", "latitude"]], axes=ax[1:])

plt.tight_layout()
plt.show()
