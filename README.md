# Lightweight Cross Fusion (RGB-T) for Small-Object UAV Detection

## 1) Environment
```bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8

# Choose torch build for your GPU (example: CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt
```

## 2) Datasets (RGB-T / UAV)
- LLVIP: https://github.com/bupt-ai-cz/LLVIP
- M3FD: https://github.com/ZhiHanZ/Multispectral-Salient-Object-Detection
- KAIST: https://soonminhwang.github.io/rgbt-ped-detection/
- M3FD/KAIST/LLVIP YAMLs: see `ultralytics/cfg/datasets/` (e.g., `LLVIP.yaml`, `LLVIP_r20.yaml`, `KAIST.yaml`, `M3FD.yaml`).

Place datasets under `datasets/` or adjust the `path` in those YAMLs.

## 3) Models/Configs using LightweightCrossFusion
- `yolov11_rgbt_lightweight.yaml` (RGB-T fusion with lightweight cross fusion at P3/P5)
- `yolov11_rgbt_enhanced_fusion.yaml` (enhanced fusion, lightweight blocks)
- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-add.yaml` (mid-level fusion example)

These configs call the module from `ultralytics.nn.modules.conv.LightweightCrossFusion`.

## 4) Quick Start (Training)
```bash
# Example: LLVIP
python train.py --model yolov11_rgbt_lightweight.yaml --data ultralytics/cfg/datasets/LLVIP.yaml --epochs 300

# Example: KAIST
python train.py --model yolov11_rgbt_lightweight.yaml --data ultralytics/cfg/datasets/KAIST.yaml --epochs 300

# Example: M3FD
python train.py --model yolov11_rgbt_lightweight.yaml --data ultralytics/cfg/datasets/M3FD.yaml --epochs 300

# Choose scale: add --scale n/s/m/l/x if supported by the YAML
```

## 5) Using the Module in Python
```python
import torch
from ultralytics.nn.modules.conv import LightweightCrossFusion

fusion = LightweightCrossFusion(c1=256, c2=256, reduction=8)
rgb = torch.randn(1, 128, 64, 64)
ir  = torch.randn(1, 128, 64, 64)
out = fusion([rgb, ir])
print(out.shape)  # fused RGB-T features
```

## 6) What is LightweightCrossFusion
- Two-branch RGB/IR fusion with channel attention per modality and a lightweight fusion conv.
- Learnable modal weights + output projection to stabilize gradients and reduce params.
- Designed for memory efficiency on resource-constrained GPUs while keeping RGB-T gains.

## 7) Key Params
- `c1`: total input channels (typically sum of RGB + IR feature dims).
- `c2`: output channels after fusion.
- `reduction`: channel attention reduction (smaller = stronger attention, more params).

## 8) Tuning Tips for RGB-T UAV Small Objects
- Keep input resolutions moderate to save memory; scale up only if recall is low.
- If VRAM tight: increase `reduction`, or reduce feature widths in the YAML (e.g., lower `c2` in fusion stages).
- Balance RGB/IR quality: if one modality is noisy, consider pre-denoise or stronger attention (lower `reduction`).

## 9) References
- Cites multispectral RGB-T detection benchmarks: LLVIP, KAIST, M3FD.
- Please also cite YOLO/Ultralytics when publishing results.


