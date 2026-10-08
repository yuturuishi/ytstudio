# ytstudio

**Language / 语言：** [简体中文](README.md) | [English](README_en.md)

> yuturuishi's open-source collection of AI algorithm training frameworks — covering object detection, image / video / audio classification, face recognition, license-plate recognition, and onnx → rknn model conversion for Rockchip edge devices.

| | |
|---|---|
| Website | https://www.yuturuishi.com |
| WeChat | yuturuishi |
| Gitee | https://gitee.com/yuturuishi/ytstudio |
| GitHub | https://github.com/yuturuishi/ytstudio |

## Introduction

ytstudio is a modular collection of training and deployment tools for AI vision and audio algorithms. Each task type maps to an independent training module, shipped with free public datasets and example scripts, so developers can move quickly through the full pipeline: data preparation → model training → edge deployment. For Rockchip hardware, use `onnx2rknn` to convert the trained onnx model into rknn format for accelerated inference on the edge.

## Modules

| Module | Task | Description | Status |
|---|---|---|---|
| Train_yolo8 / Train_yolo11 / Train_yolo26 | Object detection | Official YOLO training frameworks | Built-in |
| Train_DFINE | End-to-end object detection | D-FINE-based (Apache-2.0) NMS-free detection framework; exports to ONNX / OpenVINO / TensorRT | Built-in |
| Train_rk_yolo5 / Train_rk_yolo8 / Train_rk_yolo11 | Edge object detection | YOLO training frameworks for Rockchip devices | Built-in |
| onnx2rknn | Model conversion | Convert onnx models to rknn on x86 / arm | Built-in |
| Train_ResNet | Image classification | ResNet-based image classification training framework | Built-in |
| Train_AudioNet | Audio classification | Audio clip classification training framework | Migrated to standalone repo |
| Train_VideoNet | Video classification | Video clip classification training framework | Migrated to standalone repo |
| Train_XcFaceNet | Face recognition | Face feature extraction training framework | Migrated to standalone repo |
| Train_PaddleOCR_Plate | License-plate recognition | Commercial-grade plate recognition based on PaddleOCR & PP-OCRv4 | Migrated to standalone repo |

## Project Structure

ytstudio is organized by task type into multiple independent training modules:

```
ytstudio/
├── Train_yolo8/  Train_yolo11/  Train_yolo26/    # Official YOLO training frameworks (object detection)
├── Train_DFINE/          # D-FINE end-to-end NMS-free detection framework (Apache-2.0)
├── Train_rk_yolo5/  Train_rk_yolo8/  Train_rk_yolo11/  # Rockchip edge YOLO training frameworks
├── onnx2rknn/              # onnx → rknn model conversion tool
├── Train_ResNet/          # Image classification training framework
├── Train_AudioNet/        # Audio classification (migrated to standalone repo)
├── Train_VideoNet/        # Video classification (migrated to standalone repo)
├── Train_XcFaceNet/       # Face recognition (migrated to standalone repo)
└── Train_PaddleOCR_Plate/ # License-plate recognition (migrated to standalone repo)
```

## Train_DFINE

`Train_DFINE` is a localized fork of [Peterande/D-FINE](https://github.com/Peterande/D-FINE) (Apache-2.0), providing a reproducible pipeline: train on your own dataset → export ONNX → convert to OpenVINO / TensorRT. See the module's `README.md` for details.

- **License**: The D-FINE core framework is **Apache-2.0**, free for learning, research, and commercial use with no AGPL network-service terms — more permissive for closed-source distribution than YOLO (Ultralytics AGPL-3.0). This module's own training code inherits ytstudio's MIT; **user-trained weights are labeled "self-owned" (自有)**.
- **Usage**: High-accuracy general object detection — train a dedicated model on your own dataset and export it as ONNX / OpenVINO / TensorRT for deployment on server GPUs or x86 edge devices.
- **Capabilities**
  - End-to-end detection with **no NMS** post-processing, simplifying the deployment chain;
  - 5 official COCO pretrained weights (n / s / m / l / x), supporting fine-tuning or training from scratch;
  - COCO-format custom datasets, with a built-in YOLO → COCO batch conversion tool;
  - One-click export: ONNX (dynamic batch, opset 17) → OpenVINO IR (static fp32, target runtime 2024.6) → TensorRT engine (static fp32);
  - Validated on 13 security / behavior-analysis datasets (flame & smoke, rat, dust, fighting, student states, sleep-on-duty, phone, climbing, helmet, dense crowd, etc.).
- **Commercial Value**: Apache-2.0 permits free commercial use and closed-source distribution; NMS-free design simplifies inference; high-accuracy detection fits security surveillance, industrial inspection, and behavior analysis; exported OpenVINO / TensorRT models plug directly into commercial systems such as rebekah.

## Tech Stack

PyTorch + CUDA | Ultralytics YOLO | OpenCV / PIL / NumPy | RKNN Toolkit / ONNX Runtime | Python

## Requirements

- **Python** 3.8+
- **Training**: PyTorch + CUDA (see each module's README for exact versions)
- **Edge deployment**: Rockchip model conversion environment (see `onnx2rknn` module README for install options)

## Quick Start

1. Download a training dataset (see "Free Training Datasets" below).
2. Enter the corresponding training module directory, configure the dataset path per that module's README, and start training.
3. After training produces an onnx model, if deploying to Rockchip devices, use `onnx2rknn` to convert it to rknn.

> Each built-in module ships its own README with configuration and training details; follow the module README.

## Submodules & Standalone Repos

Some training frameworks have been split into standalone repos for maintenance:

- **AudioNet** (audio classification): [Gitee](https://gitee.com/yuturuishi/AudioNet) · [GitHub](https://github.com/yuturuishi/AudioNet)
- **ResNet** (image classification): [Gitee](https://gitee.com/yuturuishi/ResNet) · [GitHub](https://github.com/yuturuishi/ResNet)
- **VideoNet** (video classification): [Gitee](https://gitee.com/yuturuishi/VideoNet) · [GitHub](https://github.com/yuturuishi/VideoNet)
- **XcFaceNet** (face recognition): [Gitee](https://gitee.com/yuturuishi/XcFaceNet) · [GitHub](https://github.com/yuturuishi/XcFaceNet)
- **PaddleOCR_Plate** (license-plate recognition): [Gitee](https://gitee.com/yuturuishi/PaddleOCR_Plate) · [GitHub](https://github.com/yuturuishi/PaddleOCR_Plate)

## Sample Collection & Annotation

We recommend **[ytlabel](https://gitee.com/yuturuishi/ytlabel)** ([GitHub](https://github.com/yuturuishi/ytlabel)) — it combines sample collection (video frame extraction) and annotation, supports importing images / videos / LabelMe datasets and rectangle / polygon annotation, and exports YOLO-format datasets directly (with train / val / test split and `data.yaml`).

You can also annotate with [labelme](https://pan.quark.cn/s/7e6accce2a3e), then import into ytlabel to export datasets.

## Free Training Datasets

- Training datasets — Quark Drive: https://pan.quark.cn/s/5dcc2f724bcc
- Dense crowd detection dataset 20250625
- Bright kitchen (明厨亮灶) detection dataset 20250625
- Climbing detection dataset 20250624
- Smoking detection dataset 20241012
- Fighting detection dataset 20241012
- Reflective-vest detection dataset 20241013
- Dust detection dataset 20241013
- Flame & smoke detection dataset 20241012
- Head & helmet detection dataset 20241013
- Human 5-action (stand / fall / sit / squat / run) dataset 20241012
- Student 3-state dataset 20241022
- Sleep-on-duty detection dataset 20241210
- Phone detection dataset 20251206
- Cat & dog 2-class dataset — Quark Drive: https://pan.quark.cn/s/982dd16cb29d
- Vehicle 9-class dataset — Quark Drive: https://pan.quark.cn/s/f698d0e99a4b
- Snore / non-snore speech 2-class dataset — Quark Drive: https://pan.quark.cn/s/4d83dabff0a6

## License

ytstudio's own code is released under the MIT License; keep the copyright notice to use it freely. Third-party libraries are subject to their respective licenses.

### Third-party dependency compliance

- **Ultralytics (AGPL-3.0)**: training models with it and exporting to ONNX / RKNN for deployment on your own hardware is not bound by AGPL's network-service terms; but if you offer Ultralytics software itself as a closed-source product, private deployment, SaaS, or edge/embedded device, you must comply with Ultralytics' license terms or purchase its Enterprise License. ytstudio's MIT license does not cover that obligation.
- **D-FINE (Apache-2.0)**: unlike YOLO (Ultralytics AGPL-3.0), D-FINE uses the Apache-2.0 license, free for commercial use and closed-source distribution without AGPL network-service concerns. `Train_DFINE` is based on Peterande/D-FINE; D-FINE's own license obligations follow Apache-2.0, and user-trained weights are labeled "self-owned" (自有).
- **Models & datasets**: training datasets and pretrained weights referenced by this repo are governed by their provider's license; Ultralytics pretrained weights (`.pt`) are subject to AGPL / Enterprise license terms.
- **Network requests**: ytstudio is a local training tool and does not send any user data or runtime info to any external server.
- **Disclaimer**: users are responsible for the legality of their models, datasets, and deployment environment. ytstudio is provided "as is", without warranty.
