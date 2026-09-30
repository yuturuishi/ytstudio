# ytstudio

> 宇图瑞视（yuturuishi）开源 AI 算法训练框架集合 —— 覆盖目标检测、图像 / 视频 / 音频分类、人脸识别、车牌识别等任务，并提供瑞芯微（Rockchip）边缘设备的模型转换（onnx → rknn）全链路。

| | |
|---|---|
| 官网 | https://www.yuturuishi.com |
| 微信 | yuturuishi |
| Gitee | https://gitee.com/yuturuishi/ytstudio |
| GitHub | https://github.com/yuturuishi/ytstudio |

## 简介

ytstudio 是一套面向 AI 视觉与音频算法的训练与部署工具集合。它以**模块化**方式组织：每个任务类型对应一个独立训练模块，配套免费公开数据集与示例脚本，开发者可快速完成「数据准备 → 模型训练 → 边缘部署」的完整流程。对于需要落地到瑞芯微硬件的场景，使用 `onnx2rknn` 即可将训练产出的 onnx 模型转换为 rknn 格式，直接在边缘端加速推理。

## 功能模块

| 模块 | 任务类型 | 说明 | 状态 |
|---|---|---|---|
| Train_yolo8 / Train_yolo11 / Train_yolo26 | 目标检测 | YOLO 官方训练框架 | 内置 |
| Train_rk_yolo5 / Train_rk_yolo8 / Train_rk_yolo11 | 目标检测（边缘） | 适用于瑞芯微设备的 YOLO 模型训练框架 | 内置 |
| onnx2rknn | 模型转换 | 在 x86 / arm 上将 onnx 模型转换为 rknn 模型 | 内置 |
| Train_ResNet | 图像分类 | 基于 ResNet 的图片分类算法训练框架 | 内置 |
| labeltools | 数据预处理 | 样本转换脚本 | 内置 |
| Train_AudioNet | 音频分类 | 音频片段分类算法训练框架 | 已迁移独立仓库 |
| Train_VideoNet | 视频分类 | 视频片段分类算法训练框架 | 已迁移独立仓库 |
| Train_XcFaceNet | 人脸识别 | 人脸特征提取算法训练框架 | 已迁移独立仓库 |
| Train_PaddleOCR_Plate | 车牌识别 | 基于 PaddleOCR 与 PP-OCRv4 训练商用级车牌识别模型 | 已迁移独立仓库 |

## 环境要求

- **Python** 3.8 及以上
- **训练**：PyTorch + CUDA（具体版本见各模块 README 的环境说明）
- **边缘部署**：Rockchip 模型转换环境（见 `onnx2rknn` 模块 README，提供多种安装方式）

## 快速开始

1. 下载训练数据集（见下方「免费训练数据集」）。
2. 进入对应训练模块目录，按该模块的 README 配置数据集路径并启动训练。
3. 训练产出 onnx 模型后，如需部署到瑞芯微设备，使用 `onnx2rknn` 将其转换为 rknn 格式。

> 各内置模块均自带独立 README，包含该模块的配置与训练细节，请以模块 README 为准。

## 子模块与独立仓库

部分训练框架已拆分为独立仓库进行维护，点击跳转：

- **BXC_AudioNet**（音频分类）：[Gitee](https://gitee.com/yuturuishi/BXC_AudioNet) · [GitHub](https://github.com/yuturuishi/BXC_AudioNet)
- **BXC_ResNet**（图像分类）：[Gitee](https://gitee.com/yuturuishi/BXC_ResNet) · [GitHub](https://github.com/yuturuishi/BXC_ResNet)
- **BXC_VideoNet**（视频分类）：[Gitee](https://gitee.com/yuturuishi/BXC_VideoNet) · [GitHub](https://github.com/yuturuishi/BXC_VideoNet)
- **XcFaceNet**（人脸识别）：[Gitee](https://gitee.com/yuturuishi/XcFaceNet) · [GitHub](https://github.com/yuturuishi/XcFaceNet)
- **PaddleOCR_Plate**（车牌识别）：[Gitee](https://gitee.com/yuturuishi/PaddleOCR_Plate) · [GitHub](https://github.com/yuturuishi/PaddleOCR_Plate)

## 相关工具

- [视频分割图片工具 BXC_hs](https://gitee.com/yuturuishi/BXC_hs)
- [图片标注样本工具 labelme](https://pan.quark.cn/s/7e6accce2a3e)

## 免费训练数据集

- 训练数据集 - 夸克网盘下载地址：https://pan.quark.cn/s/5dcc2f724bcc
- 检测密集人群数据集 20250625
- 检测明厨亮灶数据集 20250625
- 检测攀爬数据集 20250624
- 检测抽烟数据集 20241012
- 检测打架数据集 20241012
- 检测反光衣数据集 20241013
- 检测粉尘数据集 20241013
- 检测火焰烟雾数据集 20241012
- 检测人头和安全帽数据集 20241013
- 检测人体 5 动作 - 站着 - 摔倒 - 坐 - 深蹲 - 跑数据集 20241012
- 检测学生 3 状态数据集 20241022
- 检测睡岗数据集 20241210
- 检测手机数据集 20251206
- 猫狗 2 分类数据集 - 夸克网盘下载地址 https://pan.quark.cn/s/982dd16cb29d
- 车型 9 分类数据集 - 夸克网盘下载地址 https://pan.quark.cn/s/f698d0e99a4b
- 打鼾 + 不打鼾语音识别 2 分类数据集 - 夸克网盘下载地址：https://pan.quark.cn/s/4d83dabff0a6

## 开源协议

本项目基于 [MIT License](LICENSE) 开源，可免费用于学习、研究与商业用途。
