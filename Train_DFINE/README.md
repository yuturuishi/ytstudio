# D-FINE 训练 · 导出 · 部署 一体化手册

> 基于 [Peterande/D-FINE](https://github.com/Peterande/D-FINE)（Apache-2.0，可商用）本地化改造的训练与导出工程。
> 本文件取代上游原版 `README_org.md`，聚焦 **从零训练自有数据集 → 导出 ONNX → 转 OpenVINO / TensorRT** 的可复现链路。

---

## 0. 目录结构（核心）

```
Train_DFINE/
├─ train.py                     # 训练/评测入口
├─ configs/dfine/               # 模型配置（dfine_hgnetv2_{n,s,m,l,x}_coco.yml + include/ + scenes/）
├─ configs/dataset/             # 数据集模板
├─ tools/
│  ├─ yolo2coco.py              # YOLO 格式 -> COCO json + dataset.yml（批量转换）
│  ├─ deployment/               # ★ 导出三件套
│  │  ├─ export_onnx.py         #   .pth -> ONNX（动态 opset17）
│  │  ├─ export_openvino.py     #   ONNX -> OpenVINO IR（静态 fp32 默认）
│  │  └─ export_tensorrt.py     #   ONNX -> TensorRT engine（静态 fp32 默认）
│  ├─ inference/                # ★ 推理验证：onnx_inf / openvino_inf / trt_inf
│  └─ train_all_100.py          # 多场景批量训练编排器
├─ tests/                       # 官方 COCO 五档基准脚本 + bus.jpg 测试图
├─ src/                         # 模型代码
├─ weight/hgnetv2/              # HGNetv2-B0 骨干预训练（本地，训练不联网）
└─ models/                      # ★ 权重与导出模型（见"命名规则"附录，不入库）
```

命令均在工程根目录执行，解释器 `python`（已预置 Python 3.12）。

---

## 1. 安装依赖

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt                         # 训练+评测
pip install onnx==1.23.1 onnxruntime==1.30.0 onnxsim==0.7.3   # ONNX 导出与验证
pip install openvino==2026.4.1                          # OpenVINO 转换（>=2024.6 均可）
# TensorRT 导出需在带 GPU 的部署机：pip install tensorrt
```
> 已验证：torch 2.11.0+cu128 / torchvision 0.26.0 / openvino 2026.4.1 / onnx 1.23.1 / onnxruntime 1.30.0 / numpy 2.5.3。

---

## 2. 构建数据集

D-FINE 用 **COCO 格式**：图片目录 + `instances_train.json` / `instances_val.json` + 一份 `dataset.yml`。

**方案 A — YOLO 批量转**（维护者工具）：`tools/yolo2coco.py` 扫描 `E:\datasets`、`E:\datasets\datasets_yolo26` 下 `detect_*` 数据集，生成 `E:\datasets\datasets_dfine/<name>/annotations/*.json` 与 `dataset.yml`（图片不复制，引用原路径）。改脚本顶部 `SRC_ROOTS` / `OUT_ROOT` 适配你的路径即可。

**方案 B — 手写 COCO**（通用可移植）：整理 COCO json（`category_id` **0 起始连续**），再写 `dataset.yml`：
```yaml
task: detection
num_classes: 2                      # 你的类别数
remap_mscoco_category: False        # 自有数据集必须为 False
evaluator: { type: CocoEvaluator, iou_types: ['bbox',] }
train_dataloader:
  type: DataLoader
  dataset: { type: CocoDetection, img_folder: /abs/train/images,
             ann_file: /abs/train/instances_train.json, return_masks: False, transforms: { type: Compose, ops: ~ } }
  shuffle: True; num_workers: 4; drop_last: True; collate_fn: { type: BatchImageCollateFunction }
val_dataloader:   # 同上，shuffle:False drop_last:False
```
**两条铁律**：① `category_id` 必须 0 起始、连续；② `remap_mscoco_category` 必须为 `False`。

**创建场景训练配置**：复制官方基准配置改写 `num_classes` 与数据集指向，得到 `configs/dfine/scenes/dfine_hgnetv2_n100_<tag>.yml`（参考现有 `dfine_hgnetv2_n100_fire_smoke.yml`）：
```yaml
__include__: ['/abs/your/dataset.yml', '../../runtime.yml', '../include/dataloader.yml',
              '../include/optimizer.yml', '../include/dfine_hgnetv2.yml']
output_dir: ./output/n_<tag>
DFINE: { backbone: HGNetv2 }
HGNetv2: { name: 'B0', return_idx: [2,3], freeze_at: -1, freeze_norm: False, use_lab: True }
epochs: 100
train_dataloader: { total_batch_size: 32, dataset: { transforms: { policy: { epoch: 93 } } },
                    collate_fn: { stop_epoch: 93, ema_restart_decay: 0.9999 } }
val_dataloader: { total_batch_size: 64 }
```

---

## 3. 训练

**单数据集（单卡）**
```bash
python train.py -c configs/dfine/scenes/dfine_hgnetv2_n100_fire_smoke.yml --use-amp --seed=0
python train.py -c <cfg> -r output/n_<tag>/last.pth --seed=0          # 断点续训
python train.py -c <cfg> -r output/n_<tag>/best_stg2.pth --test-only   # 只评测
```
输出落在配置 `output_dir`：`last.pth` / `best_stg1.pth` / `best_stg2.pth` / `log.txt`（JSON-lines，含 `test_coco_eval_bbox`）。
多卡：`torchrun --nproc_per_node=N train.py -c <cfg> --use-amp`。

**批量训练**：`tools/train_all_100.py` 自动生成场景配置 → 训练（100 轮）→ 解析最优 mAP → 归档 `.pth` + 导出 `.onnx`。
```bash
python tools\train_all_100.py                 # 全部数据集
python tools\train_all_100.py detect_fire_smoke   # 指定集
```
关键常量：`DATA_ROOT`(默认 `E:/datasets/datasets_dfine`)、`NO_AMP`(fp16 易发散的数据集强制 fp32)、`EPOCHS=100`、`ARCH=dfine_n`。

**⚠️ D-FINE 两个训练坑**
1. **fp16 AMP 溢出**：部分数据集 `--use-amp` 下中间值溢出、反复发散（mAP 上不去/loss 炸）。不确定时**不加 `--use-amp` 用 fp32 最稳**。`src/solver/det_engine.py` 已内置 **nan-guard v2**：只拦截真正 NaN/Inf，**不再误杀健康训练**。
2. **坏档续训**：强杀进程可能写坏 `last.pth`。修复：用同目录 `best_stg1.pth` 替换 `last.pth` 再续。

---

## 4. 测试（推理验证）

输入预处理统一：**resize 到 640×640 正方形 + `[0,1]` 归一化（无 ImageNet 均值方差）**。输出结果图。

```bash
python tools\inference\onnx_inf.py --onnx models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx --input tests/bus.jpg
python tools\inference\openvino_inf.py -ov_model models/d-fine-openvino2024.6-static-fp32/dfine_n_coco_ov2024.6_fp32_static.xml -image tests/bus.jpg
python tools\inference\trt_inf.py -trt models/d-fine-tensorrt_trt10-static-fp32/dfine_n_coco_trt10_fp32_static.engine -i tests/bus.jpg -d cuda:0
```

---

## 5. 转换 ONNX

脚本 `tools/deployment/export_onnx.py`。**opset 17 + 动态 batch + `dynamo=False`**（缺一不可，否则精度静默崩坏或导出双文件）。

```bash
python tools\deployment\export_onnx.py n                 # 官方单档
python tools\deployment\export_onnx.py n s m l x         # 五档（逐档子进程）
# 自定义权重：
python tools\deployment\export_onnx.py --config configs/dfine/scenes/dfine_hgnetv2_n100_fire_smoke.yml --resume output/n_fire_smoke/best_stg2.pth --variant n --dataset-tag fire_smoke
```
**产物**：`models/d-fine-onnx-dyn-opset17/dfine_{variant}_{dataset}_opset17.onnx`，导出后自动 ORT 冒烟（与 PyTorch 比对，scores 差 ~1e-6）。

---

## 6. ONNX 转 OpenVINO

脚本 `tools/deployment/export_openvino.py`。默认 **静态 fp32**（rebekah 内置 OV 2024.6 对动态 batch 编译会出错）。

```bash
python tools\deployment\export_openvino.py models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx
python tools\deployment\export_openvino.py <onnx> --fp16   # 仅 CPU / 2026.4.1+ 运行时才建议
```
**产物**：`models/d-fine-openvino2024.6-static-fp32/dfine_{variant}_{dataset}_ov2024.6_fp32_static.{xml,bin}`（`--fp16` → `_fp16_static`）。

---

## 7. ONNX 转 TensorRT

脚本 `tools/deployment/export_tensorrt.py`（需 **GPU + 已装 tensorrt**）。默认 **静态 fp32**（同 fp16 溢出风险）。

```bash
python tools\deployment\export_tensorrt.py models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx
python tools\deployment\export_tensorrt.py <onnx> --fp16 --workspace 4   # 先验证框正确再开 fp16
```
**产物**：`models/d-fine-tensorrt_trt10-static-fp32/dfine_{variant}_{dataset}_trt10_fp32_static.engine`（engine **不可跨 TRT 版本 / 跨 GPU 移植**）。

或用 NVIDIA `trtexec`（只需 trtexec 在 PATH）：
```bash
trtexec --onnx=models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx \
  --saveEngine=models/d-fine-tensorrt_trt10-static-fp32/dfine_n_coco_trt10_fp32_static.engine \
  --explicitBatch --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2   # fp16 追加 --fp16
```

---

## 附 A：模型输入输出签名（三运行时通用）

| 项 | 说明 |
|---|---|
| 输入 `images` | float32 `[N,3,640,640]`，RGB、`[0,1]` 归一化（无均值方差） |
| 输入 `orig_target_sizes` | int64 `[N,2]`，**`[W, H]`（宽在前）** |
| 输出 `labels` / `boxes` / `scores` | int64 `[N,300]` / float32 `[N,300,4]`(xyxy 原图坐标) / float32 `[N,300]` |
| 后处理 | **无 NMS**，按 score 阈值过滤 top-300 |

> `orig_target_sizes` 必须是 `[W, H]`：后处理对框做 `xyxy × [W,H,W,H]`，顺序错则非方形图上 x/y 错位。resize 成 640×640 时 W==H 无影响。

## 附 B：模型命名规则

| 目录 | 文件名 | 含义 |
|---|---|---|
| `models/` | `dfine_{n,s,m,l,x}_coco.pth` | PyTorch 权重（转换源头） |
| `d-fine-onnx-dyn-opset17/` | `dfine_{v}_{ds}_opset17.onnx` | ONNX 动态 opset17 |
| `d-fine-openvino2024.6-static-fp32/` | `dfine_{v}_{ds}_ov2024.6_fp32_static.{xml,bin}` | OV IR 静态 fp32（2024.6 为目标运行时） |
| `d-fine-tensorrt_trt10-static-fp32/` | `dfine_{v}_{ds}_trt10_fp32_static.engine` | TRT 引擎静态 fp32（trt10 为主版本） |

`{v}`=档位 n/s/m/l/x；`{ds}`=数据集标签（COCO 为 `coco`，自定义为你的 `--dataset-tag`）。

## 附 C：许可

- D-FINE 官方代码与 COCO 权重：**Apache-2.0**，可商用（Ultralytics 系 AGPL 不适用）。
- 自己训练、用 D-FINE 架构的权重：对外标「自有」（运行时不含 ultralytics 库代码）。
- 上游原版说明见 `README_org.md`。
