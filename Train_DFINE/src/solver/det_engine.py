"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

[2026-10-08 本地补丁 v2] 误杀修复：v1 的 nonpositive w/h 检查是错误的——D-FINE 的框由
  ltrb 分布精修得到，健康模型中后期输出个别 w/h<=0 的回归框属正常现象（实测 ep46 权重
  61/61 batch 含此类框，且全部 finite）。box_ops.box_cxcywh_to_xyxy 对 w/h 有 clamp(min=0)，
  matcher/criterion/loss 全链路都能安全处理（官方代码从不检查此项）。v1 导致健康训练被
  连续 31 batch 误判"发散"终止（8 个数据集反复"快退发散"的真凶）。
  v2 行为：只防真正的 NaN/Inf 输出（AMP 真溢出场景），w/h<=0 一律放行。
  1. pred_boxes 含 nan/inf -> 跳过该 batch（不反传、不 step、不更新 EMA）；
  2. loss 非有限值 -> 跳过该 batch；
  3. 梯度裁剪后仍含 nan/inf -> 跳过 step；
  4. 连续跳过 > 30 个 batch -> 判定真发散，抛错终止。
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    # ---- [nan-guard] 防护状态 ----
    nan_skips = 0          # 本轮累计跳过
    consecutive_nan = 0    # 连续跳过（真发散判定）

    def _skip_batch(msg: str):
        """跳过坏 batch：清空梯度，不反传、不 step、不更新 EMA。"""
        nonlocal nan_skips, consecutive_nan
        nan_skips += 1
        consecutive_nan += 1
        optimizer.zero_grad(set_to_none=True)
        print("[nan-guard] skip batch ({}/{} consecutive): {}".format(nan_skips, consecutive_nan, msg))
        if consecutive_nan > 30:
            raise RuntimeError("[nan-guard] 连续 {} 个 batch 全为 nan/inf，判定训练发散，终止".format(consecutive_nan))

    def _note_bad(msg: str):
        """梯度异常只记录+熔断，不 zero_grad、不 continue：
        交给 scaler.step 检测 found_inf 自动跳过本次更新，scaler.update 自动回退 scale。"""
        nonlocal nan_skips, consecutive_nan
        nan_skips += 1
        consecutive_nan += 1
        print("[nan-guard] {} (skip count {}/{} consecutive)".format(msg, nan_skips, consecutive_nan))
        if consecutive_nan > 30:
            raise RuntimeError("[nan-guard] 连续 {} 个 batch 全为 nan/inf，判定训练发散，终止".format(consecutive_nan))
    # -----------------------------

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            # [nan-guard] 输出框含 nan/inf 会炸 GiOU matcher -> 跳过
            # （v2：不再检查 w/h<=0，box_cxcywh_to_xyxy 有 clamp，健康模型也会输出此类框）
            pb = outputs["pred_boxes"]
            if not torch.isfinite(pb).all():
                _skip_batch("pred_boxes contains nan/inf or non-positive w/h")
                continue

            with torch.autocast(device_type=str(device), enabled=False):
                try:
                    loss_dict = criterion(outputs, targets, **metas)
                except AssertionError as e:
                    _skip_batch("criterion AssertionError: {}".format(e))
                    continue

            loss = sum(loss_dict.values())

            # [nan-guard] loss 异常 -> 跳过
            if not torch.isfinite(loss):
                _skip_batch("loss={} is not finite".format(float(loss)))
                continue

            scaler.scale(loss).backward()

            batch_ok = True
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                # [nan-guard] 裁剪后梯度仍异常：只记录，不手动 continue。
                # 手动 continue 会绕过 AMP 的动态 scale 回退（scaler.update 永远执行不到，
                # scale 卡死导致每个 batch 都溢出）；scaler.step 检测 found_inf 会自动跳过更新。
                grads_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad
                )
                if not grads_finite:
                    _note_bad("grads contain nan/inf (scaler will skip step)")
                    batch_ok = False

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)

            # [nan-guard] 输出框异常 -> 跳过（v2：仅 nan/inf，w/h<=0 放行）
            pb = outputs["pred_boxes"]
            if not torch.isfinite(pb).all():
                _skip_batch("pred_boxes contains nan/inf or non-positive w/h")
                continue

            try:
                loss_dict = criterion(outputs, targets, **metas)
            except AssertionError as e:
                _skip_batch("criterion AssertionError: {}".format(e))
                continue

            loss: torch.Tensor = sum(loss_dict.values())

            # [nan-guard] loss 异常 -> 跳过
            if not torch.isfinite(loss):
                _skip_batch("loss={} is not finite".format(float(loss)))
                continue

            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # [nan-guard] 正常走完一个 batch，重置连续计数（坏梯度 batch 由 batch_ok=False 控制）
        if 'batch_ok' not in dir() and 'batch_ok' not in locals():
            batch_ok = True
        if batch_ok:
            consecutive_nan = 0

        # ema（梯度异常被 scaler 跳过更新的 batch 不更新 EMA，防止权重被污染）
        if batch_ok and ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if nan_skips:
        print("[nan-guard] Epoch {} skipped {} bad batches in total (training continued)".format(epoch, nan_skips))

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
