# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import csv
import json
import math
import os
from collections import defaultdict
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate

import time

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None, freeze_backbone: bool = False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
# def evaluate(data_loader, model, device, use_amp=False):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     for batch in metric_logger.log_every(data_loader, 10, header):
#         images = batch[0]
#         target = batch[-1]

#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         if use_amp:
#             with torch.cuda.amp.autocast():
#                 output = model(images)
#                 if isinstance(output, dict):
#                     output = output['logits']
#                 loss = criterion(output, target)
#         else:
#             output = model(images)
#             if isinstance(output, dict):
#                 output = output['logits']
#             loss = criterion(output, target)

#         torch.cuda.synchronize()
        
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def _class_name(class_names, idx):
    """Resolve a class index to its display name, falling back to the index."""
    if class_names is not None and 0 <= idx < len(class_names):
        return str(class_names[idx])
    return str(idx)


def _classification_report_from_confusion(confusion, class_names, focus_class_name="铜瘤"):
    """Compute macro recall and the focus class recall/precision in percent.

    Macro recall averages per-class recall across classes that have at least one
    ground-truth sample. Precision uses zero_division=0 semantics when the model
    never predicts the focus class.
    """
    if confusion is None or confusion.numel() == 0:
        return {}

    confusion = confusion.detach().cpu()
    supports = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    true_positive = confusion.diag()

    valid = supports > 0
    recalls = torch.zeros(confusion.shape[0], dtype=torch.float64)
    recalls[valid] = true_positive[valid].double() / supports[valid].double()
    macro_recall = recalls[valid].mean().item() * 100.0 if valid.any() else 0.0

    report = {"macro_recall": macro_recall}
    labels = [_class_name(class_names, i) for i in range(confusion.shape[0])]
    if focus_class_name in labels:
        focus_idx = labels.index(focus_class_name)
        tp = float(true_positive[focus_idx].item())
        support = float(supports[focus_idx].item())
        predicted_count = float(predicted[focus_idx].item())
        report["copper_recall"] = 100.0 * tp / support if support > 0 else 0.0
        report["copper_precision"] = (
            100.0 * tp / predicted_count if predicted_count > 0 else 0.0
        )

    return report


def _confusion_from_predictions(targets, predictions, num_classes):
    """Build a CPU confusion matrix from flat target/prediction tensors."""
    flat = targets.to(torch.long).cpu() * num_classes + predictions.to(torch.long).cpu()
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _metrics_from_predictions(targets, predictions, class_names, copper_idx):
    """Return classification metrics used by copper operating-point sweeps."""
    num_classes = len(class_names)
    confusion = _confusion_from_predictions(targets, predictions, num_classes)
    report = _classification_report_from_confusion(confusion, class_names)
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    tp = int(confusion[copper_idx, copper_idx].item())
    fp = int(confusion[:, copper_idx].sum().item()) - tp
    fn = int(confusion[copper_idx, :].sum().item()) - tp
    predicted = int(confusion[:, copper_idx].sum().item())
    return {
        "overall_acc1": 100.0 * correct / total if total else 0.0,
        "macro_recall": report["macro_recall"],
        "copper_recall": report["copper_recall"],
        "copper_precision": report["copper_precision"],
        "copper_predicted": predicted,
        "copper_tp": tp,
        "copper_fp": fp,
        "copper_fn": fn,
    }


def _write_copper_operating_point_sweep(
    logits,
    targets,
    class_names,
    output_dir,
    bias_min=0.0,
    bias_max=5.0,
    bias_steps=101,
    threshold_min=0.001,
    threshold_max=0.5,
    threshold_steps=100,
    min_precision=98.0,
    max_acc_drop=0.10,
):
    """Sweep copper-class logit bias and one-vs-rest probability thresholds.

    Bias mode adds a constant to the copper logit before argmax.
    Threshold mode predicts copper when p(copper) >= threshold; otherwise it
    chooses the best non-copper class. Results are written to CSV plus a JSON
    summary containing the best recall point that respects the configured
    precision floor and maximum Acc@1 drop from the unmodified baseline.
    """
    if logits.numel() == 0 or targets.numel() == 0:
        return None

    labels = [str(name) for name in class_names]
    if "铜瘤" not in labels:
        print("WARNING: copper sweep skipped because class label 铜瘤 was not found")
        return None

    copper_idx = labels.index("铜瘤")
    logits = logits.float().cpu()
    targets = targets.long().cpu()
    base_predictions = logits.argmax(dim=1)
    baseline = _metrics_from_predictions(targets, base_predictions, labels, copper_idx)

    rows = []

    def add_row(mode, value, metrics):
        eligible = (
            metrics["copper_precision"] >= min_precision
            and metrics["overall_acc1"] >= baseline["overall_acc1"] - max_acc_drop
        )
        rows.append(
            {
                "mode": mode,
                "value": value,
                **metrics,
                "recall_gain": metrics["copper_recall"] - baseline["copper_recall"],
                "precision_change": metrics["copper_precision"] - baseline["copper_precision"],
                "acc1_change": metrics["overall_acc1"] - baseline["overall_acc1"],
                "eligible": int(eligible),
            }
        )

    add_row("baseline", "", baseline)

    for bias in torch.linspace(float(bias_min), float(bias_max), max(int(bias_steps), 1)):
        adjusted = logits.clone()
        adjusted[:, copper_idx] += float(bias.item())
        predictions = adjusted.argmax(dim=1)
        add_row(
            "logit_bias",
            float(bias.item()),
            _metrics_from_predictions(targets, predictions, labels, copper_idx),
        )

    probabilities = torch.softmax(logits, dim=1)
    copper_probability = probabilities[:, copper_idx]
    non_copper_logits = logits.clone()
    non_copper_logits[:, copper_idx] = float("-inf")
    best_non_copper = non_copper_logits.argmax(dim=1)

    for threshold in torch.linspace(
        float(threshold_min), float(threshold_max), max(int(threshold_steps), 1)
    ):
        threshold_value = float(threshold.item())
        predictions = best_non_copper.clone()
        predictions[copper_probability >= threshold_value] = copper_idx
        add_row(
            "prob_threshold",
            threshold_value,
            _metrics_from_predictions(targets, predictions, labels, copper_idx),
        )

    eligible_rows = [row for row in rows if row["eligible"]]
    recommended = max(
        eligible_rows,
        key=lambda row: (
            row["copper_recall"],
            row["copper_precision"],
            row["macro_recall"],
            row["overall_acc1"],
        ),
        default=rows[0],
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "copper_operating_point_sweep.csv")
    fieldnames = [
        "mode",
        "value",
        "overall_acc1",
        "macro_recall",
        "copper_recall",
        "copper_precision",
        "copper_predicted",
        "copper_tp",
        "copper_fp",
        "copper_fn",
        "recall_gain",
        "precision_change",
        "acc1_change",
        "eligible",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "constraints": {
            "min_copper_precision": min_precision,
            "max_overall_acc1_drop": max_acc_drop,
        },
        "baseline": rows[0],
        "recommended": recommended,
    }
    json_path = os.path.join(output_dir, "copper_operating_point_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote copper operating-point sweep: {csv_path}")
    print(f"Wrote copper operating-point summary: {json_path}")
    print(
        "Copper sweep baseline — Acc@1 {overall_acc1:.3f}% | Recall {copper_recall:.3f}% | "
        "Precision {copper_precision:.3f}%".format(**baseline)
    )
    print(
        "Copper sweep recommended — {mode}={value} | Acc@1 {overall_acc1:.3f}% | "
        "Recall {copper_recall:.3f}% | Precision {copper_precision:.3f}% | "
        "Recall gain {recall_gain:+.3f} pp".format(**recommended)
    )
    return summary


def _display_width(text):
    """Approximate monospace display width (East Asian chars take 2 columns)."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))


def _write_predictions(per_sample, class_names, output_dir):
    """Write per-sample predictions and the misclassified subset to CSV."""
    fieldnames = [
        "index",
        "path",
        "true_label",
        "pred_label",
        "true_prob",
        "pred_prob",
        "margin",
        "correct",
        "top5",
    ]
    rows = [
        {
            **record,
            "true_label": _class_name(class_names, record["true_label"]),
            "pred_label": _class_name(class_names, record["pred_label"]),
        }
        for record in per_sample
    ]
    # Canonical order by global sample position, even after a multi-GPU gather.
    rows.sort(key=lambda row: row["index"])

    predictions_path = os.path.join(output_dir, "eval_predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    misclassified = [row for row in rows if not row["correct"]]
    misclassified_path = os.path.join(output_dir, "eval_misclassified.csv")
    with open(misclassified_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(misclassified)

    print(f"Wrote per-sample predictions: {predictions_path}")
    print(f"Wrote misclassified subset ({len(misclassified)} rows): {misclassified_path}")


def _write_confusion_matrix(confusion, class_names, output_dir):
    """Write count and row-normalized confusion matrices to CSV."""
    num_classes = confusion.shape[0]
    labels = [_class_name(class_names, i) for i in range(num_classes)]

    counts_path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(counts_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + labels)
        for i in range(num_classes):
            writer.writerow([labels[i]] + [int(confusion[i, j].item()) for j in range(num_classes)])

    normalized = torch.zeros_like(confusion, dtype=torch.float)
    for i in range(num_classes):
        row_sum = int(confusion[i].sum().item())
        if row_sum > 0:
            normalized[i] = confusion[i].float() / row_sum * 100.0
    normalized_path = os.path.join(output_dir, "confusion_matrix_normalized.csv")
    with open(normalized_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + labels)
        for i in range(num_classes):
            writer.writerow(
                [labels[i]] + [f"{normalized[i, j].item():.2f}" for j in range(num_classes)]
            )

    print(f"Wrote confusion matrix (counts): {counts_path}")
    print(f"Wrote confusion matrix (row-normalized %): {normalized_path}")


def _print_confusion_summary(confusion, class_names):
    """Print a human-readable confusion matrix and the top confusions."""
    num_classes = confusion.shape[0]
    labels = [_class_name(class_names, i) for i in range(num_classes)]
    width = max([_display_width(label) for label in labels] + [4]) + 2

    def pad(label):
        return str(label) + " " * (width - _display_width(label))

    print("Confusion matrix (rows=true, cols=predicted):")
    print(" " * width + "".join(pad(label) for label in labels))
    for i in range(num_classes):
        row = "".join(f"{int(confusion[i, j].item()):>{width}}" for j in range(num_classes))
        print(pad(labels[i]) + row)

    pairs = [
        (i, j, int(confusion[i, j].item()))
        for i in range(num_classes)
        for j in range(num_classes)
        if i != j and int(confusion[i, j].item()) > 0
    ]
    pairs.sort(key=lambda pair: -pair[2])
    print("\nTop confusions (true -> predicted, count):")
    for i, j, count in pairs[:15]:
        print(f"  {labels[i]} -> {labels[j]}: {count}")


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, output_dir=None,
             save_predictions=False, save_confusion=False, class_names=None,
             num_classes=None, num_tasks=1, global_rank=0,
             copper_sweep=False, copper_bias_min=0.0, copper_bias_max=5.0,
             copper_bias_steps=101, copper_threshold_min=0.001,
             copper_threshold_max=0.5, copper_threshold_steps=100,
             copper_sweep_min_precision=98.0, copper_sweep_max_acc_drop=0.10):
    """Evaluate the model and, when asked, dump diagnostic artifacts.

    ``save_confusion`` writes count and row-normalized confusion matrices and
    prints a summary. ``save_predictions`` writes ``eval_predictions.csv`` (one
    row per sample: path, true/pred class, softmax probabilities, margin =
    true-class probability minus predicted-class probability — negative on
    errors, ~0 on correct — and the top-5) plus ``eval_misclassified.csv``
    (the misclassified subset). The dataset is resolved back to file paths
    through the sampler's index sequence, so it is safe under
    ``SequentialSampler`` and the default ``DistributedSampler`` (each process
    only maps the indices it was given).

    Under multi-GPU evaluation the per-process subsets are gathered and only
    rank 0 writes the artifacts, so the CSVs cover the whole eval set exactly
    once — except when the eval set is not divisible by the process count: the
    padded ``DistributedSampler`` repeats the tail samples, and those repeats
    appear in the artifacts (matching ``main_finetune``'s existing warning that
    distributed eval of a non-divisible set slightly alters results).
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluate:"

    class_stats = defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0})
    model.eval()

    # Confusion counts accumulate in dicts so an unexpected class index can
    # never crash the run; the square tensor is materialized at the end.
    confusion_counts = defaultdict(lambda: defaultdict(int))

    # Eval samplers yield dataset indices in a deterministic order. Materialize
    # that sequence once so each prediction can be traced back to its file.
    dataset = getattr(data_loader, "dataset", None)
    sample_indices = None
    if save_predictions and dataset is not None:
        sampler = getattr(data_loader, "sampler", None)
        if sampler is not None:
            try:
                sample_indices = list(sampler)
            except Exception:
                sample_indices = None
                print("WARNING: could not resolve sampler indices; eval_predictions.csv "
                      "will have no file paths")

    per_sample = []
    sweep_logits = []
    sweep_targets = []
    sample_cursor = 0  # flat position across batches on this process

    start_time = time.time()
    total_images = 0

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output["logits"]
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, dict):
                output = output["logits"]
            loss = criterion(output, target)

        torch.cuda.synchronize()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        total_images += batch_size

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        preds = output.argmax(dim=1)
        for i in range(batch_size):
            confusion_counts[target[i].item()][preds[i].item()] += 1

        if copper_sweep:
            sweep_logits.append(output.detach().float().cpu())
            sweep_targets.append(target.detach().long().cpu())

        if save_predictions:
            probs = torch.softmax(output, dim=1)
            topk_scores, topk_indices = torch.topk(
                probs, k=min(5, probs.shape[1]), dim=1
            )
            pred_prob = topk_scores[:, 0]
            true_prob = probs[torch.arange(batch_size, device=probs.device), target]
            # Margin = how far the true class sat behind (or ahead of) the
            # predicted class: < 0 when misclassified, ~0 when correct.
            margin = true_prob - pred_prob
            for i in range(batch_size):
                path = ""
                if sample_indices is not None:
                    flat_pos = sample_cursor + i
                    if flat_pos < len(sample_indices):
                        global_idx = sample_indices[flat_pos]
                        try:
                            path = dataset.samples[global_idx][0]
                        except (AttributeError, IndexError, TypeError):
                            path = ""
                top5 = " ".join(
                    f"{int(topk_indices[i, k].item())}({float(topk_scores[i, k].item()):.4f})"
                    for k in range(topk_scores.shape[1])
                )
                per_sample.append(
                    {
                        # Global position across the whole eval set: under a
                        # DistributedSampler, rank r sees flat positions
                        # r, r+K, ... so this reconstructs the true index.
                        "index": (sample_cursor + i) * num_tasks + global_rank,
                        "path": path,
                        "true_label": int(target[i].item()),
                        "pred_label": int(preds[i].item()),
                        "true_prob": float(true_prob[i].item()),
                        "pred_prob": float(pred_prob[i].item()),
                        "margin": float(margin[i].item()),
                        "correct": int((preds[i] == target[i]).item()),
                        "top5": top5,
                    }
                )
        sample_cursor += batch_size

    total_time = time.time() - start_time
    images_per_sec = total_images / total_time

    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    print(f"Evaluation speed: {images_per_sec:.2f} images/s")

    # Materialize the square confusion tensor for every evaluation. Size covers
    # both num_classes and any observed class index, so a mismatch degrades to a
    # larger matrix instead of crashing.
    confusion = None
    if confusion_counts:
        observed_max = max(
            max(confusion_counts.keys()),
            max(
                (pred_idx for inner in confusion_counts.values() for pred_idx in inner),
                default=-1,
            ),
        )
        size = max(observed_max + 1, num_classes if num_classes else 0)
        confusion = torch.zeros(size, size, dtype=torch.long, device=device)
        for true_idx, inner in confusion_counts.items():
            for pred_idx, count in inner.items():
                confusion[true_idx, pred_idx] = count

    if copper_sweep:
        local_sweep = {
            "logits": torch.cat(sweep_logits, dim=0) if sweep_logits else torch.empty((0, num_classes or 0)),
            "targets": torch.cat(sweep_targets, dim=0) if sweep_targets else torch.empty((0,), dtype=torch.long),
        }
        if num_tasks > 1 and torch.distributed.is_initialized():
            gathered_sweep = [None] * num_tasks
            torch.distributed.gather_object(
                local_sweep, gathered_sweep if global_rank == 0 else None, dst=0
            )
            if global_rank == 0:
                sweep_logits_all = torch.cat(
                    [part["logits"] for part in gathered_sweep if part is not None], dim=0
                )
                sweep_targets_all = torch.cat(
                    [part["targets"] for part in gathered_sweep if part is not None], dim=0
                )
            else:
                sweep_logits_all = torch.empty((0, num_classes or 0))
                sweep_targets_all = torch.empty((0,), dtype=torch.long)
        else:
            sweep_logits_all = local_sweep["logits"]
            sweep_targets_all = local_sweep["targets"]
    else:
        sweep_logits_all = torch.empty((0, num_classes or 0))
        sweep_targets_all = torch.empty((0,), dtype=torch.long)

    if save_predictions and per_sample and num_tasks > 1 and torch.distributed.is_initialized():
        gathered = [None] * num_tasks
        torch.distributed.gather_object(
            per_sample, gathered if global_rank == 0 else None, dst=0
        )
        if global_rank == 0:
            per_sample = [
                record
                for rank_records in gathered
                for record in (rank_records or [])
            ]
        else:
            per_sample = []
    if confusion is not None and num_tasks > 1 and torch.distributed.is_initialized():
        torch.distributed.all_reduce(confusion)
    if confusion is not None:
        confusion = confusion.cpu()

    # Rebuild per-class statistics from the synchronized confusion matrix so
    # single-GPU and distributed evaluation report exactly the same semantics.
    if confusion is not None:
        class_stats = defaultdict(lambda: {"total": 0, "correct": 0, "accuracy": 0})
        for cls in range(confusion.shape[0]):
            total = int(confusion[cls].sum().item())
            correct = int(confusion[cls, cls].item())
            class_stats[cls]["total"] = total
            class_stats[cls]["correct"] = correct
            class_stats[cls]["accuracy"] = (100.0 * correct / total) if total > 0 else 0.0

    report_stats = _classification_report_from_confusion(confusion, class_names)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    elif save_confusion or save_predictions:
        print("WARNING: --output_dir is empty; confusion/prediction artifacts were not written")

    if global_rank == 0:
        for cls in sorted(class_stats):
            stats = class_stats[cls]
            print(
                f"With {stats['total']} samples, Class "
                f"{_class_name(class_names, cls)}: {stats['accuracy']:.2f}%"
            )

        if "copper_recall" in report_stats:
            print(
                "* Overall Acc@1 {acc1:.3f}% | Macro Recall {macro:.3f}% | "
                "铜瘤 Recall {recall:.3f}% | 铜瘤 Precision {precision:.3f}%".format(
                    acc1=metric_logger.acc1.global_avg,
                    macro=report_stats["macro_recall"],
                    recall=report_stats["copper_recall"],
                    precision=report_stats["copper_precision"],
                )
            )
        else:
            print(
                "* Overall Acc@1 {acc1:.3f}% | Macro Recall {macro:.3f}% | "
                "铜瘤 metrics N/A (class label not found)".format(
                    acc1=metric_logger.acc1.global_avg,
                    macro=report_stats.get("macro_recall", 0.0),
                )
            )

        if save_confusion and confusion is not None and output_dir:
            _write_confusion_matrix(confusion, class_names, output_dir)
            _print_confusion_summary(confusion, class_names)
        if save_predictions and per_sample and output_dir:
            _write_predictions(per_sample, class_names, output_dir)
        if copper_sweep and output_dir:
            _write_copper_operating_point_sweep(
                sweep_logits_all,
                sweep_targets_all,
                class_names,
                output_dir,
                bias_min=copper_bias_min,
                bias_max=copper_bias_max,
                bias_steps=copper_bias_steps,
                threshold_min=copper_threshold_min,
                threshold_max=copper_threshold_max,
                threshold_steps=copper_threshold_steps,
                min_precision=copper_sweep_min_precision,
                max_acc_drop=copper_sweep_max_acc_drop,
            )

    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    eval_stats.update(report_stats)
    return eval_stats, class_stats



def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res