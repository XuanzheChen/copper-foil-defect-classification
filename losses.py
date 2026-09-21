"""Losses for cost-sensitive copper-foil defect classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedSoftTargetCrossEntropy(nn.Module):
    """Cross entropy for soft targets with explicit per-class costs.

    The loss for one sample is:
        -sum_c class_weights[c] * target[c] * log_softmax(logits)[c]

    This is intentionally averaged over samples without re-normalizing each
    target by its weighted mass: increasing the copper weight should increase
    the optimization cost of copper target mass, including the fractional
    copper mass created by Mixup/CutMix.
    """

    def __init__(self, class_weights):
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError("class_weights must be a 1-D tensor or sequence")
        if torch.any(weights <= 0):
            raise ValueError("all class weights must be > 0")
        self.register_buffer("class_weights", weights)

    def forward(self, logits, target):
        if target.shape != logits.shape:
            raise ValueError(
                "soft target shape must match logits shape: "
                f"got target={tuple(target.shape)} logits={tuple(logits.shape)}"
            )
        log_probs = F.log_softmax(logits, dim=-1)
        weights = self.class_weights.to(
            device=log_probs.device,
            dtype=log_probs.dtype,
        )
        target = target.to(device=log_probs.device, dtype=log_probs.dtype)
        return -(target * weights * log_probs).sum(dim=-1).mean()


class ClassWeightedLabelSmoothingCrossEntropy(nn.Module):
    """Label-smoothed cross entropy with the same per-class cost semantics."""

    def __init__(self, class_weights, smoothing=0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must satisfy 0 <= smoothing < 1")
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError("class_weights must be a 1-D tensor or sequence")
        if torch.any(weights <= 0):
            raise ValueError("all class weights must be > 0")
        self.register_buffer("class_weights", weights)
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        if target.ndim != 1:
            raise ValueError("hard target must be a 1-D class-index tensor")
        num_classes = logits.shape[-1]
        if num_classes != self.class_weights.numel():
            raise ValueError(
                f"class weight count {self.class_weights.numel()} != logits classes {num_classes}"
            )

        with torch.no_grad():
            soft_target = torch.full_like(
                logits,
                self.smoothing / num_classes,
                dtype=logits.dtype,
            )
            soft_target.scatter_(
                1,
                target.unsqueeze(1),
                (1.0 - self.smoothing) + self.smoothing / num_classes,
            )

        log_probs = F.log_softmax(logits, dim=-1)
        weights = self.class_weights.to(
            device=log_probs.device,
            dtype=log_probs.dtype,
        )
        return -(soft_target * weights * log_probs).sum(dim=-1).mean()


class ClassWeightedCrossEntropy(nn.Module):
    """Hard-label class-weighted CE using sample-mean cost semantics."""

    def __init__(self, class_weights):
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError("class_weights must be a 1-D tensor or sequence")
        if torch.any(weights <= 0):
            raise ValueError("all class weights must be > 0")
        self.register_buffer("class_weights", weights)

    def forward(self, logits, target):
        target = target.to(device=logits.device)
        losses = F.cross_entropy(logits, target, reduction="none")
        weights = self.class_weights.to(device=logits.device)
        sample_weights = weights[target].to(dtype=losses.dtype)
        return (losses * sample_weights).mean()
