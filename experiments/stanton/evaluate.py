"""Fidelity metrics against the cached teacher logits (validation and train-subset only)."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from data import CIFAR100GPU, FixedSet


@torch.no_grad()
def fidelity(model: torch.nn.Module, fixed: FixedSet, teacher_logits: torch.Tensor) -> dict:
    """Top-1 agreement, mean KL(p_teacher || p_student) at temperature 1, label accuracy."""

    student = torch.cat([model(images) for images, _ in fixed.batches(1000)])
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1)
    return {
        "agreement": float((student.argmax(-1) == teacher_logits.argmax(-1)).float().mean()),
        "kl": float(kl.mean()),
        "accuracy": float((student.argmax(-1) == fixed.labels).float().mean()),
        "finite": bool(torch.isfinite(student).all()),
    }


def evaluate(model: torch.nn.Module, data: CIFAR100GPU, teacher_logits: dict) -> dict:
    was_training = model.training
    model.eval()
    row = {}
    for split in ("val", "train_subset"):
        for key, value in fidelity(model, getattr(data, split), teacher_logits[split]).items():
            row[f"{split}_{key}"] = value
    model.train(was_training)
    return row
