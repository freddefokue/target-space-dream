"""Softmax distillation losses in logit space."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def softmax_kl(target_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ``KL(softmax(target) || softmax(student))`` in nats."""

    target_log_probs = F.log_softmax(target_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    per_example = (target_log_probs.exp() * (target_log_probs - student_log_probs)).sum(-1)
    return per_example.mean()


def softmax_kl_logit_gradient(target_logits: torch.Tensor,
                              student_logits: torch.Tensor) -> torch.Tensor:
    """Gradient of :func:`softmax_kl` with respect to the student logits: ``(p - q) / batch``."""

    difference = F.softmax(student_logits, dim=-1) - F.softmax(target_logits, dim=-1)
    return difference / student_logits.shape[0]
