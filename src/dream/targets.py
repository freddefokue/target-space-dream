"""Output representations and target-space paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove the unidentifiable common offset from every logit vector."""

    return logits - logits.mean(dim=-1, keepdim=True)


def helmert_contrast(classes: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return an orthonormal basis for the centered-logit subspace.

    The returned matrix has shape ``[classes, classes - 1]``. Multiplication by
    it preserves Euclidean distances between centered logit vectors while
    removing their redundant all-ones direction.
    """

    if classes < 2:
        raise ValueError("classes must be at least two")
    q = torch.zeros(classes, classes - 1, device=device, dtype=dtype)
    for column in range(classes - 1):
        width = column + 1
        scale = (width * (width + 1)) ** -0.5
        q[:width, column] = scale
        q[width, column] = -width * scale
    return q


@dataclass(frozen=True)
class StraightTargetPath:
    """The constant-speed output path from an initial student to a teacher."""

    initial: torch.Tensor
    final: torch.Tensor

    def __post_init__(self) -> None:
        if self.initial.shape != self.final.shape:
            raise ValueError("path endpoints must have the same shape")

    def at(self, t: float) -> torch.Tensor:
        if not 0.0 <= float(t) <= 1.0:
            raise ValueError("t must lie in [0, 1]")
        return torch.lerp(self.initial, self.final, float(t))

    @property
    def velocity(self) -> torch.Tensor:
        return self.final - self.initial

