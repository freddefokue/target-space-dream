"""Small utilities for moving between named parameters and one flat vector."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ParameterSpec:
    """A deterministic description of a named parameter pytree."""

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    numels: tuple[int, ...]
    total: int

    @classmethod
    def from_params(cls, params: Mapping[str, torch.Tensor]) -> "ParameterSpec":
        names = tuple(params)
        shapes = tuple(params[name].shape for name in names)
        numels = tuple(params[name].numel() for name in names)
        return cls(names, shapes, numels, sum(numels))

    def flatten(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if tuple(params) != self.names:
            raise ValueError("parameter names or ordering changed")
        return torch.cat([params[name].reshape(-1) for name in self.names])

    def unflatten(self, vector: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if vector.numel() != self.total:
            raise ValueError(f"expected {self.total} parameters, got {vector.numel()}")
        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        offset = 0
        for name, shape, numel in zip(self.names, self.shapes, self.numels):
            result[name] = vector[offset : offset + numel].view(shape)
            offset += numel
        return result

