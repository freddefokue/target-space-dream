"""Schedules for the target parameter t and the teacher temperature T, as functions of the budget
fraction ``s`` in [0, 1] (PREREGISTRATION.md §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def fixed_t(s: float, phi: float) -> float:
    """t rises linearly from 0 to 1 over the first fraction ``phi`` of the budget."""

    if phi <= 0.0:
        return 1.0
    return min(1.0, max(0.0, s / phi))


def geometric_temperature(s: float, initial: float, phi: float) -> float:
    """T decreases geometrically from ``initial`` to 1 over the first fraction ``phi``."""

    if phi <= 0.0 or s >= phi:
        return 1.0
    return initial ** (1.0 - max(0.0, s) / phi)


def annealing_kd_temperature(s: float, tau_max: int, phi: float) -> int:
    """Annealing-KD's integer temperature: tau_max, ..., 1 in equal stages over ``phi``, then 1."""

    if phi <= 0.0 or s >= phi:
        return 1
    stage = int(math.floor(max(0.0, s) / (phi / tau_max)))
    return max(1, tau_max - stage)


def annealing_kd_factor(temperature: float, tau_max: int) -> float:
    """``Phi(T) = 1 - (T - 1) / tau_max`` (Jafari et al., 2021)."""

    return 1.0 - (temperature - 1.0) / tau_max


def warmup_cosine(s: float, peak: float, floor: float, warmup: float = 0.0) -> float:
    """Linear warmup from 0 to ``peak`` over ``warmup``, then cosine from ``peak`` to ``floor``."""

    if warmup > 0.0 and s < warmup:
        return peak * s / warmup
    progress = (s - warmup) / (1.0 - warmup) if warmup < 1.0 else 1.0
    progress = min(1.0, max(0.0, progress))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


@dataclass
class AdaptiveT:
    """Dream-style step control of t from a monitor KL (PREREGISTRATION.md §6, Amendment 1).

    Call :meth:`check` every checking interval with the monitor ``KL(q_t || p_theta)`` and the
    budget fraction, and :meth:`value` whenever the current t is needed.
    """

    threshold: float
    step: float = 1.0 / 64.0
    step_cap: float = 1.0 / 8.0
    step_floor: float = 1.0 / 1024.0
    patience: int = 3
    deadline: float = 0.6
    ramp_end: float = 0.7
    t: float = 0.0
    advances_in_row: int = 0
    fails_in_row: int = 0
    reference_kl: float | None = None
    ramp_from: float | None = None
    events: list[dict] = field(default_factory=list)

    def value(self, s: float) -> float:
        if self.ramp_from is None and self.t < 1.0 and s >= self.deadline:
            self.ramp_from = self.t
            self.events.append({"reason": "deadline", "fraction": s, "t_before": self.t,
                                "t_after": "linear ramp to 1 by %.2f" % self.ramp_end,
                                "dt": self.step, "kl": None, "forced": True})
        if self.ramp_from is not None:
            span = self.ramp_end - self.deadline
            progress = min(1.0, max(0.0, (s - self.deadline) / span))
            self.t = min(1.0, self.ramp_from + (1.0 - self.ramp_from) * progress)
        return self.t

    @property
    def controlling(self) -> bool:
        return self.t < 1.0 and self.ramp_from is None

    def check(self, kl: float, s: float) -> dict | None:
        """Apply one monitor check; return the advance event, if any."""

        self.value(s)
        if not self.controlling:
            return None
        reason = None
        if kl < self.threshold:
            reason = "threshold"
        elif self.reference_kl is not None and kl <= 0.5 * self.reference_kl:
            reason = "relative"
        elif self.reference_kl is None:
            self.reference_kl = kl
        if reason is None:
            self.advances_in_row = 0
            self.fails_in_row += 1
            if self.fails_in_row >= self.patience:
                self.step = max(self.step * 0.5, self.step_floor)
                self.fails_in_row = 0
            return None
        before = self.t
        self.t = min(1.0, self.t + self.step)
        event = {"reason": reason, "fraction": s, "t_before": before, "t_after": self.t,
                 "dt": self.step, "kl": kl, "reference_kl": self.reference_kl,
                 "forced": reason != "threshold"}
        self.events.append(event)
        self.fails_in_row = 0
        self.reference_kl = None
        self.advances_in_row += 1
        if self.advances_in_row >= 2:
            self.step = min(self.step * 2.0, self.step_cap)
            self.advances_in_row = 0
        return event

    def state_dict(self) -> dict:
        return dict(self.__dict__)

    def load_state_dict(self, state: dict) -> None:
        self.__dict__.update(state)


def snapshot_index(t: float, intervals: int) -> int:
    """Index of the target snapshot for a path position ``t`` in [0, 1] over ``intervals`` steps."""

    return min(intervals, int(math.floor(t * intervals + 1e-6)))


def snapshot_controller(threshold: float, intervals: int) -> AdaptiveT:
    """The A2 adaptive rule in units of one snapshot (PREREGISTRATION.md §6, A6)."""

    unit = 1.0 / intervals
    return AdaptiveT(threshold=threshold, step=unit, step_cap=math.ceil(intervals / 8) * unit,
                     step_floor=unit)
