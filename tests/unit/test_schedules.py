import pytest
import torch

from dream.schedules import (AdaptiveT, annealing_kd_factor, annealing_kd_temperature, fixed_t,
                             geometric_temperature, warmup_cosine)
from dream.targets import StraightTargetPath


def test_moving_target_endpoints() -> None:
    torch.manual_seed(0)
    initial, teacher = torch.randn(4, 100), torch.randn(4, 100)
    path = StraightTargetPath(initial, teacher)
    assert torch.equal(path.at(0.0), initial)
    assert torch.equal(path.at(1.0), teacher)
    assert fixed_t(0.0, 0.5) == 0.0
    assert fixed_t(0.25, 0.5) == pytest.approx(0.5)
    assert fixed_t(0.5, 0.5) == 1.0 and fixed_t(0.9, 0.5) == 1.0


def test_temperature_path_endpoints() -> None:
    assert geometric_temperature(0.0, 16.0, 0.5) == pytest.approx(16.0)
    assert geometric_temperature(0.25, 16.0, 0.5) == pytest.approx(4.0)
    assert geometric_temperature(0.5, 16.0, 0.5) == 1.0
    assert geometric_temperature(0.99, 4.0, 0.5) == 1.0


def test_annealing_kd_schedule() -> None:
    assert annealing_kd_temperature(0.0, 10, 0.5) == 10
    assert annealing_kd_temperature(0.049, 10, 0.5) == 10
    assert annealing_kd_temperature(0.05, 10, 0.5) == 9
    assert annealing_kd_temperature(0.4999, 10, 0.5) == 1
    assert annealing_kd_temperature(0.8, 10, 0.5) == 1
    assert annealing_kd_factor(10, 10) == pytest.approx(0.1)
    assert annealing_kd_factor(1, 10) == 1.0


def test_warmup_cosine() -> None:
    assert warmup_cosine(0.0, 0.05, 1e-6, warmup=0.1) == 0.0
    assert warmup_cosine(0.05, 0.05, 1e-6, warmup=0.1) == pytest.approx(0.025)
    assert warmup_cosine(0.1, 0.05, 1e-6, warmup=0.1) == pytest.approx(0.05)
    assert warmup_cosine(1.0, 0.05, 1e-6, warmup=0.1) == pytest.approx(1e-6)
    assert warmup_cosine(0.0, 0.05, 1e-6) == pytest.approx(0.05)


def test_adaptive_t_threshold_doubling_relative_halving() -> None:
    control = AdaptiveT(threshold=0.1, step=1 / 64)
    control.check(0.0, 0.00)                     # below threshold: advance
    assert control.t == pytest.approx(1 / 64)
    control.check(0.05, 0.01)                    # second advance in a row: step doubles
    assert control.t == pytest.approx(2 / 64) and control.step == pytest.approx(2 / 64)
    control.check(0.8, 0.02)                     # above: records the reference KL
    assert control.reference_kl == pytest.approx(0.8) and control.t == pytest.approx(2 / 64)
    event = control.check(0.4, 0.03)             # halved since the last advance: forced advance
    assert event["reason"] == "relative" and event["forced"]
    assert control.t == pytest.approx(4 / 64)
    for fraction in (0.04, 0.05, 0.06):          # three failed checks: step halves
        control.check(0.9, fraction)
    assert control.step == pytest.approx(1 / 64)


def test_adaptive_t_deadline_ramp() -> None:
    control = AdaptiveT(threshold=1e-9)
    control.check(1.0, 0.1)
    assert control.t == 0.0
    assert control.value(0.6) == 0.0
    assert control.value(0.65) == pytest.approx(0.5)
    assert control.value(0.7) == 1.0 and control.value(0.9) == 1.0
    assert control.check(0.0, 0.8) is None
    deadline = [event for event in control.events if event["reason"] == "deadline"]
    assert len(deadline) == 1 and deadline[0]["forced"]


def test_snapshot_controller_moves_in_whole_snapshots() -> None:
    from dream.schedules import snapshot_controller, snapshot_index

    control = snapshot_controller(threshold=0.1, intervals=40)
    assert snapshot_index(control.t, 40) == 0
    control.check(0.0, 0.0)
    assert snapshot_index(control.t, 40) == 1
    control.check(0.0, 0.01)                     # two advances: step doubles to 2 snapshots
    control.check(0.0, 0.02)
    assert snapshot_index(control.t, 40) == 4
    for _ in range(4):                           # 4 -> 6 -> 10 -> 14 -> 19; step 2 -> 4 -> 5 (cap)
        control.check(0.0, 0.03)
    assert snapshot_index(control.t, 40) == 19
    assert control.step == pytest.approx(5 / 40)
    for fraction in (0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18):
        control.check(9.0, fraction)             # failures halve the step, never below 1
    assert control.step == pytest.approx(1 / 40)
    assert snapshot_controller(0.1, 20).step_cap == pytest.approx(3 / 20)
    assert snapshot_index(1.0, 40) == 40 and snapshot_index(3 * (1 / 40), 40) == 3
