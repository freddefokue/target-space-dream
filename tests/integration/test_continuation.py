import torch

from dream.cli import run_toy
from dream.continuation import ContinuationConfig, track_target_path
from dream.targets import StraightTargetPath


def test_toy_reaches_complete_endpoint() -> None:
    summary = run_toy()
    assert summary["reached_endpoint"]
    assert summary["final_t"] == 1.0
    assert summary["final_metrics"]["pooled_rms"] <= 1e-9


def test_failed_large_interval_is_rolled_back_and_retried() -> None:
    # A nonlinear but exactly realizable scalar family. The direct Newton step
    # from zero to one is deliberately too aggressive; halving must recover.
    initial = torch.tensor([0.0], dtype=torch.float64)

    def output(vector: torch.Tensor) -> torch.Tensor:
        return torch.sinh(vector).view(1, 1)

    path = StraightTargetPath(output(initial), torch.tensor([[3.0]], dtype=torch.float64))
    result = track_target_path(
        output,
        initial,
        path,
        ContinuationConfig(
            initial_step=1.0,
            maximum_step=1.0,
            minimum_step=1.0 / 128.0,
            correctors_per_attempt=8,
            pooled_rms_tolerance=1e-10,
            sample_rms_tolerance=1e-10,
        ),
    )
    assert result.reached_endpoint
    torch.testing.assert_close(output(result.parameters), path.at(1.0), rtol=0, atol=1e-10)

