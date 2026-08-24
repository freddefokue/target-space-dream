import torch

from dream.targets import StraightTargetPath, centered_logits, helmert_contrast


def test_centered_logits_remove_only_common_offset() -> None:
    logits = torch.tensor([[1.0, 2.0, 4.0], [-3.0, 0.0, 8.0]], dtype=torch.float64)
    shifted = logits + torch.tensor([[9.0], [-2.0]], dtype=torch.float64)
    torch.testing.assert_close(centered_logits(logits), centered_logits(shifted))
    torch.testing.assert_close(centered_logits(logits).mean(-1), torch.zeros(2, dtype=torch.float64))


def test_helmert_basis_preserves_centered_distance() -> None:
    torch.manual_seed(0)
    logits = torch.randn(7, 10, dtype=torch.float64)
    q = helmert_contrast(10, dtype=torch.float64)
    centered = centered_logits(logits)
    reconstructed = (logits @ q) @ q.transpose(0, 1)
    torch.testing.assert_close(reconstructed, centered)


def test_straight_path_endpoints_and_constant_velocity() -> None:
    initial = torch.tensor([1.0, 2.0])
    final = torch.tensor([5.0, -2.0])
    path = StraightTargetPath(initial, final)
    torch.testing.assert_close(path.at(0.0), initial)
    torch.testing.assert_close(path.at(1.0), final)
    torch.testing.assert_close(path.at(0.75) - path.at(0.25), 0.5 * path.velocity)

