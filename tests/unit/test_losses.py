import torch

from dream.losses import softmax_kl, softmax_kl_logit_gradient


def test_kl_gradient_matches_autograd() -> None:
    torch.manual_seed(0)
    target = 3.0 * torch.randn(7, 11, dtype=torch.float64)
    student = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    (autograd,) = torch.autograd.grad(softmax_kl(target, student), student)
    torch.testing.assert_close(softmax_kl_logit_gradient(target, student.detach()), autograd,
                               rtol=1e-12, atol=1e-14)


def test_kl_is_zero_at_target_and_invariant_to_logit_offsets() -> None:
    torch.manual_seed(1)
    logits = torch.randn(5, 9, dtype=torch.float64)
    assert float(softmax_kl(logits, logits + 2.5)) < 1e-14
    assert float(softmax_kl(logits, torch.zeros_like(logits))) > 0.0
