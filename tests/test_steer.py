"""Tests for the DeltaLoss steering ranker (autogrid_ext.steer)."""
import torch

from autogrid_ext.steer import delta_loss_weight, deltaloss_linear


def _quant_add(e):
    def quant(t):
        return t + e
    return quant


def test_zero_perturbation_is_zero():
    w = torch.randn(4, 16)
    grad = torch.randn(4, 16)
    assert delta_loss_weight(w, w.clone(), grad) == 0.0


def test_deltaloss_linear_matches_closed_form():
    torch.manual_seed(0)
    w = torch.randn(3, 5)                        # (out, in)
    x = torch.randn(2, 5)                        # (tokens, in)
    e = 0.01 * torch.randn_like(w)
    wq = w + e
    y = x @ wq.T
    t = x @ w.T
    g = (2.0 / y.numel()) * ((y - t).T @ x)      # dL/dwq for MSE, shape (3, 5)
    expected = float((g * e).abs().sum())
    got = deltaloss_linear(w, x, _quant_add(e))
    assert abs(got - expected) / (abs(expected) + 1e-12) < 1e-5


def test_deltaloss_grows_with_perturbation():
    torch.manual_seed(1)
    w = torch.randn(4, 16)
    x = torch.randn(8, 16)
    small = deltaloss_linear(w, x, _quant_add(0.001 * torch.ones_like(w)))
    big = deltaloss_linear(w, x, _quant_add(0.1 * torch.ones_like(w)))
    assert big > small > 0.0
