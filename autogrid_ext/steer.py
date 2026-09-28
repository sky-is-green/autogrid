"""DeltaLoss steering ranker (SignRoundV2, arXiv 2512.04746 sections 3.1-3.2).

AUTOGRID classifies every tensor FREE / TERNARY / STEER and deliberately
refuses STEER tensors: they need a steered pipeline (rotation, corrections,
reconstruction).  DeltaLoss is a candidate way to *rank* which STEER tensors
get precision or corrections first.  It combines the quantisation
perturbation with the task-aware gradient on a small calibration set:

    DeltaLoss = || g_w * (W_q - W) ||_1        (+ an activation term in the paper)

Weight-space form here: ``g_w`` comes from an output-reconstruction loss on
the quantized weights (``MSE(x @ Wq.T, x @ W.T)``), so no labels are needed
and the metric works per tensor with the deployed container's quantizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def delta_loss_weight(w: torch.Tensor, wq: torch.Tensor,
                      grad: torch.Tensor) -> float:
    """Core term: L1 norm of gradient times quantisation perturbation."""
    return float((grad * (wq - w)).abs().sum())


def deltaloss_linear(weight: torch.Tensor, x: torch.Tensor, quant_fn) -> float:
    """DeltaLoss of a linear weight given calibration inputs.

    ``quant_fn`` maps the full-precision weight to the deployed container
    (any callable ``Tensor -> Tensor`` of the same shape).  Gradients are
    taken with respect to the quantized weight, matching the paper's g_wq.
    """
    w = weight.detach().float()
    x = x.detach().float()
    wq = quant_fn(w).detach().requires_grad_(True)
    y = F.linear(x, wq)
    with torch.no_grad():
        target = F.linear(x, w)
    loss = F.mse_loss(y, target)
    (grad,) = torch.autograd.grad(loss, wq)
    return delta_loss_weight(w, wq.detach(), grad)
