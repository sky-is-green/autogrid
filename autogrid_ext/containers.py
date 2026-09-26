"""Container simulations for the sky-is-green AUTOGRID fork.

The upstream tool measures a tensor on the balanced-ternary digit ladder.  This
module lets a scan additionally ask "how would this tensor fare *inside a
deployed container*?" — currently the TAARDIS ``Q1_0_g128`` MoE format.

All quantizers here are independent torch reimplementations of the rules used
by the TAARDIS llama.cpp fork (MIT, (c) CodeMasterCody3D) — see NOTICE.  The
``ternary_lloyd`` rule is the fixed point the Q1_0_g128 quantizer uses (Lloyd-
Max refinement, fp16 scale round-trip reproduced), kept alongside the plain
absmean rule for comparison.  They are not code copies.
"""
from __future__ import annotations

import torch

GROUP = 128
SAMPLE = 2_000_000

#: bits/value for a k-digit balanced-ternary stack: 2-bit codes + fp16 scale
#: per 128 weights.
def container_bpw(name: str, k: int | None = None) -> float:
    if name == "ternary":
        if not k:
            raise ValueError("ternary bpw needs k")
        return (k * 2 * GROUP + 16) / GROUP
    if name == "q1_0_g128":
        return (2 * GROUP + 16) / GROUP
    raise KeyError(name)


# ---------------------------------------------------------------- quantizers
def _absmean_scale(g: torch.Tensor) -> torch.Tensor:
    return g.abs().mean(dim=-1).clamp_min(1e-8)


def _lloyd_scale(g: torch.Tensor, mean: torch.Tensor,
                 iters: int = 8, inits=(0.5, 0.7, 0.9, 1.1)) -> torch.Tensor:
    """Lloyd-Max / TWN fixed-point group scale (the Q1_0_g128 rule).

    Iterates ``a <- mean(|w| : |w| > a/2)`` from four starts and keeps the one
    with the best residual reduction.  Matches the TAARDIS quantizer's fixed
    point (independent reimplementation; see NOTICE).
    """
    best_a = mean.clone()
    best_obj = torch.full_like(mean, -1.0)
    for init in inits:
        a = init * mean
        s1 = torch.zeros_like(mean)
        sw = torch.zeros_like(mean)
        for _ in range(iters):
            mask = g.abs() > 0.5 * a.unsqueeze(-1)
            s1 = (g.abs() * mask).sum(-1)
            sw = mask.sum(-1).float()
            a = torch.where(sw > 0, s1 / sw, torch.zeros_like(a))
        obj = torch.where(sw > 0, s1 * s1 / sw.clamp_min(1e-9),
                          torch.zeros_like(s1))
        take = obj > best_obj
        best_obj = torch.where(take, obj, best_obj)
        best_a = torch.where(take, a, best_a)
    return torch.where(best_a > 0, best_a, mean)


def _recon_groups(g: torch.Tensor, rule: str) -> torch.Tensor:
    """Reconstruct [..., group] float groups under ``absmean`` or ``lloyd``."""
    if rule == "absmean":
        a = _absmean_scale(g).unsqueeze(-1)
        return torch.clamp(torch.round(g / a), -1, 1) * a
    if rule != "lloyd":
        raise KeyError(rule)
    mean = g.abs().mean(-1)
    a = _lloyd_scale(g, mean).half().float()
    q = torch.clamp(torch.round(g / a.unsqueeze(-1).clamp_min(1e-12)), -1, 1)
    return q * a.unsqueeze(-1)


def ternary_absmean(w: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Per-group absmean ternary reconstruction (Bonsai/RTN baseline)."""
    if group <= 0 or w.shape[-1] % group:
        group = w.shape[-1]
    g = w.reshape(-1, group)
    return _recon_groups(g, "absmean").reshape(w.shape)


def ternary_lloyd(w: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Per-group Lloyd-refined ternary reconstruction (Q1_0_g128 rule)."""
    if group <= 0 or w.shape[-1] % group:
        group = w.shape[-1]
    g = w.float().reshape(-1, group)
    return _recon_groups(g, "lloyd").reshape(w.shape).to(w.dtype)


# ---------------------------------------------------------------- evaluation
def _sample_flat(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    flat = t.reshape(1, -1).float()
    if flat.numel() > SAMPLE:
        flat = flat[:, :SAMPLE]
        return flat, True
    return flat, False


def evaluate(name: str, t: torch.Tensor, floor: float | None,
             safety: float = 1.0) -> dict:
    """Metrics for one container on one tensor.

    Returns a dict with the container's rules applied to a bounded sample of
    the tensor.  ``free`` is the same verdict the upstream scan uses: the
    container's relative error at or below the tensor's storage noise floor.
    """
    if name != "q1_0_g128":
        return {"error": f"unknown container {name!r}"}
    if not t.is_floating_point() or t.numel() == 0:
        return {"skipped": "non-float or empty tensor"}

    group = GROUP
    if t.shape[-1] % group:
        group = t.shape[-1] or 1          # same fallback as the pipeline rule

    flat, sampled = _sample_flat(t)
    if flat.shape[-1] % group:
        flat = flat[:, :flat.shape[-1] - (flat.shape[-1] % group)]
    g = flat.reshape(-1, group)

    orig = g.clone()
    absmean = _recon_groups(g, "absmean")
    lloyd = _recon_groups(g, "lloyd")

    def rel(recon: torch.Tensor) -> float:
        return float((recon.double() - orig.double()).norm() /
                     orig.double().norm().clamp_min(1e-30))

    err = rel(lloyd)
    out = {
        "rule": "lloyd",
        "rel_err": err,
        "rel_err_absmean": rel(absmean),
        "bpw": container_bpw(name),
        "group": group,
        "sampled": sampled,
        "floor": floor,
    }
    if floor is not None:
        out["free"] = bool(err <= floor * safety)
    return out
