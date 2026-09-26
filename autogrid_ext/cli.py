"""CLI extensions for the sky-is-green AUTOGRID fork.

``autogrid.main()`` adds these arguments and delegates here only when a fork
feature is actually requested; every upstream invocation keeps its original
code path and output.  ``autogrid-ext`` is a convenience entry point that runs
the same parser.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import plan as _plan
from . import scan as _scan

KNOWN_CONTAINERS = ("ternary", "q1_0_g128")


def add_arguments(ap) -> None:
    g = ap.add_argument_group("fork extensions (sky-is-green/autogrid)")
    g.add_argument("--plan-out", metavar="FILE",
                   help="write a versioned correction plan for the scan")
    g.add_argument("--containers", metavar="LIST", default="ternary",
                   help="container sims to evaluate: comma list of "
                        "ternary,q1_0_g128 (default: ternary)")
    g.add_argument("--banks", action="store_true",
                   help="print the MoE bank report after scanning")


def _parse_containers(a) -> tuple[str, ...]:
    names = tuple(n.strip() for n in (getattr(a, "containers", "ternary") or
                                      "ternary").split(",") if n.strip())
    for n in names:
        if n not in KNOWN_CONTAINERS:
            raise SystemExit(
                f"unknown container {n!r} (known: {', '.join(KNOWN_CONTAINERS)})")
    return names or ("ternary",)


def wants_extended(a) -> bool:
    if getattr(a, "scan", None) and (
            a.plan_out or a.banks or _parse_containers(a) != ("ternary",)):
        return True
    if getattr(a, "convert", None) and getattr(a, "plan", None) and \
            _plan.is_extended_plan(a.plan):
        return True
    return False


def _print_banks(banks: list[dict], summary: dict) -> None:
    if not banks:
        print("no MoE tensors detected")
        return
    print(f"\nMoE banks: {summary.get('expert_banks', 0)} expert banks "
          f"({summary.get('expert_tensors', 0)} tensors, "
          f"{summary.get('expert_params', 0):,} params) · "
          f"{summary.get('routers', 0)} routers")
    for b in banks:
        corr = b.get("correction") or {}
        extra = (f"  -> {corr.get('placement')} rank ~{corr.get('suggested_rank')}"
                 if b.get("action") == "correct" else "")
        print(f"  layer {str(b.get('layer')):<3} {b['role']:<11} "
              f"{b['n_tensors']:>4} tensors {b['params']:>13,} params  "
              f"{b['verdict']:<8} {b['action']}{extra}")
    cl = summary.get("correction_layers") or []
    if cl:
        from .moe import layer_spec
        print(f"  correction layers: {layer_spec(cl)}")


def main(a, *, scan_model, convert_model, revert_model) -> int:
    # A versioned plan -> convert through the upstream converter unchanged.
    if getattr(a, "convert", None):
        ops = _plan.load_plan(a.plan)
        bk = a.backup or a.convert.rstrip("/") + ".autogrid-backup.safetensors"
        done, worst, skipped = convert_model(a.convert, ops, bk)
        print(f"converted {done}, worst {worst:.3e}, backup {bk}, "
              f"skipped {skipped}")
        return 0

    if not getattr(a, "scan", None):
        raise SystemExit("nothing to do on the extension path")

    names = _parse_containers(a)

    def prog(i: int, n: int, name: str) -> None:
        if i % 25 == 0 or i == n:
            print(f"  [{i}/{n}] {name[-60:]}", flush=True)

    rep = _scan.scan_extended(a.scan, a.safety, a.max_k, names, progress=prog)

    s = rep["summary"]
    tot = max(s["params_total"], 1)
    print(f"\n{s['tensors']} tensors, {tot:,} params")
    print(f"  already ternary : {100*s['already_ternary_params']/tot:6.2f}%"
          f"  ({s['scale_convertible']} scale-convertible)")
    print(f"  FREE convert    : {100*s['free_convert_params']/tot:6.2f}%"
          f"  ({s['free_tensors']} tensors)")
    print(f"  needs STEERING  : {100*s['steer_params']/tot:6.2f}%"
          f"  ({s['steer_tensors']} tensors)")
    for r in rep["tensors"][:12]:
        k = r.get("rec_k")
        e = f"{r['errs'][k]:.2e}" if k and r.get("errs") else "  —  "
        print(f"  {r['cls']:<8} k={k or '-':<2} err {e}  {r['tensor'][-58:]}")

    for c in names:
        if c == "ternary":
            continue
        metrics = [m for r in rep["tensors"]
                   if (m := ((r.get("container_errs") or {}).get(c) or {}))]
        vals = [m for m in metrics if "rel_err" in m]
        if not vals:
            continue
        bpw = next((m.get("bpw") for m in vals if m.get("bpw")), None)
        med = statistics.median(m["rel_err"] for m in vals)
        absmean = [m["rel_err_absmean"] for m in vals
                   if m.get("rel_err_absmean") is not None]
        med_abs = statistics.median(absmean) if absmean else None
        free = sum(1 for m in vals if m.get("free"))
        line = (f"  container {c}: bpw {bpw:.3f} · rel err median {med:.3f}"
                + (f" · absmean {med_abs:.3f}" if med_abs is not None else "")
                + f" · {free}/{len(vals)} ≤ floor")
        skipped = len(metrics) - len(vals)
        if skipped:
            line += f" ({skipped} skipped)"
        print(line)

    if a.banks:
        _print_banks(rep.get("banks", []), rep.get("moe_summary") or {})

    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
        print(f"report -> {a.json}")

    if a.plan_out:
        plan = _plan.build_plan(rep, names)
        _plan.save_plan(plan, a.plan_out)
        print(f"plan -> {a.plan_out}")
        print(_plan.plan_summary(plan))
    return 0


def main_entry() -> int:
    import autogrid
    autogrid.main()
    return 0
