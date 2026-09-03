#!/usr/bin/env python3
"""TAARDIS AUTOGRID — the noise-floor auto-detector & integer converter.

Cody, 2026-09-02: "that exact method you're using to find the best. i need you
to make an auto detector thing with a ui, make it a desktop app where we can
run a full model and test every part of it ... and it will convert it for you."

THE METHOD (validated on the 27B, paper §1.2). A float container has its own
rounding noise: bf16 rounds every value by up to 2^-8 ~ 0.39% just by existing.
A k-digit balanced-ternary stack (w ~ s * sum d_i 3^-i, d in {-1,0,+1}) has
error 3^-k. Whenever the MEASURED conversion error of a tensor sits at or
below its container's noise floor, converting it to ternary-integer digit
stacks is FREE — the model cannot notice a perturbation smaller than the noise
it was saved with. Measured proof: k8 norms on the 27B = 0.009% PPL over 274
chunks, one hundredth of the error bar.

WHAT THIS APP DOES to any HF-layout checkpoint (safetensors):
  SCAN     walk every tensor and classify it:
             TERNARY  values already on the {-1,0,+1}xscale grid (placed
                      weights) -> analyze its GROUP SCALES for digit-stack
                      conversion instead (the scale surgery)
             FREE@k   float tensor whose measured k-digit error <= its floor
                      -> convertible for free at the recommended k
             STEER    error above the floor even at k=8 -> this tensor needs
                      the full correction pipeline (rotation/Hessian/recon),
                      not a numeral-system change. The app refuses to touch it.
  CONVERT  apply the recommended (or overridden) k per tensor, IN PLACE:
           byte-length-guarded patch, originals backed up first to one
           safetensors file. Codes/shapes/dtypes never change, so every
           patch is reversible and the file never grows.
  REVERT   restore every converted tensor byte-exactly from the backup.

Same one-law as the pipeline: nothing leaves this tool off-grid, and nothing
is written without a backup and a reload check.

    ./autogrid.py                      # the desktop UI
    ./autogrid.py --scan PATH [--json out.json]      # headless scan
    ./autogrid.py --convert PATH --plan plan.json    # headless convert
    ./autogrid.py --revert PATH --backup FILE        # headless revert
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file


# ---- inlined from the TAARDIS pipeline (no external deps) ------------------
def resolve_group(width: int, group: int) -> int:
    """Largest usable group <= `group` that DIVIDES `width`, by halving."""
    if group <= 0:
        group = width
    g = int(group)
    while g > 1 and width % g:
        g //= 2
    return max(g, 1)


def encode(w: torch.Tensor, g: int, k: int) -> torch.Tensor:
    """Balanced-ternary signed-digit expansion: w ~ s * sum_i d_i 3^-i,
    d_i in {-1,0,+1}, per group of `g`. Relative error is exactly 3^-k."""
    shp = w.shape
    grp = w.reshape(-1, g)
    s = grp.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) * 2.0
    r = grp / s
    acc = torch.zeros_like(r)
    p = 1.0
    for _ in range(k):
        p /= 3.0
        d = torch.round(r * 3.0).clamp_(-1, 1)
        acc += d * p
        r = r * 3.0 - d
    return (acc * s).reshape(shp)
# ---------------------------------------------------------------------------


GROUP = 128                       # the ternary weight group (matches the pipeline)
SCALE_G = 8                       # scales share an exponent per group of 8
KS = (2, 3, 4, 5, 6, 8)           # the digit ladder rungs we sweep
SAMPLE = 2_000_000                # elements sampled per tensor for the sweep
CHUNK_ROWS = 2048                 # row-block size for big-tensor passes

# max relative ROUNDING error of the container the tensor is stored in:
# p significant bits (incl. implicit) -> half-ulp = 2^-p
FLOORS = {
    "BF16": 2.0 ** -8,            # 8 sig bits  -> 0.39%
    "F16":  2.0 ** -11,           # 11 sig bits -> 0.049%
    "F32":  2.0 ** -24,
    "F64":  2.0 ** -53,
}
TORCH_FLOOR = {torch.bfloat16: FLOORS["BF16"], torch.float16: FLOORS["F16"],
               torch.float32: FLOORS["F32"], torch.float64: FLOORS["F64"]}


# ----------------------------------------------------------------- engine ----
def _shards(model_path: Path):
    """-> {tensor_name: shard_path}. Accepts a dir (HF index / single file
    inside) or a .safetensors file directly."""
    p = Path(model_path)
    if p.is_file():
        with safe_open(str(p), framework="pt") as h:
            return {k: p for k in h.keys()}
    idx = p / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.load(open(idx))["weight_map"]
        return {n: p / sh for n, sh in wm.items()}
    one = p / "model.safetensors"
    if one.is_file():
        with safe_open(str(one), framework="pt") as h:
            return {k: one for k in h.keys()}
    raise SystemExit(f"no safetensors found under {p}")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() /
                 b.double().norm().clamp_min(1e-30))


def _encode_rel(flat: torch.Tensor, g: int, k: int) -> float:
    """relative error of the k-digit stack on flat [1,N] (already float)."""
    return _rel(encode(flat, g, k), flat)


def _ternary_probe(t: torch.Tensor):
    """Is this 2D tensor already {-s,0,+s} per g128 group? -> (bool, scales)"""
    if t.ndim != 2 or t.shape[-1] % GROUP:
        return False, None
    O, I = t.shape
    worst, scales = 0.0, []
    for r0 in range(0, O, CHUNK_ROWS):
        v = t[r0:r0 + CHUNK_ROWS].float().reshape(-1, I // GROUP, GROUP)
        s = v.abs().amax(-1)
        codes = torch.where(s.unsqueeze(-1) > 0,
                            v / s.unsqueeze(-1).clamp_min(1e-30), v)
        worst = max(worst, float((codes - codes.round().clamp(-1, 1)).abs().max()))
        if worst > 1e-6:
            return False, None
        scales.append(s.reshape(-1))
    return True, torch.cat(scales)


def scan_tensor(name: str, t: torch.Tensor, safety: float = 1.0, ks=KS):
    """-> dict row for one tensor: class, floor, per-k errors, recommendation."""
    floor = TORCH_FLOOR.get(t.dtype)
    row = {"tensor": name, "shape": list(t.shape), "params": t.numel(),
           "dtype": str(t.dtype).replace("torch.", ""), "floor": floor}
    if floor is None:                                # int tensors etc.
        row.update({"cls": "SKIP", "why": "non-float dtype"})
        return row

    on_grid, scales = _ternary_probe(t)
    if on_grid:
        # already ternary -> the SCALES are the remaining floats
        flat = scales.reshape(1, -1)
        g = resolve_group(flat.shape[1], SCALE_G)
        errs = {k: _encode_rel(flat, g, k) for k in ks}
        rec = next((k for k in ks if errs[k] <= floor * safety), None)
        row.update({"cls": "TERNARY", "mode": "scales", "n_scales": scales.numel(),
                    "errs": errs, "rec_k": rec})
        return row

    flat = t.flatten()
    if flat.numel() > SAMPLE:                        # deterministic sample
        flat = flat[:SAMPLE - (SAMPLE % GROUP) or SAMPLE]
    flat = flat.reshape(1, -1).float()
    g = resolve_group(flat.shape[1], GROUP)
    errs = {k: _encode_rel(flat, g, k) for k in ks}
    rec = next((k for k in ks if errs[k] <= floor * safety), None)
    row.update({"cls": "FREE" if rec else "STEER", "mode": "values",
                "errs": errs, "rec_k": rec})
    return row


def scan_model(model_path, safety=1.0, max_k=8, progress=lambda *a: None):
    ks = tuple(k for k in KS if k <= max_k) or (max_k,)
    wm = _shards(model_path)
    rows, t0 = [], time.time()
    names = sorted(wm)
    for i, n in enumerate(names):
        with safe_open(str(wm[n]), framework="pt") as h:
            t = h.get_tensor(n)
        rows.append(scan_tensor(n, t, safety, ks))
        progress(i + 1, len(names), n)
    report = {
        "model": str(model_path), "safety": safety,
        "elapsed_s": round(time.time() - t0, 1), "tensors": rows,
        "summary": _summarize(rows),
    }
    return report


def _summarize(rows):
    tot = sum(r["params"] for r in rows)
    by = lambda c: [r for r in rows if r.get("cls") == c]
    tern, free, steer = by("TERNARY"), by("FREE"), by("STEER")
    return {
        "params_total": tot,
        "tensors": len(rows),
        "already_ternary_params": sum(r["params"] for r in tern),
        "free_convert_params": sum(r["params"] for r in free),
        "steer_params": sum(r["params"] for r in steer),
        "scale_convertible": sum(1 for r in tern if r.get("rec_k")),
        "free_tensors": len(free), "steer_tensors": len(steer),
    }


def _chunked_encode_values(t, k):
    """full-tensor k-digit encode, row-chunked so memory stays flat."""
    flat = t.reshape(1, -1) if t.ndim == 1 else t.reshape(t.shape[0], -1)
    out = torch.empty_like(t, dtype=torch.float32).reshape(flat.shape)
    step = max(1, CHUNK_ROWS)
    for r0 in range(0, flat.shape[0], step):
        blk = flat[r0:r0 + step].float()
        g = resolve_group(blk.shape[-1], GROUP)
        out[r0:r0 + step] = encode(blk.reshape(1, -1), g, k).reshape(blk.shape)
    return out.reshape(t.shape).to(t.dtype).contiguous()


def _snap_scales_tensor(t, k):
    """ternary tensor -> same codes x k-digit-snapped scales."""
    O, I = t.shape
    outs = []
    for r0 in range(0, O, CHUNK_ROWS):
        v = t[r0:r0 + CHUNK_ROWS].float().reshape(-1, I // GROUP, GROUP)
        s = v.abs().amax(-1)
        codes = torch.where(s.unsqueeze(-1) > 0,
                            v / s.unsqueeze(-1).clamp_min(1e-30), v).round().clamp(-1, 1)
        flat = s.reshape(1, -1)
        g = resolve_group(flat.shape[1], SCALE_G)
        sp = encode(flat, g, k).reshape(s.shape)
        outs.append((codes * sp.unsqueeze(-1)).reshape(-1, I))
    return torch.cat(outs).to(t.dtype).contiguous()


def convert_model(model_path, plan, backup_path, progress=lambda *a: None):
    """plan: [{tensor, mode: values|scales, k}] — in-place, backed up, verified.
    Returns (n_done, worst_rel, skipped)."""
    wm = _shards(model_path)
    plan = [p for p in plan if p.get("k")]
    # 1) backup originals (one file; refuse to write without it)
    backup = {}
    for p in plan:
        with safe_open(str(wm[p["tensor"]]), framework="pt") as h:
            backup[p["tensor"]] = h.get_tensor(p["tensor"]).contiguous()
    save_file(backup, str(backup_path))
    # 2) patch shard by shard
    byshard = {}
    for p in plan:
        byshard.setdefault(wm[p["tensor"]], []).append(p)
    done, worst, skipped = 0, 0.0, []
    for sh, items in byshard.items():
        with open(sh, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        with open(sh, "r+b") as f:
            for p in items:
                t = backup[p["tensor"]]
                new = (_snap_scales_tensor(t, p["k"]) if p["mode"] == "scales"
                       else _chunked_encode_values(t, p["k"]))
                worst = max(worst, _rel(new, t))
                raw = new.flatten().view(torch.uint8).numpy().tobytes()
                s0, s1 = hdr[p["tensor"]]["data_offsets"]
                if len(raw) != s1 - s0:                       # THE GUARD
                    skipped.append(p["tensor"]); continue
                f.seek(base + s0); f.write(raw)
                done += 1
                progress(done, len(plan), p["tensor"])
    # 3) reload check on the first patched tensor
    if plan and not skipped:
        n0 = plan[0]["tensor"]
        with safe_open(str(wm[n0]), framework="pt") as h:
            delta = float((h.get_tensor(n0).double() -
                           backup[n0].double()).abs().max())
        if delta == 0:
            skipped.append(f"{n0}: reload identical — nothing was written?")
    return done, worst, skipped


def revert_model(model_path, backup_path):
    wm = _shards(model_path)
    orig = load_file(str(backup_path))
    byshard = {}
    for n in orig:
        byshard.setdefault(wm[n], []).append(n)
    done = 0
    for sh, names in byshard.items():
        with open(sh, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        with open(sh, "r+b") as f:
            for n in names:
                raw = orig[n].flatten().view(torch.uint8).numpy().tobytes()
                s0, s1 = hdr[n]["data_offsets"]
                assert len(raw) == s1 - s0, n
                f.seek(base + s0); f.write(raw)
                done += 1
    return done


# --------------------------------------------------------------------- UI ----
def launch_ui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("TAARDIS AUTOGRID — noise-floor detector & integer converter")
    root.geometry("1180x720")
    q: "queue.Queue" = queue.Queue()
    state = {"report": None, "path": None}

    top = ttk.Frame(root, padding=8); top.pack(fill="x")
    path_var = tk.StringVar()
    ttk.Label(top, text="Model").pack(side="left")
    ent = ttk.Entry(top, textvariable=path_var, width=64); ent.pack(side="left", padx=6)
    def browse():
        p = filedialog.askdirectory() or filedialog.askopenfilename(
            filetypes=[("safetensors", "*.safetensors")])
        if p: path_var.set(p)
    ttk.Button(top, text="Browse", command=browse).pack(side="left")
    safety_var = tk.DoubleVar(value=1.0)
    ttk.Label(top, text="  safety×floor").pack(side="left")
    ttk.Spinbox(top, from_=0.25, to=4.0, increment=0.25, width=5,
                textvariable=safety_var).pack(side="left")
    maxk_var = tk.IntVar(value=8)
    ttk.Label(top, text="  max k (compression dial)").pack(side="left")
    ttk.Spinbox(top, from_=1, to=8, width=4, textvariable=maxk_var).pack(side="left")

    summ = tk.StringVar(value="scan a model to begin — the method: convert "
                              "whatever measures below its container's noise floor")
    ttk.Label(root, textvariable=summ, padding=(10, 4)).pack(fill="x")

    cols = ("cls", "dtype", "params", "floor", "best", "reck")
    treefr = ttk.Frame(root); treefr.pack(fill="both", expand=True, padx=8)
    treefr.rowconfigure(0, weight=1); treefr.columnconfigure(0, weight=1)
    tree = ttk.Treeview(treefr, columns=cols, show="tree headings", height=22)
    tree.heading("#0", text="tensor")
    # stretch=False keeps the data columns COMPACT; only the tensor name grows
    for c, w, txt in (("cls", 92, "class"), ("dtype", 66, "dtype"),
                      ("params", 92, "params"), ("floor", 64, "floor"),
                      ("best", 78, "err@k"), ("reck", 44, "k")):
        tree.heading(c, text=txt)
        tree.column(c, width=w, minwidth=w, anchor="e", stretch=False)
    tree.column("#0", width=420, minwidth=240, stretch=True)
    tree.tag_configure("TERNARY", foreground="#155fb8")
    tree.tag_configure("FREE", foreground="#1d7a45")
    tree.tag_configure("STEER", foreground="#b3402f")
    tree.tag_configure("SKIP", foreground="#888888")
    vs = ttk.Scrollbar(treefr, orient="vertical", command=tree.yview)
    hs = ttk.Scrollbar(treefr, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vs.grid(row=0, column=1, sticky="ns")
    hs.grid(row=1, column=0, sticky="ew")

    prog = ttk.Progressbar(root, mode="determinate"); prog.pack(fill="x", padx=8)
    log = tk.Text(root, height=6, bg="#101418", fg="#d7dbe0")
    log.pack(fill="x", padx=8, pady=(2, 4))
    def say(msg):
        log.insert("end", msg + "\n"); log.see("end")

    def worker_scan():
        try:
            rep = scan_model(state["path"], safety_var.get(), maxk_var.get(),
                             progress=lambda i, n, name: q.put(("p", i, n, name)))
            q.put(("done", rep))
        except Exception as e:
            q.put(("err", str(e)))

    def do_scan():
        p = path_var.get().strip()
        if not p:
            messagebox.showerror("AUTOGRID", "pick a model first"); return
        state["path"] = p
        tree.delete(*tree.get_children())
        say(f"scanning {p} ...")
        threading.Thread(target=worker_scan, daemon=True).start()

    def fill(rep):
        state["report"] = rep
        s = rep["summary"]; tot = max(s["params_total"], 1)
        summ.set(f"{s['tensors']} tensors · {tot:,} params ·  "
                 f"already-ternary {100*s['already_ternary_params']/tot:.1f}%  ·  "
                 f"free-convertible {100*s['free_convert_params']/tot:.1f}%  ·  "
                 f"needs-steering {100*s['steer_params']/tot:.1f}%  ·  "
                 f"{s['scale_convertible']} ternary tensors scale-convertible")
        for r in rep["tensors"]:
            k = r.get("rec_k")
            best = f"{r['errs'][k]:.2e}" if k and r.get("errs") else "—"
            tree.insert("", "end", text=r["tensor"], tags=(r["cls"],),
                        values=(r["cls"] + ("/scales" if r.get("mode") == "scales" else ""),
                                r["dtype"], f"{r['params']:,}",
                                f"{r['floor']:.1e}" if r.get("floor") else "—",
                                best, k or "—"))
        say(f"scan done in {rep['elapsed_s']}s — green FREE rows and blue "
            f"TERNARY(scales) rows convert for free; red STEER rows need the "
            f"full pipeline and are refused here")

    def plan_from(rows):
        return [{"tensor": r["tensor"], "mode": r.get("mode", "values"),
                 "k": r.get("rec_k")} for r in rows if r.get("rec_k")]

    def do_convert(selected_only=False):
        rep = state.get("report")
        if not rep:
            messagebox.showerror("AUTOGRID", "scan first"); return
        rows = rep["tensors"]
        if selected_only:
            names = {tree.item(i, "text") for i in tree.selection()}
            rows = [r for r in rows if r["tensor"] in names]
        plan = plan_from(rows)
        if not plan:
            messagebox.showinfo("AUTOGRID", "nothing convertible in selection"); return
        bk = Path(state["path"]).with_suffix("") if Path(state["path"]).is_file() \
            else Path(state["path"])
        bk = Path(str(bk) + ".autogrid-backup.safetensors")
        n_par = sum(r["params"] for r in rows if r.get("rec_k"))
        if not messagebox.askyesno(
                "AUTOGRID — convert",
                f"Convert {len(plan)} tensors ({n_par:,} params) IN PLACE?\n"
                f"Backup -> {bk.name}\nEvery patch is byte-length guarded and "
                f"reversible with Revert."):
            return
        say(f"converting {len(plan)} tensors ...")
        def w():
            try:
                done, worst, skipped = convert_model(
                    state["path"], plan, bk,
                    progress=lambda i, n, name: q.put(("p", i, n, name)))
                q.put(("conv", done, worst, skipped, str(bk)))
            except Exception as e:
                q.put(("err", str(e)))
        threading.Thread(target=w, daemon=True).start()

    def do_revert():
        p = filedialog.askopenfilename(
            filetypes=[("autogrid backup", "*.autogrid-backup.safetensors")])
        if not p: return
        n = revert_model(state["path"] or path_var.get(), p)
        say(f"reverted {n} tensors byte-exactly from {Path(p).name}")

    def do_export():
        rep = state.get("report")
        if not rep: return
        p = filedialog.asksaveasfilename(defaultextension=".json")
        if p:
            Path(p).write_text(json.dumps(rep, indent=1))
            say(f"report -> {p}")

    bar = ttk.Frame(root, padding=(8, 2)); bar.pack(fill="x")
    ttk.Button(bar, text="SCAN", command=do_scan).pack(side="left")
    ttk.Button(bar, text="Convert all safe",
               command=lambda: do_convert(False)).pack(side="left", padx=6)
    ttk.Button(bar, text="Convert selected",
               command=lambda: do_convert(True)).pack(side="left")
    ttk.Button(bar, text="Revert from backup", command=do_revert).pack(side="left", padx=6)
    ttk.Button(bar, text="Export report", command=do_export).pack(side="left")

    def pump():
        try:
            while True:
                m = q.get_nowait()
                if m[0] == "p":
                    _, i, n, name = m
                    prog["maximum"], prog["value"] = n, i
                    if i % 25 == 0 or i == n:
                        say(f"  [{i}/{n}] {name[-70:]}")
                elif m[0] == "done":
                    fill(m[1]); prog["value"] = 0
                elif m[0] == "conv":
                    _, done, worst, skipped, bk = m
                    say(f"CONVERTED {done} tensors, worst rel err {worst:.3e}; "
                        f"backup {bk}")
                    if skipped: say(f"SKIPPED (guard): {skipped}")
                    prog["value"] = 0
                elif m[0] == "err":
                    say("ERROR: " + m[1]); prog["value"] = 0
        except queue.Empty:
            pass
        root.after(120, pump)
    pump()
    root.mainloop()


# -------------------------------------------------------------------- cli ----
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", metavar="PATH")
    ap.add_argument("--convert", metavar="PATH")
    ap.add_argument("--revert", metavar="PATH")
    ap.add_argument("--plan", metavar="JSON")
    ap.add_argument("--backup", metavar="FILE")
    ap.add_argument("--json", metavar="OUT")
    ap.add_argument("--safety", type=float, default=1.0)
    ap.add_argument("--max-k", type=int, default=8)
    a = ap.parse_args()
    if a.scan:
        rep = scan_model(a.scan, a.safety, a.max_k,
                         progress=lambda i, n, name:
                         print(f"  [{i}/{n}] {name[-60:]}", flush=True)
                         if i % 25 == 0 or i == n else None)
        s = rep["summary"]; tot = max(s["params_total"], 1)
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
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=1))
            print(f"report -> {a.json}")
    elif a.convert:
        plan = json.loads(Path(a.plan).read_text())
        bk = a.backup or a.convert.rstrip("/") + ".autogrid-backup.safetensors"
        done, worst, skipped = convert_model(a.convert, plan, bk)
        print(f"converted {done}, worst {worst:.3e}, backup {bk}, skipped {skipped}")
    elif a.revert:
        print("reverted", revert_model(a.revert, a.backup), "tensors")
    else:
        launch_ui()


if __name__ == "__main__":
    main()
