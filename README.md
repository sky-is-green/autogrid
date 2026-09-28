# AUTOGRID

**Turn any model into a full-size *integer* model — for free, where the math says it's free.**

> **This is a fork.** [`sky-is-green/autogrid`](https://github.com/sky-is-green/autogrid)
> is forked from [AUTOGRID](https://github.com/CodeMasterCody3D/autogrid) by
> Cody Dixon (MIT). Fork point: upstream commit `0058823`. Changes are tracked
> in [CHANGELOG.md](CHANGELOG.md); attribution is in [NOTICE](NOTICE).

AUTOGRID is a desktop app + CLI that scans every tensor of a safetensors
checkpoint, measures what it would cost to put that tensor on a
**balanced-ternary integer grid** (`w ≈ s · Σ dᵢ·3⁻ⁱ`, digits `dᵢ ∈ {−1,0,+1}`),
and converts the ones where the conversion is **below the noise the model was
already saved with.** No retraining. No calibration data. Fully reversible.

This is **not quantization** — it doesn't make the model smaller. It's a
**numeral-system change**: the same values, re-expressed as stacks of ternary
digits so the model's parameters become *integers* instead of floats.

---

## The idea: the conversion is quieter than the container

A float format has its own rounding noise just by existing. **bf16** keeps 8
significant bits, so it rounds every value it stores by up to **~0.39%**.

A **k-digit** balanced-ternary stack has a relative error of exactly **3⁻ᵏ**:

| digits `k` | error `3⁻ᵏ` | bits/value (group 128) |
|---|---|---|
| 6 | 0.14% | ~12.1 |
| 8 | 0.015% | ~16.1 |

At **k=8**, the ternary-integer error (0.015%) is **~25× smaller** than the bf16
container's own rounding (0.39%). So converting a bf16 tensor to a k8 digit stack
**discards less information than the checkpoint's storage format already did** —
the model cannot notice a perturbation smaller than the noise it was saved with.

**Measured proof:** on a 27-billion-parameter model, converting all 353
normalization / state-space tensors to k8 integer stacks changed wikitext
perplexity by **0.009%** over 274 chunks — one hundredth of the measurement's
error bar. Field convention says "never touch the norms"; the noise-floor math
says you can, and the measurement agrees.

---

## What it does

For each tensor, AUTOGRID measures the per-digit-budget error and classifies it:

- 🟢 **FREE @ k** — a float tensor whose measured k-digit error is **at or below
  its container's noise floor**. Converting it to a ternary-integer stack is free.
  AUTOGRID recommends the smallest sufficient `k`.
- 🔵 **TERNARY** — a tensor whose *values* are already on the `{−1,0,+1}×scale`
  grid (e.g. a placed/quantized weight). AUTOGRID analyzes its **group scales**
  for digit-stack conversion instead.
- 🔴 **STEER** — error is still above the floor even at k=8. This tensor needs a
  real correction pipeline (rotation / Hessian-aware placement / reconstruction),
  **not** a numeral-system change. **AUTOGRID refuses to touch it.**

Then **CONVERT** applies the recommended `k` per tensor, **in place**, with the
originals backed up to one file first — every patch is byte-length-guarded and
**REVERT**-ible byte-for-byte.

### The `max k` dial = the two regimes

- **k = 8 … 3** — the *free zone*. Convert to integer with no measurable quality
  cost. You get integer arithmetic and bit-exact reproducibility; the model is
  the **same size** (this is conversion, not compression).
- **k = 2 … 1** — the *compression regime* (2–4 bits/value). Here naive
  conversion **collapses** — this is where quantization lives, and where a
  steered pipeline (not AUTOGRID) is required. Set `max k` to 2 and watch the
  weight matrices flip to 🔴 STEER: that's the app telling you the truth about
  where "free" ends and "earned" begins.

---

## Usage

**Requirements:** Python 3.9+, `torch`, `safetensors` (`pip install torch safetensors`).

```bash
# Desktop UI (tkinter, no browser, no server):
python autogrid.py

# Headless scan (prints the class of every tensor + a recommended k):
python autogrid.py --scan /path/to/model            # HF dir or a .safetensors file
python autogrid.py --scan /path/to/model --max-k 2  # see what survives the compression grid
python autogrid.py --scan /path/to/model --json report.json

# Convert in place (backs up originals first), then revert if you want:
python autogrid.py --convert /path/to/model --plan plan.json
python autogrid.py --revert  /path/to/model --backup model.autogrid-backup.safetensors
```

`--scan` accepts a HuggingFace-layout directory (with
`model.safetensors.index.json` or a single `model.safetensors`) or a
`.safetensors` file directly.

---

## Why "integer, not smaller" is worth wanting

- **Integer arithmetic** end-to-end where it matters — no float in the values.
- **Bit-exact reproducibility** — integer values don't drift across machines.
- **A principled starting point for compression.** Once a model is on the integer
  ladder, dropping digit planes (k8 → … → k1) is a clean, exactly-known descent,
  and the dropped planes are themselves ternary corrections. AUTOGRID is the
  *front door*; the compression regime below k3 is where a steered pipeline
  takes over.

---

## Fork extensions (sky-is-green)

This fork keeps the upstream engine and CLI untouched and adds an extension
package (`autogrid_ext/`) for MoE work:

- **MoE-aware bank reports** (`--banks`) — expert tensors are parsed into
  roles and aggregated into banks with `FREE@k` / `STEER` verdicts,
  per-projection breakdowns, and correction policy hints from the measured
  placement rules (residual-stream `moe_out` branches, rank ~512; rank-64
  router deltas).
- **Correction plans** (`--plan-out plan.json`) — a versioned plan the
  upstream converter consumes directly: safe convert ops plus the correction
  layers, targets and routers for the training pipeline.
- **Deployed-container simulation** (`--containers ternary,q1_0_g128`) —
  measures each tensor inside the TAARDIS `Q1_0_g128` container (2-bit codes
  + fp16 Lloyd-Max group scale = 2.125 bpw), with the absmean rule kept for
  comparison.
- **Python API + tests** — `import autogrid` for the engine,
  `import autogrid_ext` for banks/plans/containers; pytest suite and CI in
  this fork.
- **DeltaLoss steering ranker** (`autogrid_ext.steer`, SignRoundV2) — ranks
  STEER tensors by gradient × quantization perturbation on calibration
  inputs, for deciding what the steered pipeline touches first.

```bash
python autogrid.py --scan /path/to/model --banks --plan-out plan.json
python autogrid.py --scan /path/to/model --containers ternary,q1_0_g128
python autogrid.py --convert /path/to/model --plan plan.json
```

Fork changes are recorded in [CHANGELOG.md](CHANGELOG.md); attribution is in
[NOTICE](NOTICE).

## Part of TAARDIS

AUTOGRID is the noise-floor auto-detector from **TAARDIS** (*Ternary Adaptive
Alignment & Rotation for Dense Integer Stacking*) — a pipeline for full-ternary,
fully-integer large language models. The digit-ladder math and the noise-floor
principle are validated there at 27B scale.

MIT licensed. By Cody Dixon, 2026.
