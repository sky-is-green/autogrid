# Changelog

All notable changes to this fork are recorded here. Upstream is
[`CodeMasterCody3D/autogrid`](https://github.com/CodeMasterCody3D/autogrid).

## Unreleased

- Forked from upstream commit `0058823` (2026-09-03).
- Added fork attribution scaffolding: `NOTICE`, this changelog, and a fork
  banner in `README.md`.
- `autogrid_ext`: MoE-aware tensor parsing and expert-bank reporting with
  correction policy hints (`moe_out`, rank ~512, rank-64 router deltas).
- `--plan-out`: versioned correction-plan export, consumed by the existing
  `--convert --plan` path (legacy bare-list plans still load).
- `--containers`: deployed-container simulation; `q1_0_g128` (2-bit codes +
  fp16 Lloyd-Max group scale, 2.125 bpw) with absmean comparison.
- Python API (`import autogrid`, `import autogrid_ext`), pytest suite, CI.
- `autogrid.py` keeps upstream behaviour; a small hook delegates to the
  extension only when fork flags are used.
- Fixed the in-place converter's false "reload identical" warning:
  `safe_open` tensors are memory-mapped views, so the in-memory backup now
  gets cloned before patching and the reload check verifies a real write.
