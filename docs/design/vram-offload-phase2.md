# VRAM ladder phase 2: offload dense projection layers to the CPU at runtime

Status: design only (Nov 24, 2026). Builds on PR #31 (VRAM ladder rungs: expert cache to min, attention K/V to host,
degraded wait, stop). This adds a rung between "K/V to host" and "degraded wait": give back the dense projection weights
of some layers, so pyre can keep serving at the full context when another app takes several GB of VRAM.

## What is on the GPU today (IQ3_S prod, cube backend)

- `attach_gpu()` uploads, per layer, a `CLayer` into `Model.cl: Vec<Option<CLayer>>`: mixer in/out projections
  (`min`, `mout`), the two hyper-connection blocks (`hca`, `hcf`), router + shared expert (`fin`, `sdown`), DeltaNet
  weights (`gdn`) and attention norms (`an`). All 48 layers come to 5.8 GiB of Q8, about 124 MiB per layer.
- The host copies stay available (the GGUF mmap and `Model.layers`), so a layer can be uploaded again later without
  touching the disk. Upload speed at load is ~6 s for 48 layers, ~125 ms per layer.
- `gl` (burn `GLayer`) is the older non-cube path and stays empty when the cube is on.

## Why it is not a small change

1. **Immutability.** The engine owns an `Arc<Model>`, and `cl` is a plain `Vec` filled once in `attach_gpu(&mut self)`.
   Dropping or adding a layer at runtime needs interior mutability.
2. **All-or-nothing fast paths.** Several paths assume *every* layer has a `CLayer`:
   - `forward()`: the GPU-resident DeltaNet state handoff (`xexact() && !st.gdn_gpu && cl.len() == layers.len()`).
   - `forward_inner()`: the graph/fast decode gate (`cl.iter().zip(layers).all(... gdn.is_some() ...)`).
   - `multi_ok()`: the same all-layers test; it gates `forward_multi_x()` (MTP verify + multi-token prefill), which
     then calls `self.cl[nl].as_ref().unwrap()` for every layer.
   - `layer_cube_multi_p()`, `dense_gpu_only()` (unwrap), `xgr_launch()` (`?`, safe).
3. **Captured graphs.** `Model.vg` / `Model.lg` cache CUDA-graph / launch state keyed by buffer handles. Freeing a
   layer's buffers invalidates every entry that refers to them.
4. **GPU DeltaNet state.** With `st.gdn_gpu`, the conv/ssm state lives only on the GPU. A layer leaving the GPU means
   the state must be read back first (the existing "read back before any CPU step" logic does this when the gates
   above turn false, so it must run *before* the buffers are freed).

## Design

### A. Per-step view instead of a mutable `cl`

- Store `cl: Vec<std::sync::Mutex<Option<std::sync::Arc<CLayer>>>>` (or a single `Mutex<Vec<Option<Arc<CLayer>>>>`).
- `step_hook()` already runs at the start of every step and applies the expert-cache target. Extend it: take one
  snapshot `Vec<Option<Arc<CLayer>>>` per step (48 Arc clones, < 1 us) and keep it in the step (e.g. in `State`
  or a thread-local). All readers use the snapshot, so nothing changes mid-step and the hot path takes no locks.
- Offload or onload happens only in `step_hook()`: replace a slot with `None` or `Some(new)`. The GPU memory is freed
  when the last `Arc` drops (the end of the step holding the old snapshot).

### B. Make the all-layers paths degrade instead of unwrap

- Replace `self.cl[nl].as_ref().unwrap()` in `forward_multi_x()`, `layer_cube_multi_p()` and `dense_gpu_only()` by
  snapshot lookups that return `None` (callers fall back to the per-layer CPU path), and make sure every caller is
  gated by `multi_ok()` on the same snapshot.
- When any layer is offloaded: `multi_ok()` is false (MTP speculative decode and multi-token prefill turn off), and
  the graph gate in `forward_inner()` is false. Before freeing, read the GPU DeltaNet state back to the host
  (`gdn_gpu = false`). Clear `vg` / `lg` entries (simplest: clear both maps on any change to the layer set).
- The first offloaded layer costs the most (it switches off the all-GPU fast paths); each further layer only moves its
  own matmuls to the CPU. So offload in **groups of 4 layers (~0.5 GiB)**, enough to clear the deficit plus 256 MiB.

### C. Ladder rung and recovery

- New rung **2b** in `limits::vram_watch()` (after K/V to host, before the degraded wait): set a target count
  `GPU_LAYERS_TGT` (atomic, like `XC_TGT`). `step_hook()` applies it. Log a `vram_pressure` event
  `{"action": "offload_layers", "from": n, "to": m, "free_mib": ...}`.
- Choose layers from the **end of the stack backwards** (keeps the early layers, whose outputs feed everything, on the
  GPU, and matches how `attach_gpu` already leaves the tail on the CPU under its reserve).
- Recovery with hysteresis: after `free >= floor + PYRE_KV_REGPU_MB + group size` for 120 s, raise the target by one
  group (re-upload from the host copies via `cb.upload_mix`, ~0.5 s per group, at a step boundary). 120 s cooldown
  between groups. When all layers are back, the fast paths turn on again by themselves (the gates are recomputed per
  snapshot). K/V comes back after the layers (rung order reversed).
- Budget accounting: adjust `cb.budget` / `g.budget` on each change so `headroom()` and the expert-cache plan stay right.

### D. Testing without breaking prod

- `PYRE_QN_OFFLOAD_TEST=N`: after `attach_gpu`, offload N layers through the same `step_hook` path, then raise back
  to 48 after `PYRE_QN_OFFLOAD_TEST_S` seconds. This exercises both directions in a maintenance window, without
  needing another app to take VRAM.
- Check: greedy output on a fixed prompt is identical for N = 0, 4, 48 (CPU and GPU projections use the same Q8
  arithmetic, as the existing reserve fallback already relies on); decode tok/s per N; no growth in `vg`/`lg`; free
  VRAM rises by ~124 MiB per layer.
- Unit-test the target selection (`deficit -> groups`, hysteresis timing) as a pure function, like `next_rung()` in
  PR #31.

## Expected effect

At the Nov 24 15:32 incident (free 1007 MiB vs floor 1024, expert cache already at min), rung 2 (K/V to host) alone
should have been enough. Phase 2 matters when another app takes GBs (a game, a second model): today pyre stops; with
phase 2 it would drop to e.g. 32/48 layers on the GPU, run slower (all-CPU was ~3x slower in the 15:36 accident), and
climb back when the other app exits.