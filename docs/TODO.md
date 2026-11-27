# TODO

## Inference optimisation gaps vs [tensorfish/inference-optimisation-skill](https://github.com/tensorfish/inference-optimisation-skill)

Compared on 2026-11-25 (America/Edmonton) against `qwen-int` at `d9b8eec` and the skill's `SKILL.md` / `references/{levers,ceilings,methodology,publishing,gotchas}.md` on `main`. Target is the live HYTE setup: Qwen-Next IQ3_S on an RTX 3060 12 GB, 1 GB VRAM floor, native Rust/Burn only.

Each open item is one line: what it is, why it helps pyre, and rough effort (**S** a day or two, **M** about a week, **L** weeks). Items already in pyre, already in flight, or a poor fit are listed so they are not filed again.

### Already in pyre (not gaps)

- CubeCL autotune and compiled-kernel cache persists across restarts (`%LOCALAPPDATA%\cubecl\default.db`, with a hardware-change notice).
- CUDA decode / verify / MTP graphs (`PYRE_QN_VGRAPH`, `PYRE_MTP_GRAPH`; `PYRE_QN_MOEGRAPH` opt-in) and load-time `prewarm_verify`.
- MTP drafter plus n-gram fallback, adaptive draft length (PR #17), and a pruned draft LM head (`PYRE_MTP_DRAFT_VOCAB`, default `en`).
- Verify runs on the decode path, not the prefill attention kernel.
- Prefix cache with `cached_tokens` in usage, RAM sharing (PR #25), and disk snapshots (PR #22); PR #27 (cache key on LLM args only) is already in int79.
- Prefill chunk 1024 (PR #19), split-K prefill GEMMs, native IQ3/IQ4 kernels and a GPU expert cache (PR #15).
- KV 4-bit default / 8-bit option, context past 256k (1M hard ceiling).
- 1 GB VRAM floor, VRAM ladder (PR #31), expert-cache governor, optional expert working-set lock (PR #24).
- `stream_options.include_usage` so clients can count real tokens, not SSE chunks.
- Graceful drain then stop (`stopgrace.ps1`).
- One-shot benches (`scripts/hyte_iq3s.ps1`, `bench_concurrency.py`) and optional `PYRE_SPEC_TIME` / `PYRE_SPEC_TRACE` (acceptance by draft position is already printed there, not in `/api/status`).

### In flight (do not redo)

- PR #33 (draft, held): allow allocations in the VRAM-ladder band so a 768–1024 MiB free window does not abort the server. Needs Lyam's OK to merge and deploy.
- PR #32 (draft): design for runtime layer offload (VRAM ladder phase 2).
- PR #30 (draft): `kv_pull` reads only the missing K/V rows (snapshot readback).
- HANDOFF next ideas already named: skip re-reading unchanged GDN state on snapshots; GPU expert streaming for large prefill chunks; drop cache slots without a full rebuild.

### Do first — methodology (cheap, unlocks the rest)

- [ ] **Ceilings from measured bandwidth and FLOPs.** Write `docs/THEORY.md` and a small native or PowerShell ceiling helper from a real device-copy bandwidth and a large GEMM, counting this checkpoint's bytes per token (dense + routed experts + LM head + KV + GDN state). Live decode is 16.8–17.1 tok/s and prefill about 40–47; without a ceiling it is unclear how much of the remaining gap vs Strata (about 30–50 tok/s) is physics vs kernels. **S**
- [ ] **Hill-climb harness.** Interleaved control / candidate / control, medians of at least six decode samples, unique cold prompts sized with the real tokenizer, token counts from usage stats, a `ledger.jsonl`, and a `PLAN.md` that records failures. Today's scripts are short one-shots, so ±5–10% noise can look like a win. Prefer PowerShell or Rust talking to the OpenAI API; do not take a Python or ONNX dependency inside the server. **S–M**
- [ ] **Standing stage profile.** Break prefill and decode into GEMM, attention, MoE, GDN, host sampler, and CPU-expert wait, and keep the current split next to the ceilings. Decode is already known to be CPU-expert bound; prefill's 40–47 tok/s still needs a fresh split before picking the next kernel. **S**

### High impact on HYTE speed

- [ ] **GPU expert streaming for large prefill (and more decode hits).** Routed experts still run on the CPU, and HANDOFF already calls this the main remaining speed gap vs Strata. Streaming or packing experts onto the GPU for 1024-token prefill chunks, and holding a larger working set in the expert cache, is the highest-leverage kernel lever. **L**
- [ ] **Int8 prefill GEMMs, labelled lossy.** The skill measured about 1.89× faster linears at +0.17% NLL with decode kept bit-exact. That targets compute-bound prefill on Ampere INT8 tensor cores. Gate it, measure quality, leave decode exact. **L**
- [ ] **Grouped MoE GEMM packing.** Avoid padding cliffs and use larger M-blocks that reuse decoded weights (skill: +7% prefill, +22% at c16). This pays once more MoE work is on the GPU, so do it after expert streaming. **M–L**

### Medium

- [ ] **GPU sampling.** Logits are copied to the host and sampled in Rust (`src/sampler.rs`). The skill saw about 3 ms/step (about +10% c1) moving the sampler to the GPU; at ~17 tok/s that is a few percent if the host sample is in the same ballpark. **M**
- [ ] **Kernel tile and thread sweep for the 3060.** The skill found defaults copied from a bigger card were wrong (B70: 4096→1024 threads, +17%). CubeCL autotune helps, but several `qn_cube` tiles are fixed (for example 128×64). Profile, then sweep. **M**
- [ ] **Long-context needle gate.** Exact-match needles at about 50% and 100% of the working context before a config is called "best". Pyre advertises >256k / 1M YaRN; speed benches do not prove retrieval. Fold into the harness. **S–M**
- [ ] **Draft-length sweep by context, and acceptance-by-depth in the default report.** Adaptive k already exists, and `PYRE_SPEC_TIME` prints per-position acceptance, but that is not in `/api/status` or a standard bench. The skill found three drafts can lose at 128k. Sweep k at 4k / 32k / 128k and publish acceptance by depth. **S** once the harness exists
- [ ] **Faster long-context attention, and a KV-dtype speed check.** Recheck 4-bit and 8-bit KV attention throughput at 32k–256k (the skill saw FP8 KV halve one flash-attention kernel until a dequant + LSE merge recovered it), then improve the CubeCL kernel if that stage is the bottleneck. Native only; do not pull in a Python attention library. **M–L**
- [ ] **Startup: recaptured CUDA graphs.** Autotune and kernels already persist; graphs are recaptured every boot in `prewarm_verify`. Measure startup against the disk / page-cache floor and cut recapture if it is a large slice. **S** to measure, **M** to cut
- [ ] **Qualification gates on a clean pinned build.** The skill's six gates (load, chat, reasoning, tools, needle, speed) plus a clean-vs-best A/B. Complements draft PR #29 (GPU-free smoke). **M**

### Architectural

- [ ] **Concurrent multi-stream decode (cN).** The largest architectural gap: the qwen-next worker is single-slot ("requests run one at a time"). The skill treats cN and a KV pool of N×context as first-class, which needs per-request GDN / SSM state slots, a sized KV pool, and a concurrent full-length soak. On a 12 GB card with a 1 GB floor and CPU experts, the payoff may be small for the usual one-agent workload; it is still required if we ever advertise N users at full context. **L**

### Does not fit, or defer

- **Do not vendor the skill's Python `bench.py` / `ceiling.py` into the pyre runtime.** A measurement-only client that talks to the OpenAI API is fine; inference stays Rust/Burn, with no Python or ONNX in the server.
- **FP8 tensor-core decode or prefill on the 3060.** Ampere has no FP8 units. IQ / INT4 is already used; INT8 prefill is the lossy compute trade that fits this card.
- **vLLM, SGLang, llama.cpp, and ExLlama flags or stack-specific gotchas.** Useful as priors only, not as work items.
- **EAGLE or a second draft model in VRAM.** MTP, n-gram, and a pruned draft head already exist. A second model is a poor fit under the 12 GB / 1 GB floor unless MTP plateaus with evidence.
- **Unified-memory cgroup accounting.** HYTE's 3060 is a discrete NVIDIA GPU. Only relevant if the iGPU path is pursued.
- **Raising or removing the 1 GB VRAM floor to chase Strata's ~125 MiB free.** That is policy, not a missing feature. PR #33 is the related ladder bugfix.
- **Eager vs mmap weight loading** (a vLLM gotcha in the skill) is not a pyre gap: GGUF experts are intentionally mmap'd on the CPU, and PR #24 already offers a soft working set / VirtualLock.
