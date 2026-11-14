# pyre handoff: state as of 2026-11-14 10:00 (America/Edmonton)

## Project
- Repo: https://github.com/Lyamc/pyre, default branch `qwen-int` (head `c4298bc` after the PR #15 merge; PR #16 merged as `f37385c`, PR #17 as `60b5824`). Branch `master` was created from qwen-int after this doc landed; `qwen-int` stays the default branch.
- pyre is a native Rust/Burn/CubeCL inference server. It covers LLM chat (Qwen3.8-Flash-Next MoE), STT, TTS, diarization, image/"cabin" workflows and planned 3D tooling, with a web UI.
- Host machine: **HYTE** (Windows, RTX 3060 12 GB). Model: Qwen-Next IQ3_S GGUF mix at `C:\AI\models\models\IQ3_S`. MTP drafter weights are at `C:\AI\models\mtp\mtp.safetensors`.

## Hard rules from the owner (Lyam)
- **Native Rust/Burn only.** No Python, ONNX or external inference service at runtime, not even temporarily. Weight conversion is done in Rust too.
- Each vendored dependency lives in its **own GitHub repo, pulled in as a submodule**.
- Normal pushes only. **No force-push and no history rewrite.**
- `C:\Build\ai\pyre` is the single main checkout. It has **uncommitted, incomplete edits by Lyam** (src/main.rs, src/cabin.rs, docs/3d/plan.md, ...). **Don't touch, commit or build them.** Use separate worktrees or clones such as `C:\Build\ai\pyre-<topic>`.
- Build on HYTE only with `C:\Temp\qnx\build.ps1`, in the foreground, with no pipes or redirects.
- For production restarts use `C:\Temp\qnx\int\srv.ps1` or `restore_prod.ps1`; for a graceful stop use `stopgrace.ps1`.
- Keep at least **1 GB VRAM free**. Keep **pyre_nano on :8102** running. Host binds to `0.0.0.0` by default.
- Don't change system settings without approval, never print secrets, treat `Z:\Manga` as read-only, and don't delete Lyam's files without approval.
- Expose model capabilities (text/image input, etc.) via the API.

## Production (live now)
- `C:\Temp\qnx\int\pyre_int67.exe` (code = qwen-int 60b5824, i.e. **without** PR #15/#16). Server PID 15100, watchdog PID 49776, `0.0.0.0:8099`. Production runs the exact profile (resident MTP drafter); at load the watcher shrinks its expert cache 3.5 -> 2.5 GiB to hold the floor.
- Loaded: IQ3_S LLM, MTP drafter (draft_len 2, adaptive), Whisper base and Parakeet TDT v2 on CPU (`WHISBURN_MODELS_DIR=C:\AI\models\stt`).
- Free VRAM is about 1.6 GB after startup.
- Launch scripts `srv.ps1`, `restore_prod.ps1` and `Start-Pyre.ps1` point at int67. The int66 backups are `*_int66_backup.ps1`, and the pre-MTP backups are `*_premtp_backup.ps1`.
- pyre_nano runs as PID 39652 on :8102.
- Speed is about **7–10 tok/s**, with MTP acceptance of 63–100%. Strata reached 30–50 tok/s on the same model. A qwen-int build with PR #15 measured ~15–20 tok/s on HYTE but is **not deployed yet** (next step: build qwen-int c4298bc as int68 and switch production with srv.ps1).

## Why pyre is slower than Strata (diagnosed)
1. The "IQ3_S" file is a mix: gate/up use IQ3_XXS, IQ2_S, IQ3_S and IQ4_XS, and down uses IQ4_NL or Q2_0. The GPU expert cache only handles IQ2_S, IQ2_XXS, IQ1_M and Q2_0 (`qn_cube.rs` around line 5947, `xtype`). So only 5 of 48 layers can be cached, and the hit rate is 9–12%.
2. The CPU fallback `row_dot_deq` (`qdot.rs`) dequantizes to f32 and computes scalar dot products.
3. Dense weights are converted to Q8 (6.35 GiB, against Strata's 1.47 GiB).
4. Strata partly reached its speed by running VRAM down to about 125 MiB free, which pyre's floor forbids.
5. MTP drafting was broken. It is now fixed (PR #17), but it only adds about 4% until verify gets cheaper (see PR #15).

## Open work
| Item | State | Next step |
|---|---|---|
| Deploy qwen-int c4298bc (PR #15 + #16) as int68 | Built and tested in worktree `C:\Build\ai\pyre-iq3`, not deployed | Build in a clean worktree, copy to `C:\Temp\qnx\int\pyre_int68.exe`, point `srv.ps1`/`restore_prod.ps1` at it, restart, check `/api/status` (draft_len 2) and free VRAM >= 1 GB. |
| PR #15 exactness | Exact mode output diverges from int67 exact after ~15 tokens (PR #15 keeps more dense IQ4 native) | Decide whether exact mode should stay bit-identical to the CPU path; if so, gate native IQ4 dense off under `PYRE_QN_XEXACT=1`. |
| Fluid Speech Pro (PR #16) | Merged; English OK (4 s clip), `pyre transcribe file.mp4` works | `--translate` falls back to Whisper base and is poor; French/WER not measured. |
| Parakeet v3 (French about 70% WER) and Nemotron STT (about 16% WER) | Behind `PYRE_STT_EXPERIMENTAL` | Fix accuracy (see `docs/stt/plan.md`; test data in `C:\Temp\qnx\stt13`). |
| 3D suite (image to 3D, 3D to 3D, auto rigging, mesh tools, mesh to SDF) | Plan only (`docs/3d/plan.md`, PR #14) | Not started. The plan uses TripoSR plus geometric tools; IDOL and Pixal3D are too big for 12 GB. |
| Native Chatterbox voice cloning (Burn) | Full waveform integrated (d626cf3), about 50x realtime on CPU | Audible quality unverified; check quality and optimize before enabling it by default. MMS TTS weights are currently missing on HYTE. |
| TTS and diarization model caches | Missing on HYTE, not loaded | Restore the caches, then re-add them to the launch args. |
| FAST mode repetition mitigation | Unchecked | Verify. |
| HYTE iGPU | Idea only | Evaluate only if it gives a net gain. |

## Recently done
- PR #15 (merged `c4298bc`): IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS experts cacheable on the GPU, AVX2 integer CPU dots, native IQ4 dense, per-size cache slots (XCLASS default on). HYTE: fast mode ~15 tok/s cold / ~20 tok/s warm (int67 ~9.6), cache hits 11% -> 62%. VRAM fix `d1e8a60`: the cache is planned from live free VRAM minus later allocations (resident drafter + `PYRE_QN_XLATE_MB`, default 1792 under XEXACT / 512 otherwise) and keeps max(`PYRE_VRAM_FLOOR_MB`, `--vram-reserve`) free, because an XCLASS cache cannot be shrunk by the watcher. Exact mode now starts with a 2.83 GiB cache and min free ~1283 MiB (18 tok/s). `scripts/hyte_iq3s.ps1` takes `-Exe` / `-NoBuild`. Quick test harness: `C:\Temp\qnx\iq3bench\run2.ps1`.
- PR #16 (merged `f37385c`): Fluid Speech Pro STT using the Parakeet log-mel front end, `pyre transcribe` CLI, browser-side audio extraction.
- PR #17 (drafting fix): `spec::draft_len()` no longer caches its fallback of 0; the model default comes from `spec::set_model_default`. `/api/status` reports draft_len, source, adaptive and drafter. Adaptive draft length skips drafting when it doesn't pay off and probes every 24 tokens. A/B testing gave +3.3–4.7%, and exact mode (`PYRE_QN_XEXACT=1 --fast false`) is byte-identical. Benchmarks are in `C:\Temp\qnx\draftbench\`.
- MTP weights were packed by a Rust tool (`C:\Temp\qnx\mtppack`), and `PYRE_MTP` was added to the launch scripts.
- Model capabilities API (PR #12), faster Whisper plus Parakeet/Nemotron (PR #13), and 3D/STT plans (PR #14).
- pyre trees were consolidated into `C:\Build\ai\pyre`; the `qwen-moe-graph` branch was pushed to origin.
- Strata (`C:\Build\ai\strata-src`) no longer auto-opens a browser (`--open` removed, `open_browser:false`). It is currently **stopped**.

## Worktrees and leftovers on HYTE
- `C:\Build\ai\pyre-draft`: fix/mtp-drafting at 90cfd08, merged and safe to remove.
- `C:\Build\ai\pyre-fluid`: PR #16 test (merged, safe to remove).
- `C:\Build\ai\pyre-iq3`: PR #15 work, local branch `iq3-fix` (pushed, merged, safe to remove).
- `C:\Build\ai\pyre-handoff`: worktree used to commit this doc (safe to remove).
- Two empty locked folders, `C:\Build\ai\pyre-qwen-int` and `pyre-nano-int`, can be removed after a reboot.
- `C:\Users\Lyam\patch.ps1` is possibly a leftover. Ask before deleting it.
