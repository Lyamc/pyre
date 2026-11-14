# pyre handoff: state as of 2026-11-14 13:40 (America/Edmonton, UTC-6)

## Project
- Repo: https://github.com/Lyamc/pyre, default branch `qwen-int` (this doc is a docs-only commit on top of `6049385`, the PR #18 merge). Branch `master` is being fast-forwarded to this same head; `qwen-int` stays the default branch.
- pyre is a native Rust/Burn/CubeCL inference server. It covers LLM chat (Qwen3.8-Flash-Next MoE), STT, TTS, diarization, image/"cabin" workflows and planned 3D tooling, with a web UI.
- Host machine: **HYTE** (Windows, RTX 3060 12 GB, machineId `39c6a724-91a3-4133-b282-7864b570787c`). Model: Qwen-Next IQ3_S GGUF mix at `C:\AI\models\models\IQ3_S` (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`). MTP drafter weights are at `C:\AI\models\mtp\mtp.safetensors`. STT weights live under `C:\AI\models\stt` (`WHISBURN_MODELS_DIR`).

## Hard rules from the owner (Lyam)
- **Native Rust/Burn only.** No Python, ONNX or external inference service at runtime, not even temporarily. Weight conversion is done in Rust too.
- Each vendored dependency lives in its **own GitHub repo, pulled in as a submodule**.
- Normal pushes only. **No force-push and no history rewrite.**
- `C:\Build\ai\pyre` is the single main checkout. It has **uncommitted, incomplete edits by Lyam** (src/main.rs, src/cabin.rs, docs/3d/plan.md, ...). **Don't touch, commit or build them.** Use separate worktrees or clones such as `C:\Build\ai\pyre-<topic>` or `C:\Temp\qnx\...`.
- Build on HYTE only with `C:\Temp\qnx\build.ps1`, in the foreground, with no pipes or redirects.
- For production restarts use `C:\Temp\qnx\int\srv.ps1` or `restore_prod.ps1`; for a graceful stop use `C:\Temp\qnx\stopgrace.ps1` (that script is not under `\int`).
- Keep at least **1 GB VRAM free**. Keep **pyre_nano on :8102** running. Host binds to `0.0.0.0` by default.
- Don't change system settings without approval, never print secrets, treat `Z:\Manga` as read-only, and don't delete Lyam's files without approval.
- Expose model capabilities (text/image input, etc.) via the API.
- The owner prefers quick checks over extensive tests.

## Production (live now)
- `C:\Temp\qnx\int\pyre_int69.exe` (built 11:35 AM America/Edmonton from PR #18 head `93ccccb`; same code as qwen-int merge `6049385`). Server PID 45876, watchdog PID 60484, started 11:38 AM America/Edmonton on 2026-11-14, `0.0.0.0:8099`. Production runs the exact launch profile with a resident MTP drafter. `/api/status` reports `draft_len` 2, source `model`, `adaptive` true, drafter `mtp`. Fast mode is on; `xexact` is false.
- Loaded at startup (and in `srv.ps1` / `restore_prod.ps1` / `Start-Pyre.ps1`): IQ3_S LLM on CUDA, MTP drafter, and STT on CPU: Whisper base, Parakeet TDT v2, and Fluid Speech Pro (`--load stt:cpu=base --load stt:cpu=parakeet-tdt-0.6b-v2 --load stt:cpu=fluid-speech-pro`). Env: `WHISBURN_MODELS_DIR=C:\AI\models\stt`, `PYRE_MTP=C:\AI\models\mtp\mtp.safetensors`. MMS TTS and diarization are still omitted because those HF caches are missing.
- **Fluid Speech Pro is loaded but still marked experimental.** The UI "Experimental models" checkbox (`experimental=true`) is required to use it. `PYRE_STT_EXPERIMENTAL` is **not** set globally, because that switch also enables long-audio chunking for the other STT models.
- After startup, Whisper `large-v3-turbo` was also loaded on CPU (first seen in `/api/status` around 12:57 PM America/Edmonton). It is **not** in the launch scripts, so a restart will drop it unless someone adds it.
- Free VRAM was about **2.7 GB** just after start. A live check at 13:38 America/Edmonton showed about **1952 MiB free** (10162 MiB used of 12288), still above the 1 GB floor (`vram_reserve` 1228 MiB, floor 1024 MiB). Expert cache is 3.38 GiB resident, hit rate about 70% overall / 73% recent (better than the 61.5% measured on the int68 bench). Context is configured for 131072 tokens at 4-bit KV; with current free VRAM a 128k window does not fit (needs ~2632 MiB KV while keeping the floor; a 16k window would). Process RSS was about 64 GB of 103 GB RAM when checked.
- Launch scripts `srv.ps1`, `restore_prod.ps1` and `Start-Pyre.ps1` point at int69. Backups in `C:\Temp\qnx\int`: `*_int68_backup.ps1` (int68, pre-Fluid in the default audio list), `*_int68fluid_backup.ps1` (int68 + Fluid Speech Pro), plus older `*_int67_backup.ps1`, `*_int66_backup.ps1` and `*_premtp_backup.ps1`.
- pyre_nano is still running as PID 39652 on :8102 (up since 2026-11-06).
- Speed: int68 (same LLM code as int69) measured **20.6 tok/s warm**, 61.5% expert cache hit. Short recent int69 decodes in `/api/status` are typically about 17–21 tok/s; long-context prompts are much slower. MTP acceptance on the live process was about 68% (318 accepted of 467 drafted) when checked.

## Client audio extraction (PR #18)
- Merged into qwen-int as `604938522ebb69a340c7846afbda5d19794cfaf0` ([PR #18](https://github.com/Lyamc/pyre/pull/18), head `93ccccb`). Production int69 already runs this code; nothing was redeployed after the merge.
- Uploading a video (e.g. `.mp4`) in the Audio tab extracts **16 kHz mono WAV on the client** and sends only that audio to the server. A 5.8 MB test video became a 450 KB upload and transcribed correctly.
- Mediabunny is vendored as a submodule: `vendor/mediabunny` → https://github.com/Lyamc/mediabunny.git, branch `pyre-dist` at `4d5270b`.
- Over localhost or HTTPS, Mediabunny / WebCodecs can decode large files in pieces. **Plain-HTTP LAN addresses** (e.g. `http://<HYTE-IP>:8099`) cannot use WebCodecs, so the UI falls back to the browser's built-in WebAudio decoder, which loads the whole file into memory.

## Why pyre was slower than Strata (and what changed)
These were the original gaps versus Strata (30–50 tok/s on the same model). PR #15 closed most of them; int67 was about 7–10 tok/s, int68/int69 are about 20 tok/s warm. A remaining gap versus Strata is still open.
1. The "IQ3_S" file is a mix: gate/up use IQ3_XXS, IQ2_S, IQ3_S and IQ4_XS, and down uses IQ4_NL or Q2_0. Before PR #15 the GPU expert cache only handled IQ2_S, IQ2_XXS, IQ1_M and Q2_0, so only 5 of 48 layers could be cached (9–12% hit rate). **Fixed in PR #15:** IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS experts are cacheable; live hit rate is around 62–70%.
2. The CPU fallback `row_dot_deq` used to dequantize to f32 and compute scalar dots. **Fixed in PR #15** with AVX2 integer dots.
3. Dense weights used to be converted to Q8 (6.35 GiB, against Strata's 1.47 GiB). **PR #15 keeps IQ4 dense weights native.**
4. Strata partly reached its speed by running VRAM down to about 125 MiB free, which pyre's 1 GB floor still forbids.
5. MTP drafting was broken, then fixed in PR #17. It adds a modest gain once verify is cheaper (PR #15).

## Open work
| Item | State | Next step |
|---|---|---|
| PR #15 exactness | Exact mode output diverges from int67 exact after ~15 tokens (PR #15 keeps more dense IQ4 native) | Decide whether exact mode should stay bit-identical to the CPU path; if so, gate native IQ4 dense off under `PYRE_QN_XEXACT=1`. |
| `--translate` (en→en and similar) | Falls back to Whisper base and is poor | Improve translation routing / quality. |
| Parakeet v3 (French about 70% WER) and Nemotron STT (about 16% WER) | Behind `PYRE_STT_EXPERIMENTAL` | Fix accuracy (see `docs/stt/plan.md`; test data in `C:\Temp\qnx\stt13`). |
| Fluid Speech Pro UX | Loaded on int69; still requires the "Experimental models" checkbox | Only flip `PYRE_STT_EXPERIMENTAL` (or an equivalent narrower flag) after deciding whether long-audio chunking should come on with it. |
| 3D suite (image to 3D, 3D to 3D, auto rigging, mesh tools, mesh to SDF) | Plan only (`docs/3d/plan.md`, PR #14) | Not started. The plan uses TripoSR plus geometric tools; IDOL and Pixal3D are too big for 12 GB. |
| Native Chatterbox voice cloning (Burn) | Full waveform integrated (d626cf3), about 50x realtime on CPU | Audible quality unverified; check quality and optimize before enabling it by default. MMS TTS weights are currently missing on HYTE. |
| TTS and diarization model caches | Missing on HYTE, not loaded | Restore the caches, then re-add them to the launch args. |
| FAST mode repetition mitigation | Unchecked | Verify. |
| HYTE iGPU | Idea only | Evaluate only if it gives a net gain. |
| Worktree / leftover cleanup | Listed below; not deleted | Ask before deleting anything. |

## Recently done
- **int69 in production** (this afternoon): Fluid Speech Pro added to `srv.ps1` / `restore_prod.ps1` / `Start-Pyre.ps1`; browser audio extraction from PR #18 shipped in the same binary. Restarted via the production scripts; nano on :8102 was left running; no rollback.
- PR #18 (merged `6049385`): browser-side audio extraction with Mediabunny and a WebAudio fallback, so video uploads send only 16 kHz mono WAV. Submodule `vendor/mediabunny` from [Lyamc/mediabunny](https://github.com/Lyamc/mediabunny) branch `pyre-dist` (`4d5270b`).
- **int68 in production** (earlier today, then superseded by int69): qwen-int at `1451ccb` (PRs #15 + #16 + #17 + the previous handoff). About 20.6 tok/s warm, 61.5% expert cache hit, VRAM stayed above the 1 GB floor.
- PR #15 (merged `c4298bc`): IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS experts cacheable on the GPU, AVX2 integer CPU dots, native IQ4 dense, per-size cache slots (XCLASS default on). HYTE: fast mode ~15 tok/s cold / ~20 tok/s warm (int67 ~9.6), cache hits 11% -> 62%. VRAM fix `d1e8a60`: the cache is planned from live free VRAM minus later allocations (resident drafter + `PYRE_QN_XLATE_MB`, default 1792 under XEXACT / 512 otherwise) and keeps max(`PYRE_VRAM_FLOOR_MB`, `--vram-reserve`) free, because an XCLASS cache cannot be shrunk by the watcher. Exact mode starts with a 2.83 GiB cache and min free ~1283 MiB (18 tok/s). `scripts/hyte_iq3s.ps1` takes `-Exe` / `-NoBuild`. Quick test harness: `C:\Temp\qnx\iq3bench\run2.ps1`.
- PR #16 (merged `f37385c`): Fluid Speech Pro STT using the Parakeet log-mel front end, and the `pyre transcribe` CLI. (The earlier handoff incorrectly listed "browser-side audio extraction" here; that work landed in PR #18.)
- PR #17 (merged `60b5824`, drafting fix): `spec::draft_len()` no longer caches its fallback of 0; the model default comes from `spec::set_model_default`. `/api/status` reports draft_len, source, adaptive and drafter. Adaptive draft length skips drafting when it doesn't pay off and probes every 24 tokens. A/B testing gave +3.3–4.7%, and exact mode (`PYRE_QN_XEXACT=1 --fast false`) was byte-identical to int67 before the PR #15 dense-IQ4 change. Benchmarks are in `C:\Temp\qnx\draftbench\`.
- MTP weights were packed by a Rust tool (`C:\Temp\qnx\mtppack`), and `PYRE_MTP` was added to the launch scripts.
- Model capabilities API (PR #12), faster Whisper plus Parakeet/Nemotron (PR #13), and 3D/STT plans (PR #14).
- pyre trees were consolidated into `C:\Build\ai\pyre`; the `qwen-moe-graph` branch was pushed to origin.
- Strata (`C:\Build\ai\strata-src`) no longer auto-opens a browser (`--open` removed, `open_browser:false`). It is currently **stopped**.
- `master` was first created at `1451ccb` (the previous handoff). After this commit it should match `qwen-int`.

## Worktrees and leftovers on HYTE (safe to delete, but ask first)
- `C:\Build\ai\pyre-draft`: fix/mtp-drafting at 90cfd08, merged and safe to remove.
- `C:\Build\ai\pyre-fluid`: PR #16 test (merged, safe to remove).
- `C:\Build\ai\pyre-iq3`: PR #15 work, local branch `iq3-fix` (pushed, merged, safe to remove).
- `C:\Build\ai\pyre-handoff`: worktree used to commit the previous handoff (safe to remove).
- `C:\Temp\qnx\pyre-handoff-int69`: clone used to commit this docs update (safe to remove after push).
- `C:\Build\ai\mediabunny-dist`: leftover from vendoring Mediabunny (safe to remove once confirmed unused).
- `C:\Users\Lyam\AppData\Local\Temp\pyreui_xfer`: temp transfer dir from the UI/audio work (safe to remove).
- `C:\Temp\qnx\stt16\srv69*.log`: int69 launch logs (safe to remove if no longer needed).
- Two empty locked folders, `C:\Build\ai\pyre-qwen-int` and `pyre-nano-int`, can be removed after a reboot.
- `C:\Users\Lyam\patch.ps1` is possibly a leftover. Ask before deleting it.

A box copy of this file also lives at `/workspace/handoff/pyre-handoff-2026-11-14.md`.
