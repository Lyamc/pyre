# pyre handoff: state as of 2026-11-24 12:00 (America/Edmonton, UTC-6)

## Project
- Repo: https://github.com/Lyamc/pyre, default branch `qwen-int` (this doc is a docs-only commit on top of `377ae13`, the PR #25 merge). `master` is fast-forwarded to the same head; `qwen-int` stays the default branch.
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

## Production (live now): int79
- `C:\Temp\qnx\int\pyre_int79.exe`, built from a local, unpushed merge `a6058be` (branch `local/int79` in the `pyre-prefill` worktree) = qwen-int `088eaee` (**PR #31 VRAM ladder, merged Nov 24 20:20 with Lyam's OK**) + PR #27 (prefix-cache key on LLM args only; still a draft, and already in int78). Deployed 20:33 America/Edmonton on 2026-11-24 at a Bend-agent step boundary; after the ladder tests the live server is the latest `restore_prod.ps1` start (see `C:\Temp\qnx\int\restore_prod.log`). `0.0.0.0:8099`; pyre_nano PID 23936 on `0.0.0.0:8102`. Lyam's llama-server (127.0.0.1:8081) also runs on this GPU; leave it alone.
- Launch: `restore_prod.ps1` / `Start-Pyre.ps1` -> `srv.ps1 -exe pyre_int79.exe -tag int79`, the same model/audio args as int78 (IQ3_S on CUDA, MTP drafter; Whisper base + large-v3-turbo, Parakeet v2 and Fluid Speech Pro on CPU). `srv.ps1` keeps the previous run's `C:\Temp\qnx\server_<tag>.log/.err.log` as `*.prev`.- Env (set via `-e @{...}` in `restore_prod.ps1` / `Start-Pyre.ps1`; `srv.ps1` sets `PYRE_VRAM_FLOOR_MB=1024`, `PYRE_MTP`, `PYRE_QN_TOKENIZER`):

  | Var | Value | From | Effect |
  |---|---|---|---|
  | `WHISBURN_MODELS_DIR` | `C:\AI\models\stt` | - | STT weights |
  | `PYRE_PREFILL_CHUNK`, `PYRE_QN_BPG_CHUNK` | `1024` | #19 | prefill chunk (2.7k-token prompt 30.7 -> ~40-47 tok/s) |
  | `PYRE_QN_PCACHE_DIR` | `C:\Temp\qnx\pcache` | #22 | prefix-cache snapshots persisted; restarts resume long prompts (verified: resumed at 15505 tok, 136 ms restore) |
  | `PYRE_QN_WS_MIN_GB` | `auto` | #24 | soft min working set = experts + 24 GiB (70.8 GiB); halved disk page-ins in decode |
  | `PYRE_XC_GROW_ABOVE_MB` / `PYRE_XC_GROW_KEEP_MB` | `1024` / `768` | #23 | expert cache regrows above floor+1024 MiB, keeping floor+768 free |
| `PYRE_KV_REGPU_MB` | `256` | #31 | K/V returns to the GPU once free >= floor + 256 MiB for 60 s (the default 1536 would keep it on the host for good at prod's usual 1.1-2 GB free) |

- New tunables (defaults unless noted): `PYRE_QN_ATTN_SC_MB` (256), `PYRE_QN_PCACHE_ALL` (0) [#19]; `PYRE_QN_PCACHE_DISK_N` (2), `PYRE_QN_PCACHE_MIN_FREE_GB` (10), `PYRE_QN_PCACHE_SAVE_S` (900; each save writes the whole snapshot, ~1-3 GB, so mind SSD wear) [#22]; `PYRE_XC_SHRINK_BELOW_MB` (512), `PYRE_XC_GROW_STEP_MB` (256), `PYRE_XC_MIN_MB` (128), `PYRE_XC_HOLD_S` (600) [#23]; `PYRE_QN_LOCK_EXPERTS` (0; 1 = VirtualLock ~47 GiB, dedicated box only) [#24].
- Live numbers (agent workload, ~15-30k-token contexts): decode 16.8-17.1 tok/s (int72 16.2-16.5), prefill ~40-47 tok/s, VRAM 1.7-2.2 GB free, expert cache 0.5-1.1 GB, recent hit rate 0.33-0.46. Decode is bound by CPU expert compute / memory bandwidth (24 pinned expert workers, AVX2+FMA), not by the GPU cache size or paging.
- **Rollback to int78** (int77 + #27, old 1024 MiB watchdog): copy `srv_int78_backup.ps1`, `restore_prod_int78_backup.ps1`, `Start-Pyre_int78_backup.ps1` over the live scripts in `C:\Temp\qnx\int`, then stopgrace + `restore_prod.ps1`. Older: `*_int78pre_backup.ps1` (int77 + turbo), `*_int77_backup.ps1`, `*_int76_backup.ps1` (#19-#24).
- **Older rollback:** in `C:\Temp\qnx\int`, copy `srv_int72b_backup.ps1`, `restore_prod_int72b_backup.ps1`, `Start-Pyre_int72b_backup.ps1` over `srv.ps1` / `restore_prod.ps1` / `Start-Pyre.ps1` (int72 = #19-#21), then `C:\Temp\qnx\stopgrace.ps1 -id <pid>` and `restore_prod.ps1`. Older: `*_int75_backup.ps1` (int75), `*_int71_backup.ps1` (#19+#20), `*_int70_backup.ps1` (#19), `*_int69_backup.ps1` (pre-Nov-24). Exes `pyre_int69..int79.exe` (+ `int79b`) are all kept (int73 = #22 only, int74 = #23 only; neither ever ran in prod).
- **VRAM ladder (#31), tested on int79 with `C:\Temp\qnx\vramhog\vramhog.exe` (v2 reads NVML; v1 used cuMemGetInfo, which under WDDM over-reported free VRAM by ~6.7 GB and overshot):** rung 1 (shrink the expert cache) alone absorbed two 950 MiB-target pushes (cache 2560 -> 1536 -> 512 MiB, free back to ~1.9 GB, no interruption). A stepped push walked rungs 1 -> 2 (`kv_to_host`, layers 0-11) -> 3 (degraded wait), and pyre kept serving for 68 s at 806-877 MiB free, then stopped at rung 4 when free hit 763 MiB (< 768 hard minimum), as designed. **Known gap, fixed in draft PR #33 (not deployed):** `vram_check` aborts a >= 8 MiB allocation in the 768-1024 MiB band once rung 1 has already lowered the cache target (seen: 12 MiB at 957 MiB free -> exit 86). The candidate exe `pyre_int79b.exe` (int79 + #33) is staged, but not deployed until Lyam OKs #33. K/V recovery (rung 2 -> GPU) is not verified yet.
- Watchdog: since int79, `srv.ps1` starts `C:\Temp\qnx\server_watchdog.ps1` from the template `server_int79_watchdog.ps1`, which stops pyre only when free VRAM < **700** MiB (a last-resort backstop below the ladder's 768 MiB hard minimum; the old template `server_int21_watchdog.ps1` stopped at 1024 and would have pre-empted the ladder). `C:\Temp\qnx\bendrun\pyre_guard.ps1` (bendrun harness) runs `restore_prod.ps1` if :8099 is down 3+ min and restarts nano.

## Nov 24 merges (PRs #19-#25, all merged into qwen-int)
| PR | Merge | What |
|---|---|---|
| [#19](https://github.com/Lyamc/pyre/pull/19) perf/prefill-batch | `0b22b5b` | 1024-token prefill chunks by default; attention score scratch sub-batched (`PYRE_QN_ATTN_SC_MB`); prefix snapshots only at the cut + last grid boundary |
| [#20](https://github.com/Lyamc/pyre/pull/20) fix/xclass-adaptive-vram | `b4bb2ae` | XCLASS expert cache actually shrinks/regrows under VRAM pressure (slots re-laid in `xc_resize`) |
| [#21](https://github.com/Lyamc/pyre/pull/21) fix/xcache-hysteresis | `3a6d018` | governor hysteresis: min target 128 MiB (no 0-target stall), grow cooldown with backoff, bounded grow steps |
| [#22](https://github.com/Lyamc/pyre/pull/22) feat/pcache-disk | `881315a` | prefix-cache snapshots persisted to disk, loaded at start; idle + periodic background saves; free-space guard |
| [#23](https://github.com/Lyamc/pyre/pull/23) fix/xcache-governor-tune | `c2eaaf1` | env-tunable cache gates; no-op at min target (int72 logged 1784 fake shrinks); hold shrinks that free no VRAM |
| [#24](https://github.com/Lyamc/pyre/pull/24) perf/expert-residency | `90dd7a5` | opt-in soft min working set / VirtualLock for routed experts |
| [#25](https://github.com/Lyamc/pyre/pull/25) perf/pcache-shared-chunks | `377ae13` | prefix-cache snapshots share 1024-position Arc K/V chunks inside the LCP; RAM cap counts unique chunks; disk format unchanged (int77: snapshot 0.4-1.1 s -> ~0.3-0.4 s, 5 snapshots 2.96 GB with 3.4 GB shared) |

Next ideas: snapshot time is now mostly GPU GDN-state reads + kv_pull + the <1024-token tail (skip re-reading unchanged GDN state); GPU expert streaming for large prefill chunks; dropping cache slots without a full rebuild.
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
- **Nov 24 12:08: int77 in production**: PR #25 shared-chunk prefix snapshots merged (`377ae13`); master fast-forwarded.
- **Nov 24: int76 in production**: PRs #19-#24 merged (table above); master fast-forwarded to qwen-int. C: build dirs cleaned (deps/incremental, 62.9 GB freed, approved by Lyam). bendrun harness (C:\Temp\qnx\bendrun) drives pyre as a coding agent writing a Bend SDF UI library in C:\Build\ui\bend-sdf-ui; see CHECKUP.md there.
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
