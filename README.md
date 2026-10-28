# pyre

A llama.cpp `llama-server`-style inference server written in Rust where **[Burn](https://github.com/tracel-ai/burn)** (one version for everything: [Lyamc/burn](https://github.com/Lyamc/burn) @ `9db7598`, 0.23.0-pre.1) hosts and runs the model. No llama.cpp, candle or ggml does any of the compute: every matmul, attention and norm is a Burn tensor op on CUDA (CubeCL), wgpu (Vulkan/DX12/Metal) or the CPU (Flex / NdArray).

## Build

```powershell
cargo build --release            # cuda + wgpu + cpu(flex) + ndarray
cargo build --release --no-default-features --features wgpu   # smaller build
cargo test --release
```

* Toolchain: `rust-toolchain.toml` pins `stable-x86_64-pc-windows-msvc`, and `.cargo/config.toml` pins the MSVC `link.exe` linker. This overrides the user-level `~/.cargo/config.toml` (which sets `rust-lld` for msvc and MinGW gcc for gnu) and the `-gnu` default host. **Build scripts must be compiled for the msvc host**: with a `-gnu` host, `tracel-llvm-bundler` (used by `burn-cuda`) emits `-lstdc++`, which fails to link.
* CUDA: the `cuda` backend JIT-compiles kernels through CubeCL's bundled LLVM/NVPTX, so you don't need `nvcc` at build time. You need an NVIDIA driver at runtime.

## Run

```powershell
pyre serve -m Qwen/Qwen2.5-0.5B-Instruct -b cuda --port 8080
pyre serve -m qwen -b wgpu --api-key secret          # alias from the pyre store
pyre serve -m C:\models\SmolLM2-360M-Instruct -b cpu  # plain folder
pyre serve -m Qwen/Qwen3-0.6B --prompt "hi"           # one-shot greedy run + tok/s, then exit
```

| flag | meaning |
|---|---|
| `-m, --model` | alias, store name, HF repo id `org/name[@rev]`, or a folder |
| `-b, --backend` | `cuda` \| `wgpu` \| `vulkan` (feature) \| `cpu` (Flex) \| `ndarray` |
| `--dtype` | `auto` (bf16 on CUDA for bf16 checkpoints, f16 on wgpu, f32 on CPU) \| `f32` \| `f16` \| `bf16` |
| `-c, --ctx-size` | context / KV-cache length (default 4096, capped at `max_position_embeddings`) |
| `-n, --n-predict` | default max tokens |
| `--api-key` / `PYRE_API_KEY` | require `Authorization: Bearer <key>` (everything except `/health`) |
| `--host`, `--port`, `-a/--alias` | the same as llama-server |

## Storage

pyre **reuses the standard Hugging Face hub cache in place**, so a model you already downloaded with `huggingface-cli`, Python `transformers`/`huggingface_hub` or another tool **costs no extra disk**, and pyre's downloads are readable by those tools.

* HF cache location: `--hf-cache` > `HF_HUB_CACHE` > `$HF_HOME/hub` > `$XDG_CACHE_HOME/huggingface/hub` > `~/.cache/huggingface/hub` (on Windows `%USERPROFILE%\.cache\huggingface\hub`). The layout is `models--org--name/{blobs,refs,snapshots/<commit>}`. Snapshot entries may be symlinks into `blobs/` (Linux/macOS, or Windows with Developer Mode) or plain files/hardlinks (Windows without symlink rights); pyre handles both.
* Downloads go through `hf-hub` 1.0 into that same cache. On Windows without symlinks, hf-hub 1.0 writes *both* a blob and a snapshot **copy**, doubling disk use. pyre then replaces each snapshot copy with a **hardlink** to its blob, which saves the space and stays compatible.
* `HF_TOKEN` (or the `huggingface-cli login` token file) is used by hf-hub for gated repos and is never printed. `HF_HUB_OFFLINE=1` or `--offline` disables all network access.
* **pyre store** (`--models-dir` > `PYRE_MODELS_DIR` > `models_dir` in the config file `%APPDATA%\pyre\config.toml` / `~/.config/pyre/config.toml` > `%LOCALAPPDATA%\pyre\models` / `~/.local/share/pyre/models`). The layout is `<kind>/<org>/<name>/<rev>/` plus `pyre-model.toml` (source repo/revision, kind, architecture, dtype, files with sizes and sha256, import mode, optional alias).
  * `import --link` (default) hardlinks the files, so they take **no extra space**. If hardlinks fail (different volume) it falls back to symlinks, then to copying, and it tells you which it used.
  * `--copy` copies the files. `--move` takes them out of the cache. `--register` records a local folder in place and copies nothing.
* Search order for `-m`: explicit path → store alias/name → HF cache → download (unless offline). Change it with `--search-order path,store,hf,download` or `search_order = [...]` in the config file.

```text
pyre models list [--json]
pyre models pull <repo>[@rev] [--store hf|pyre]      # hf = cache only (default); pyre = also link into store
pyre models import <repo|path> [--link|--copy|--move|--register] [--name N] [--kind llm|image|stt|tts]
pyre models rm <name> [--from-hf-cache] [-y]         # never deletes HF blobs unless --from-hf-cache (asks first)
pyre models alias <name> <alias>
pyre models info <name>
```

## Speech to text (whisburn)

STT is backed by [whisburn](file:///C:/Build/machine-learning/whisburn), Lyam's Burn speech project. pyre
uses it unmodified through a thin adapter (`src/stt.rs`).

- **Code:** whisburn @ `d27f1fc` is vendored as `crates/pyre-whisper-{core,audio,engine,models}` and
  ported to the Lyamc/burn device-dispatch API (no backend generics). The whisburn repo itself is untouched.
- **One Burn:** `cargo tree` shows a single burn (0.23.0-pre.1, Lyamc/burn#9db7598) and a single cubecl
  (4fdcb6e) for LLM, STT and SD. Whisper weights are re-converted from HF into burnpack (`model.bpk`);
  the published 0.21 `.mpk` bundles are not readable by this Burn and are skipped.
- **Formats:** wav/mp3/flac/aac/m4a, Vorbis ogg, and Opus in webm/ogg (browser mic recordings) via a
  pure-Rust Opus decoder (`opus-rs`) registered into symphonia. Verified: `samples/speech.webm` and
  `speech.ogg` (ffmpeg libopus) transcribe the same as the wav.
- **Backends:** whisburn has wgpu (any GPU, including NVIDIA through Vulkan/DX12) and ndarray (CPU).
  `--backend cuda`/`wgpu` both map to whisburn's wgpu; `cpu`/`flex`/`ndarray` map to its CPU runtime.
- **Models:** any whisburn ASR registry name (`tiny`, `tiny_en`, `base`, `small_en`, `large-v3-turbo`,
  `distil-large-v3`, `parakeet-tdt-0.6b-v3`, `moonshine-tiny`, `qwen3-asr-0.6b`, ...), its HF repo id
  (`openai/whisper-tiny.en`), or a path to an existing whisburn bundle directory.
  Bundles are whisburn's converted format (`model.mpk` + tokenizer + config), not raw HF safetensors.
  They live in `WHISBURN_MODELS_DIR` if set, else `<pyre store>/stt/whisburn/<name>`. A missing bundle
  is fetched by whisburn's own downloader (its published bundle from `lyamc/whisburn`, or a conversion
  from upstream HF). `--offline` disables this. Because `WHISBURN_MODELS_DIR` is process-global, all
  STT models in one pyre process share one bundle dir.

```
pyre serve -m qwen --stt-model tiny_en
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@speech.wav -F model=whisper-1 -F response_format=verbose_json
```

`response_format`: `json` (default), `text`, `srt`, `vtt`, `verbose_json` (segments, duration, language, model; speaker when set), plus whisburn renderings `txt` and `whisburn_json`. Extra fields used by [peal](../peal): `timestamps`, `sentences`, `beam_size`, `max_tokens`, and `stream=true` (SSE `segment` events, then `done`).
`/v1/audio/translations` runs whisburn's translate task. `language` is honoured; `prompt`, `temperature`
and `timestamp_granularities` are accepted and ignored. Input is decoded by whisburn-audio (symphonia:
wav/mp3/flac/ogg-vorbis/m4a-aac). Browser mic recordings (webm/opus) may not decode.

Measured on an RTX 3060 12 GB (512x512, 30 steps default, f32):

| | before (burn 0.20 diffusers-burn) | now (Lyamc/burn, persistent cache + warm-up) |
|---|---|---|
| CUDA first image after startup | ~9 min | **14.1 s** (startup incl. warm-up ~4 min on a cold cache, ~28 s warm) |
| CUDA steady state | ~9.4 s (20 steps) | 13.7 s (30 steps) |
| wgpu first / steady | n/a | **25 s** steady (20 steps; was ~515 s). 3x3 convs run as chunked im2col + matmul (`crates/pyre-sd/src/models/fastconv.rs`), because without tensor-core MMA Burn only has a direct conv kernel, which took 97% of the time. Output matches CUDA (mean abs diff 0.04/255). `PYRE_SD_IM2COL=0/1` overrides; the default is on for non-CUDA. |

The CubeCL kernel/autotune cache is now persisted (`target/environment/default.db`, burn `persistence`
feature). On Windows this needed a local cubecl fix: its Turso cache requested multiprocess WAL, which
Turso's Windows IO backend rejects, so the cache silently fell back to memory every run. Fix is on a local
branch (`C:/Build/ai/cubecl-pyre-fix`, `pyre-windows-wal-fix`) wired in with `[patch]`; not pushed.
Load-time warm-up: `PYRE_SD_WARMUP=0` disables, `PYRE_SD_WARMUP_STEPS` (default 2).

## Text to speech (VITS, hand-written Burn port)

`--tts-model facebook/mms-tts-eng` loads Meta's MMS English VITS (`src/tts.rs`, no burn-onnx). The
vocabulary is characters, so text handling is pure Rust and there is no phonemizer or espeak-ng. The
duration predictor's rational-quadratic spline runs on the host. `POST /v1/audio/speech` takes
`input`, `response_format` (`wav`|`pcm`), `speed`, `noise_scale`, `seed`; output is 16 kHz mono.
Parity with HF transformers (`scripts/hf_vits_ref.py`): same length (49,664 samples), correlation
0.999 with noise off. Cold first call ~63 s (kernel autotune), warm ~0.2 s for 3.1 s of audio. A
Whisper round trip reads it back correctly. The web UI and GUI Audio page call it. Note: the MMS
weights are **CC-BY-NC 4.0** (non-commercial). Samples: `samples/tts-pyre.wav`, `samples/tts-hf-ref.wav`.

## GGUF

A folder containing `model.gguf` (plus config.json/tokenizer.json) loads through a local port of
Lyamc/burn-gguf to the burn fork (`C:/Build/ai/burn-gguf`, branch `pyre-burn-fork`, path dependency,
not pushed; its 168 tests pass). Weights stay **packed on the GPU** (`src/model/kquant.rs`, exactness unit tests):

- Q8_0 -> Burn Q8F block 32 (f16 scales).
- Q5_0 -> Q8F block 32, exact.
- Q6_K -> Q8F block 16 with f32 scales, exact.
- Q4_K -> Q4F block 32 with the exact f32 sub-block scale and q-8, plus a per-block offset matrix C
  (`[out, in/32]`). The linear becomes `x·W'ᵀ + blocksum32(x)·Cᵀ`.

`PYRE_GGUF_KQUANT=0` dequantizes the K/Q5 types at load; `PYRE_GGUF_DEQUANT=1` dequantizes everything.
Burn's quantized matmul dequantizes each weight inside the op (no hand-written fused int4 kernel), so
speed is roughly bf16 speed, not faster. Token embeddings are dequantized (the lookup needs floats).

RTX 3060, CUDA, util cap off (`scripts/gguf_bench.ps1`; extra VRAM after load + 3 requests):

| model | packed | dequantized | Q8_0 | bf16 safetensors |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | +991 MiB, 43.4 tok/s | +1376 MiB, 46.4 tok/s | +1040 MiB, 48.9 tok/s | +1376 MiB, 45.8 tok/s |
| Qwen2.5-1.5B Q4_K_M | +2863 MiB, 28.2 tok/s | +3759 MiB, 28.5 tok/s | +2760 MiB, 38.4 tok/s | — |

All outputs are coherent text. In the 0.5B Q4_K_M file most tensors are Q5_0 (896 is not a multiple of 256).
In the 1.5B file, packing saves ~0.9 GB. It still doesn't beat Q8_0, because Burn's memory pool keeps
dequantized f32 temporaries for the largest weights.

## Fused K-quant matmul (burn fork `../burn-pyre`, branch `pyre-fused-quant`)

pyre's `[patch."https://github.com/Lyamc/burn"]` points every burn crate at a local clone of the
fork (based on 9db7598). The changes live in `burn-cubecl/src/kernel/matmul/fused_quant.rs`:

- **What it is.** `try_fused_qmatmul` runs before the normal path in `kernel::matmul::matmul`.
  It takes a float `x` and the transposed packed weight that pyre keeps resident (`Q4F
  PackedU32`, Q4_K/Q5_0, block 32; `Q8F/Q8S Native`, Q8_0/Q6_K, block 32/16). It reads the
  packed words and the f32 block scales straight from the weight's own buffers, so it never
  creates a dequantized copy. It handles row-pitched buffers and broadcast leading dims.
- **Decode, m ≤ 8 (the default).** A GEMV: one 32-lane plane per output column, 8 columns per
  cube, and `plane_sum` to reduce.
- **Prefill.** A 64×64 shared-memory tiled GEMM exists, but it is opt-in (`BURN_FQ_GEMM=1`)
  because it is 6–11× slower than dequantize + cubek's matmul (q4 8960×1536, m=300: 6.6 ms vs
  0.64 ms). Prefill therefore still dequantizes, and only temporarily, one layer at a time.
- **Switches.** `BURN_FUSED_QMATMUL=0` turns the whole thing off, which also stops
  burn-cubecl-fusion from capturing the matmul. `BURN_FQ_GEMV_MAX=<m>` lowers the GEMV cutoff.
- **Exactness.** `cargo test --release --features cuda,wgpu,stt,image fused_qmatmul -- --ignored
  --nocapture` (`PYRE_TEST_BACKEND=wgpu` for wgpu) compares against bf16 dequant + matmul and
  checks that the fused kernel really was launched. Max relative error is 3.5e-3–6.9e-3 on CUDA
  bf16, the same as the unfused path, and 4e-4–9e-4 on wgpu f16. m=1,4,8 run fused; m=37,300
  run unfused.
- **Portability.** The kernel runs on wgpu (Vulkan, f16) as well as CUDA. On wgpu, Q8 Native
  (i8 storage) does not compile even for plain dequantize, so only Q4 is tested there.

Kernel microbenchmarks (m=1, µs):

| shape | CUDA fused | CUDA unfused | CUDA bf16 | wgpu fused | wgpu unfused | wgpu f16 |
|---|---|---|---|---|---|---|
| q4 896×896 | 9.8 | 40.4 | 13.6 | 8.9 | 104.8 | 21.3 |
| q8 4864×896 | 20.2 | 22.5 | 30.1 | — | — | — |
| q8/b16 1536×8960 | 85.2 | 68.4 | 98.6 | — | — | — |
| q4 8960×1536 | 427 | 492 | 97 | 47 | 283 | 150 |

End to end on CUDA (`scripts/fq_bench.ps1`): decode tok/s over 3 runs of 256 tokens, with
`PYRE_GPU_UTIL_MAX=100 PYRE_LOW_PRIORITY=0`, and VRAM above the idle baseline. The desktop was
in use during these runs (all of them are well below this README's earlier ~41–48 tok/s), so
compare rows with each other, not with the older table.

| model | fused | unfused (`BURN_FUSED_QMATMUL=0`) |
|---|---|---|
| 0.5B Q4_K_M | +1072 MiB, 19.0 tok/s, prefill 425 | +1001 MiB, 17.4 tok/s, prefill 404 |
| 0.5B Q8_0 | +1038 MiB, 21.9 tok/s, prefill 724 | +1061 MiB, 18.5 tok/s, prefill 570 |
| 0.5B bf16 | +1390 MiB, 19.8 tok/s | n/a |
| 1.5B Q4_K_M | **+2026 MiB**, 14.4 tok/s | +2882 MiB, 12.3 tok/s |
| 1.5B Q8_0 | +2632 MiB, 16.7 tok/s | +2778 MiB, 15.9 tok/s |

Results against the goal:

- The fused kernel speeds up decode by 10–18%.
- On 1.5B Q4_K_M it removes the temporary dequant memory: 856 MiB less VRAM than unfused, and
  less than Q8_0.
- **Q4_K_M is still not faster than Q8_0 at decode.** Two reasons:
  - At these sizes decode is bound by launch and host overhead, not by weight bandwidth. bf16
    runs at the same speed.
  - pyre's Q4_K layout adds an offset matmul per projection.
  - The q4 8960×1536 shape (the MLP down-projection) is also slower than bf16 in the GEMV on
    CUDA. That shape is the next thing to tune.
- 1.5B bf16 was not measured, because it is not downloaded locally.
## Continuous batching

The engine (`src/engine.rs`) runs up to `PYRE_PARALLEL` (default 8) sequences per decode step,
with a KV-cache pool reused by longest common prefix. Aggregate tok/s, Qwen2.5-0.5B on CUDA, 128 tokens
(`scripts/bench_concurrency.py`): 1 → 26.9, 4 → 80.6, 8 → 129.0 (bf16); in f32, 27.2 / 70.4 / 127.4,
and batched output is identical to solo output.

## Gemma 2/3 and Qwen3-MoE

Supported in `src/model/llama.rs`: GeGLU, (1+w) norms, embedding scale, pre/post-FFN norms,
softcaps, sliding window, local RoPE, and the MoE router with top-k. Greedy parity vs HF f32
(`scripts/hf_greedy.py`, `samples/parity-gemma-moe.log`) is 20/20 tokens for unsloth/gemma-3-1b-it,
unsloth/gemma-2-2b-it and yujiepan/qwen3-moe-tiny-random. The official google/gemma-* repos are gated
(401 without a token), so the unsloth mirrors were used. No real MoE fits in 12 GB, so only the tiny
random one was tested.

## Desktop GUI (GPUI)

`gui/` is a separate workspace (`pyre-gui`) on [Lyamc/gpui](https://github.com/Lyamc/gpui) @ `08629d4`.
It talks to the server over HTTP only and does not link Burn. Pages: Chat (SSE streaming), Models
(load/unload/pull/import/alias), Images, Audio (transcribe a file, TTS), Status.

```powershell
cargo build --release --manifest-path gui/Cargo.toml
gui\target\x86_64-pc-windows-msvc\release\pyre-gui.exe http://127.0.0.1:8080
```

Screenshots: `samples/gui-{chat,models,images,audio,status}.png`.


Screenshot with TTS: `samples/gui-audio-tts.png` (`PYRE_GUI_TTS="text"` speaks at startup).

## GPU footprint limits (on by default)

pyre leaves room for the desktop and other apps:

- `--vram-reserve 1G` / `PYRE_VRAM_RESERVE`: VRAM kept free. The default is max(1 GiB, 10% of VRAM), which is 1228 MiB
  on a 12 GB card; `0` disables. Free VRAM is read from NVML, for both CUDA and wgpu on NVIDIA.
  - A load whose weight files don't fit above the reserve is refused with a clear error.
  - A load that leaves less than the reserve free is rolled back.
  - LLM batch slots, each of which owns a full-context KV cache, are capped to fit. Example:
    `-c 32768 --vram-reserve 8G` gives 3 slots instead of 8.
  - Image generation at a new, larger size is checked first.
- `--gpu-util-max 90` / `PYRE_GPU_UTIL_MAX`: duty-cycle cap; `100` or `0` disables.
  - After each decode step, prefill chunk and diffusion step, the worker syncs the device and sleeps
    `busy·(100-u)/u`.
  - Prefill runs in 512-token chunks (`PYRE_PREFILL_CHUNK`), so no submission runs long. This is gentler
    on the driver watchdog.
  - The process runs at below-normal priority (`PYRE_LOW_PRIORITY=0` turns that off).
- Current limits and usage are in `/api/status` → `limits`, on the web UI Status page and on the GPUI Status page.

Cost of the defaults (RTX 3060):

| workload | default limits | limits off |
|---|---|---|
| Qwen2.5-0.5B, 1 stream | 37.8 tok/s | 41.1 tok/s |
| Qwen2.5-0.5B, 8 streams (aggregate) | 172.4 tok/s | 190.6 tok/s |
| SD 1.5 512x512, 20 steps, CUDA | 10.5 s | 9.6 s |

Burn's memory pool still caches freed buffers. Once the pool is warm, a long SD session can sit close to
the reserve (seen: 1566 MiB free with a 1228 MiB reserve), and within one op it can't be capped.

## Known gaps

- Chat hang under STT + chat: not reproducible after 40+ mixed concurrent rounds (with image generation, 60 s noise audio, stretched audio; `scripts/stress_mix.py`). Most likely it was the old opus wrong-sample-rate bug, which is fixed.
- K-quants: a fused decode GEMV exists (see above), but Q4_K_M decode is still not faster than Q8_0, and the tiled prefill GEMM is too slow to turn on by default.
- wgpu Stable Diffusion: 25 s vs ~10 s on CUDA (no tensor cores through wgpu).
- LLM parity: Qwen2.5-0.5B greedy, 15 tokens, matches HF f32 on the first 14; token 15 is a near tie
  (`.` vs `.\n`), identical to the pre-port build. Decode (0.5B, RTX 3060): CUDA 43.3 tok/s bf16 /
  38.2 f32 (was 30.5), wgpu 41.9 tok/s (was 38.4).

## LAN hosting

    pyre serve --host 0.0.0.0 --port 8099 [--api-key SECRET] -m <model>

The startup log prints the URLs other machines can use (e.g. `http://192.168.0.30:8099`). The web
UI uses same-origin relative URLs, and the API answers CORS preflights from any origin (like
llama-server). Without `--api-key` anyone on the network can use the server (a warning is logged).

## Apps (companion programs)

    pyre app list | pyre app install peal | pyre app run peal -- transcribe -i a.wav | pyre app remove peal

An app is described by `pyre-app.toml` in its repo (`[app]` name/description/version/min_pyre_api/web_ui,
`[source]` path or git, `[[build]]` steps (`cargo` crate dir + `bins`, `prebuilt`, or `command`; `optional`),
`[[entry]]` cli/gui commands, `[pyre]` url_env / api_key_env / gui_args with `{url}`, `[[models]]` kind + refs).
Built-in entries live in `apps/*.toml` (peal, cabin). Installs go to `PYRE_APPS_DIR` or
`%LOCALAPPDATA%\pyre\apps\<name>` (bin/, manifest copy, `<entry>.cmd` launchers with the server URL);
records in `installed.json`. Same operations over HTTP: `GET /api/apps`, `POST /api/apps/install|remove|run`
(`{"name": ...}`), and the web UI's Apps page.

## Qwen3.8-Flash-Next (experimental)

    pyre qwen-next <hf-snapshot-dir> --prompt "..." -n 32 [-b cuda] [--experts cpu|same] [--dump logits.json]

Burn port of the `qwen4_exp` text model (`src/model/qwen4exp.rs`): hyper-connections, Gated DeltaNet,
gated attention with partial RoPE, 512-expert MoE + shared expert, hashed n-gram PLE. Routed experts and
the n-gram table sit on the CPU device (Strata's split). Verified against transformers on a tiny random
config (`scripts/qwen_next_ref.py`). Not yet: QSA indexer (exact only up to 2048 tokens), MTP drafting,
GGUF/quantized experts, server/engine integration.

### Real weights: GGUF, dense on the GPU and experts on the CPU (IQ2_XS)

    pyre serve -m ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF -b cuda   # OpenAI API + web chat
    pyre qwen-next <dir-with-gguf> --prompt "..." -n 32 [--chat] [-b cuda|cpu] [--peek TENSOR:ROW]

`src/model/qwen4exp_gguf.rs` runs the `qwen4exp` GGUF arch straight from mmapped, still-quantized shards
(`src/model/gguf_q.rs`): IQ1_M, IQ2_XXS, IQ2_S, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q6_K, Q8_0, BF16, F16, F32 are
checked bit-for-bit against gguf-py; Q3_K/Q4_K/Q5_K/IQ2_XS decoders exist but no tensor in this file uses them.
The tokenizer and chat template come from the `Qwen/Qwen3.8-Flash-Next` base repo in the HF cache.
`<think>...</think>` goes to `reasoning_content` (`PYRE_REASONING=none` turns that off).

With `-b cuda` (`src/model/qn_gpu.rs`), every non-routed weight is requantized to Q8_0 and uploaded as Burn
packed Q8 tensors within pyre's VRAM reserve (NVML free - reserve - 1.5 GiB). That covers the
hyper-connection mixers, the attention/DeltaNet in/out projections, routers, the shared expert, the PLE
projections and the head. The residual stays on the device, with two reads per layer (mixer projections;
then the FFN input plus router logits). The DeltaNet/attention recurrences and the 512 routed experts run on
the CPU. The shared expert is queued on the GPU while the CPU pool runs the routed experts. Chunks larger
than `PYRE_QN_GPU_TMAX` tokens (default 8) take the CPU SGEMM path, which is faster for prefill here.

The CPU expert kernels (`src/model/qdot.rs`) quantize activations to Q8 blocks of 32 and run integer AVX2
dots against IQ2_S / IQ2_XXS / IQ1_M / Q2_0 rows. Q2_0 and IQ1_M offsets are applied as corrections from
per-8 activation sums. A row shared by more than 2 tokens is decoded once. When a prefill gives an expert
8 or more tokens, it uses a dequantized SGEMM.

Env: `PYRE_QN_QUANT` (default IQ2_XS), `PYRE_QN_THREADS` (16), `PYRE_PREFILL_CHUNK` (256),
`PYRE_QN_GPU_DTYPE` (f32), `PYRE_QN_GPU_TMAX` (8), `PYRE_QN_GPU_HC` (1; 0 = hybrid with the HC mixers on the
CPU), `PYRE_QN_HC_Q8` (1), `PYRE_QN_QDOT` (1), `PYRE_QN_BLK`/`PYRE_QN_DBLK` (64/128),
`PYRE_QN_GEMM_MIN` (8), `PYRE_QN_PREFAULT` (1), `PYRE_QN_GPUBENCH`, `PYRE_QN_TOKENIZER`,
`PYRE_QN_PROFILE`, `PYRE_QN_DEBUG`, `PYRE_QN_BENCH`. Context is capped at 2048 tokens.

Measured on a 16-core desktop with a 12 GB RTX card (Windows/WDDM). Prompt B is a 64-token chat prompt:

| | decode tok/s | prefill tok/s (64 tok) | VRAM |
|---|---|---|---|
| pyre CPU, start of this work | 2.8 | 13 (noisy) / 6 | 0 |
| pyre CPU now | 3.4 | 9.8 | 0 |
| pyre `-b cuda`, Burn ops (`PYRE_QN_CUBE=0`) | 4.3-4.9 | 9.1-9.9 | ~5 GB |
| pyre `-b cuda` now (own CubeCL kernels + GPU expert cache) | 10.8-12.0 steady (~10 overall) | 8-10 | ~8.9 GB peak |
| Strata 0.1.48 (same GGUF) | 15.4 (B), 25.9 (short prompt) | 80 | 10.9 GB |

Greedy output matches between the pyre CPU and GPU paths. It also matches Strata token for token on a
32-token run. On a 42-token run, the outputs diverge at one near-tie logit after token 32.

Ideas taken from Strata (github.com/Niko1221/Strata, MIT). These are pyre's own implementations and
contain no Strata code:
- Dense/non-routed weights on the GPU, routed experts on the CPU (decode 3.1 -> 4.5 tok/s).
- llama.cpp-style integer i-quant dot products with Q8-quantized activations on the CPU pool
  (routed experts 75 -> 44 ms per token).
- One pool worker per core with coarse row blocks per expert phase.
- Experts kept resident (prefault of the mapped GGUF, no copy). This removed about 11k page faults per
  token but did not change the speed.
- The shared expert overlapped with the CPU expert pass.
- Large prefill chunks with batched dense work and per-head DeltaNet threading (CPU prefill on a
  720-token prompt went from 8.3 to 16.7 tok/s).

Decode path with `-b cuda` (`src/model/qn_cube.rs`, pyre's own CubeCL kernels on one stream): Q8 GEMV,
fused hyper-connection RMSNorm / gate / combine, Gated DeltaNet conv + delta rule with the recurrent state
kept on the GPU (copied back when a CPU prefill continues the same session), the shared expert, and a
routed-expert cache. The cache holds the most-routed experts (counted from the prompt, refreshed every 16
tokens) in their native IQ2_S / IQ2_XXS / IQ1_M / Q2_0 bytes, repacked word-aligned. It is sized from the
VRAM left above pyre's reserve minus 2 GiB of headroom (`PYRE_QN_XCACHE_GB` caps it; 0 disables it).
Cached experts run on the GPU while a spinning, P-core-pinned worker pool (`src/model/spin_pool.rs`) runs
the rest. This leaves two readbacks per attention layer, one per DeltaNet layer, and a few for PLE and the
head. More env: `PYRE_QN_CUBE`, `PYRE_QN_GDN_GPU`, `PYRE_QN_XREFILL`, `PYRE_QN_XSWAP`, `PYRE_QN_XDEBUG`,
`PYRE_QN_SPIN`, `PYRE_QN_SPIN_US` (100), `PYRE_QN_SPIN_THREADS`, `PYRE_QN_PIN` (cpu list or `none`),
`PYRE_QN_SBLK`/`PYRE_QN_SDBLK`.

More ideas taken from Strata, with their effect on prompt B decode:
- The fused per-layer GPU step instead of many framework ops (Burn path 4.5 -> 6.1 tok/s).
- DeltaNet on the GPU (100 -> 64 readbacks per token, ~8.1 tok/s).
- A persistent pool with one worker per physical P-core, pinned (routed 47 -> ~22 ms/token).
- The GPU expert cache ranked by usage, overlapping the CPU pool (45-55% hits).
Together: about 10-12 tok/s.

Not adopted yet:
- CUDA graphs. cubecl at this revision exposes `graph_prepare`/`start_capture`/`stop_capture`, so no cubecl
  change is needed. But GPU utilisation is only ~35% and host enqueue is ~2 ms/token, so the remaining cost
  is readback latency more than launch overhead.
- Native IQ4_XS dense weights. Q8 doubles the bytes, about 5 ms/token at the 3060's bandwidth.
- MTP/prompt-lookup drafting (this GGUF has no MTP tensors).
- The QSA indexer beyond 2048 tokens.
- GPU expert streaming for prefill.
Strata's refusal-direction ("speed projection") vector is
deliberately not ported.

Attribution: the graph follows llama.cpp `src/models/qwen4exp.cpp` and `conversion/qwen4exp.py`. The
i-quant grids in `src/model/iq_tables.rs` and the vec_dot activation-quantization scheme come from ggml
(MIT, (c) the ggml authors).

### Qwen-Next decode, Oct 27 evening round
- CPU experts: Q8_K-style activations (one scale per 256) with int32 super-block accumulation for IQ2_S/IQ2_XXS/IQ1_M (the q8_K vec_dot idea from ggml, MIT); software prefetch of expert rows.
- GPU: full attention for decode (RMS norm + RoPE + KV append, softmax*V with gate), KV synced lazily with the host; cached experts of a layer in one batched launch from 1 GiB arenas; 8-values-per-lane expert kernels.
- Native IQ4_XS dense GEMV exists but is opt-in (PYRE_QN_NATIVE=1); it is not faster than Q8_0 on an RTX 3060 yet.
- Int8 HC mix is opt-in (PYRE_QN_HC_Q8=1): it changed greedy output on a short prompt.
- Steady decode about 12-14 tok/s (prompt A) and 12-13 tok/s (prompt B), vs Strata 25.9 / 15.4 tok/s.
