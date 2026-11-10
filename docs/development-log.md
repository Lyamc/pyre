# Development log

Historical notes that used to live in the repository README. How to build and start the server is in the README at the repo root. Some older lines in this file have broken punctuation from an encoding mix-up; the behavior they describe is not a substitute for the current README or the source.

# pyre

A llama.cpp `llama-server`-style inference server written in Rust where **[Burn](https://github.com/tracel-ai/burn)** (one version for everything: [Lyamc/burn](https://github.com/Lyamc/burn) @ `9db7598`, 0.23.0-pre.1) hosts and runs the model. No llama.cpp, candle or ggml does any of the compute: every matmul, attention and norm is a Burn tensor op on CUDA (CubeCL), wgpu (Vulkan/DX12/Metal) or the CPU (Flex / NdArray).

## Build

Clone with vendored dependency submodules:

```powershell
git clone --recurse-submodules https://github.com/Lyamc/pyre.git
cd pyre
git checkout qwen-int
# already cloned without submodules?
git submodule update --init --recursive
```

| submodule | GitHub | branch / pinned commit |
|---|---|---|
| `vendor/burn-pyre` | [Lyamc/burn](https://github.com/Lyamc/burn) | `pyre-fused-quant` |
| `vendor/burn-gguf` | [Lyamc/burn-gguf](https://github.com/Lyamc/burn-gguf) | `pyre-burn-fork` |
| `vendor/cubecl-pipe` | [Lyamc/cubecl-pipe](https://github.com/Lyamc/cubecl-pipe) | `pyre-hostmem` |

`burn-gguf` is a single crate and is used as a `path = "vendor/burn-gguf"` dependency. Burn and CubeCL are full Cargo workspaces; Cargo cannot path-depend on nested workspaces that live *inside* this repo, so those two are pinned in `Cargo.toml` with **git revs that match the submodule commits** (`[patch]` + direct deps). After `git submodule update`, the trees under `vendor/` are the same commits Cargo builds.

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
* Search order for `-m`: explicit path ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ store alias/name ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ HF cache ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ download (unless offline). Change it with `--search-order path,store,hf,download` or `search_order = [...]` in the config file.

```text
pyre models list [--json]
pyre models pull <repo>[@rev] [--store hf|pyre]      # hf = cache only (default); pyre = also link into store
pyre models import <repo|path> [--link|--copy|--move|--register] [--name N] [--kind llm|image|stt|tts]
pyre models rm <name> [--from-hf-cache] [-y]         # never deletes HF blobs unless --from-hf-cache (asks first)
pyre models alias <name> <alias>
pyre models info <name>
```

## Image tagging (EVA02 Danbooru tagger, e.g. Kagami-24k)

Model kind `tagger` (task `image-classification`). [Redstonexs/kagami-24k](https://huggingface.co/Redstonexs/kagami-24k)
is an EVA02-Large/14 @ 448 Danbooru tagger with 24,000 general tags, written natively in Burn
(`src/tagger.rs`): no onnxruntime, no torch. Weights load in place from the HF cache: the repo's
`weights.pt` is read by the Burn fork's own `pytorch-reader` (no conversion; a timm `model.safetensors`
works too) and `selected_tags.csv` provides the vocabulary.

```
pyre models pull Redstonexs/kagami-24k            # or: hf download Redstonexs/kagami-24k
pyre tag page.jpg some\folder --jsonl out.jsonl   # folders are walked recursively; --batch 4 default
pyre serve --port 8099 --tagger-model Redstonexs/kagami-24k     # or --load tagger=Redstonexs/kagami-24k
```

`POST /v1/images/tags` with JSON `{"image": "<base64|data URL>", "images": [...], "threshold": 0.37,
"top_k": 50, "watch": ["tag", ...], "rating_threshold": 0.5, "include_probs": false}` or multipart
(`image` parts + the same fields). Each result has `tags` grouped by Danbooru category (all `general`
for Kagami), `watch` (scores of censorship / content / manga-format tags, always reported), and
`rating`. Kagami has **no rating tags** (and no character tags), so `rating_source: "derived"`: the
explicit / questionable / sensitive scores are the max over evidence tags (e.g. `sex`, `penis`,
`censored`, `mosaic_censoring` -> explicit; `nude`, `nipples`, `convenient_censoring` -> questionable;
`cleavage`, `swimsuit`, `underwear` -> sensitive) with the evidence listed. Taggers that ship rating
tags (WD v3 style, category 9) report them directly (`rating_source: "model"`).

Model-card details implemented: alpha over white, white square pad (centred), bicubic resize to 448
(Pillow's resampler reproduced exactly, `pil_bicubic`), [-1, 1] (mean = std = 0.5), **RGB -> BGR**,
NCHW; default threshold 0.37 (macro-F1; 0.54 for micro-F1). Note: Kagami's shipped ONNX uses the
raw 32x32 RoPE grid, *not* timm's default 16x16 rescale for `eva02_large_patch14_448`; with the
rescale probabilities differ by up to 0.96, so pyre follows the ONNX.

Reference check (`C:\Build\ai\kagami-ref`: timm + torch CPU with `ref_feat_shape=(32,32)`, which
matches the ONNX to 0.004): same preprocessed input -> max |dp| 0.0022 (TF32 matmuls); end to end
from files (pyre's own decode + resize) -> max |dp| 0.005 over 9 manga pages (PNG ~0.001, JPEG
decoder differences up to 0.005); top-10 tags identical in order on all 9; 1 tag flips across 0.37.

RTX 3060, f32: ~215 ms/image at batch 1, ~195 ms/image at batch 8 (~5 img/s); VRAM ~1.7 GB (b1),
~2.3 GB (b4), ~2.9 GB (b8) including the CUDA context. `PYRE_TAGGER_PRECISION=f16` exists but is
**not accurate** (EVA02 overflows in half precision here: up to 0.97 prob error); keep f32.
Env: `PYRE_TAGGER_BATCH` (default 4), `PYRE_TAGGER_WARMUP=0`, `PYRE_TAGGER_RGB=1` (feed RGB).

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

The CubeCL autotune results **and compiled kernels** are persisted in `%LOCALAPPDATA%\cubecl\default.db`
(burn `persistence` feature). cubecl only caches autotune by default and its default location depended on
the working directory (repo `target/environment` vs the user cache dir), so pyre now turns on the
compiled-kernel cache and pins the user cache dir at startup (`src/main.rs`, `kernel_cache_config`;
`PYRE_KERNEL_CACHE=0` disables, a `cubecl.toml` `[environment] path` still wins). Measured on the RTX 3060,
tiny.en, 7.5 s clip, CUDA: first transcription after a restart 45 s -> **0.7 s**; TTS first call 61 s ->
**0.1 s**. The cache is keyed on the binary's build id, so the first start after a rebuild recompiles once.
STT also warms up in the background at load (1 s of silence; `PYRE_STT_WARMUP=0` disables). wgpu
(WGSL) kernels are compiled by the driver and are not cacheable: first wgpu STT ~10 s on a warm autotune
cache (the `vulkan` backend uses SPIR-V, which cubecl can cache).
On Windows the persisted cache also this needed a local cubecl fix: its Turso cache requested multiprocess WAL, which
Turso's Windows IO backend rejects, so the cache silently fell back to memory every run. Fix is on a local
branch (`C:/Build/ai/cubecl-pyre-fix`, `pyre-windows-wal-fix`) wired in with `[patch]`; not pushed.
Load-time warm-up: `PYRE_SD_WARMUP=0` disables, `PYRE_SD_WARMUP_STEPS` (default 2).

## Text to speech (VITS, hand-written Burn port)

`--tts-model facebook/mms-tts-eng` loads Meta's MMS English VITS (`src/tts.rs`, no burn-onnx). The
vocabulary is characters, so text handling is pure Rust and there is no phonemizer or espeak-ng. The
duration predictor's rational-quadratic spline runs on the host. `POST /v1/audio/speech` takes
`input`, `response_format` (`wav`|`pcm`), `speed`, `noise_scale`, `seed`; output is 16 kHz mono.
Parity with HF transformers (`scripts/hf_vits_ref.py`): same length (49,664 samples), correlation
0.999 with noise off. Cold first call ~63 s (kernel compile + autotune; ~0.1 s once the kernel cache is warm), warm ~0.2 s for 3.1 s of audio. A
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
  (`[out, in/32]`). The linear becomes `xÃÆÃâÃÂ¢Ã¢âÂ¬ÃÂ¡ÃÆÃ¢â¬Å¡ÃâÃÂ·W'ÃÆÃâÃâÃÂ¡ÃÆÃ¢â¬Å¡ÃâÃÂµÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ + blocksum32(x)ÃÆÃâÃÂ¢Ã¢âÂ¬ÃÂ¡ÃÆÃ¢â¬Å¡ÃâÃÂ·CÃÆÃâÃâÃÂ¡ÃÆÃ¢â¬Å¡ÃâÃÂµÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬`.

`PYRE_GGUF_KQUANT=0` dequantizes the K/Q5 types at load; `PYRE_GGUF_DEQUANT=1` dequantizes everything.
Burn's quantized matmul dequantizes each weight inside the op (no hand-written fused int4 kernel), so
speed is roughly bf16 speed, not faster. Token embeddings are dequantized (the lookup needs floats).

RTX 3060, CUDA, util cap off (`scripts/gguf_bench.ps1`; extra VRAM after load + 3 requests):

| model | packed | dequantized | Q8_0 | bf16 safetensors |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | +991 MiB, 43.4 tok/s | +1376 MiB, 46.4 tok/s | +1040 MiB, 48.9 tok/s | +1376 MiB, 45.8 tok/s |
| Qwen2.5-1.5B Q4_K_M | +2863 MiB, 28.2 tok/s | +3759 MiB, 28.5 tok/s | +2760 MiB, 38.4 tok/s | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ |

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
- **Decode, m ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ°ÃÆÃ¢â¬Å¡ÃâÃÂ¤ 8 (the default).** A GEMV: one 32-lane plane per output column, 8 columns per
  cube, and `plane_sum` to reduce.
- **Prefill.** A 64ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ64 shared-memory tiled GEMM exists, but it is opt-in (`BURN_FQ_GEMM=1`)
  because it is 6ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬Ãâ¦Ã¢â¬Å11ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ slower than dequantize + cubek's matmul (q4 8960ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ1536, m=300: 6.6 ms vs
  0.64 ms). Prefill therefore still dequantizes, and only temporarily, one layer at a time.
- **Switches.** `BURN_FUSED_QMATMUL=0` turns the whole thing off, which also stops
  burn-cubecl-fusion from capturing the matmul. `BURN_FQ_GEMV_MAX=<m>` lowers the GEMV cutoff.
- **Exactness.** `cargo test --release --features cuda,wgpu,stt,image fused_qmatmul -- --ignored
  --nocapture` (`PYRE_TEST_BACKEND=wgpu` for wgpu) compares against bf16 dequant + matmul and
  checks that the fused kernel really was launched. Max relative error is 3.5e-3ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬Ãâ¦Ã¢â¬Å6.9e-3 on CUDA
  bf16, the same as the unfused path, and 4e-4ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬Ãâ¦Ã¢â¬Å9e-4 on wgpu f16. m=1,4,8 run fused; m=37,300
  run unfused.
- **Portability.** The kernel runs on wgpu (Vulkan, f16) as well as CUDA. On wgpu, Q8 Native
  (i8 storage) does not compile even for plain dequantize, so only Q4 is tested there.

Kernel microbenchmarks (m=1, ÃÆÃâÃÂ¢Ã¢âÂ¬ÃÂ¡ÃÆÃ¢â¬Å¡ÃâÃÂµs):

| shape | CUDA fused | CUDA unfused | CUDA bf16 | wgpu fused | wgpu unfused | wgpu f16 |
|---|---|---|---|---|---|---|
| q4 896ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ896 | 9.8 | 40.4 | 13.6 | 8.9 | 104.8 | 21.3 |
| q8 4864ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ896 | 20.2 | 22.5 | 30.1 | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ |
| q8/b16 1536ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ8960 | 85.2 | 68.4 | 98.6 | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ | ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ |
| q4 8960ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ1536 | 427 | 492 | 97 | 47 | 283 | 150 |

End to end on CUDA (`scripts/fq_bench.ps1`): decode tok/s over 3 runs of 256 tokens, with
`PYRE_GPU_UTIL_MAX=100 PYRE_LOW_PRIORITY=0`, and VRAM above the idle baseline. The desktop was
in use during these runs (all of them are well below this README's earlier ~41ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬Ãâ¦Ã¢â¬Å48 tok/s), so
compare rows with each other, not with the older table.

| model | fused | unfused (`BURN_FUSED_QMATMUL=0`) |
|---|---|---|
| 0.5B Q4_K_M | +1072 MiB, 19.0 tok/s, prefill 425 | +1001 MiB, 17.4 tok/s, prefill 404 |
| 0.5B Q8_0 | +1038 MiB, 21.9 tok/s, prefill 724 | +1061 MiB, 18.5 tok/s, prefill 570 |
| 0.5B bf16 | +1390 MiB, 19.8 tok/s | n/a |
| 1.5B Q4_K_M | **+2026 MiB**, 14.4 tok/s | +2882 MiB, 12.3 tok/s |
| 1.5B Q8_0 | +2632 MiB, 16.7 tok/s | +2778 MiB, 15.9 tok/s |

Results against the goal:

- The fused kernel speeds up decode by 10ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢âÂ¬ÃÂ¡ÃâÃÂ¬ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬Ãâ¦Ã¢â¬Å18%.
- On 1.5B Q4_K_M it removes the temporary dequant memory: 856 MiB less VRAM than unfused, and
  less than Q8_0.
- **Q4_K_M is still not faster than Q8_0 at decode.** Two reasons:
  - At these sizes decode is bound by launch and host overhead, not by weight bandwidth. bf16
    runs at the same speed.
  - pyre's Q4_K layout adds an offset matmul per projection.
  - The q4 8960ÃÆÃâÃâ Ã¢â¬â¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢âÂ¬ÃÂ1536 shape (the MLP down-projection) is also slower than bf16 in the GEMV on
    CUDA. That shape is the next thing to tune.
- 1.5B bf16 was not measured, because it is not downloaded locally.
## Decode launch overhead

Dense float checkpoints (safetensors bf16/f16/f32) concatenate q/k/v and gate/up at load, so each layer runs
two matmuls fewer per token (`PYRE_FUSE_PROJ=0` disables; packed Q8 / Q4_K-offset GGUF weights stay separate).
RoPE's rotate_half is one flip with the sign folded into the sin table. Qwen2.5-0.5B bf16 on CUDA, same
greedy output: 16.8 -> 21.0 tok/s (noisy: desktop in use, default GPU limits on).

The fused weights are joined on the host and uploaded once (a device-side `cat` left ~220 MiB of freed
inputs in the allocator pool). GGUF: same-type packed weights (Q8_0, Q4_K, Q5_0, Q6_K) are stacked
before repacking, so q/k/v and gate/up become one fused GEMV plus, for Q4_K, one block-offset matmul
(mixed types such as Q4_K q/k + Q6_K v stay separate). Qwen2.5-1.5B, CUDA, same output: Q4_K_M
15.9 -> ~18 tok/s and -220 MiB; Q8_0 20.3 -> ~25 tok/s. The KV cache is updated in place (no measurable
change at one sequence). Folding the Q4_K offset into the fused kernel itself would need Burn's quant
scheme to carry a per-block offset; not done.

## Continuous batching

The engine (`src/engine.rs`) runs up to `PYRE_PARALLEL` (default 8) sequences per decode step,
with a KV-cache pool reused by longest common prefix. Aggregate tok/s, Qwen2.5-0.5B on CUDA, 128 tokens
(`scripts/bench_concurrency.py`): 1 ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ 26.9, 4 ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ 80.6, 8 ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ 129.0 (bf16); in f32, 27.2 / 70.4 / 127.4,
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

### Dashboard (GPUI) and `/api/metrics`

The GUI opens on a live **Dashboard** page (`PYRE_GUI_PAGE=dashboard`, the default; point it anywhere:
`pyre-gui http://192.168.0.30:8099`). It polls `GET /api/metrics` once a second and shows: server and LAN
URL, loaded models with kind and context size, decode/prefill tok/s (live for the running request, else
the last one), active/queued/finished requests and token totals, context used / max, VRAM used/free with
pyre's reserve marked, GPU util against the duty-cycle cap, RAM (system and pyre RSS), CPU, CPU-expert vs
GPU time, expert-cache hit rate, sparklines (decode tok/s, GPU util, VRAM) and a recent-request table
(prompt/cached/generated tokens, prefill and decode tok/s, outcome ok/error/cancelled).
`/api/status` also carries the same request metrics under `metrics` (the web Status tab shows them).

Model code can feed the backend-specific panels through `src/metrics.rs`:
`metrics::add_cpu_expert_ns(ns)`, `metrics::add_gpu_ns(ns)`, `metrics::expert_cache(hits, misses)`.
Until a backend calls them they read "n/a"/"no data" (the Qwen-Next path should call them; see `qwen-perf`).

Layout idea (gauges + sparklines + request table at a glance) inspired by the TUI of
[Strata](https://github.com/Niko1221/Strata); no code is shared.
Screenshots: `samples/dashboard-live.png`, `samples/dashboard-requests.png`.

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
    `busyÃÆÃâÃÂ¢Ã¢âÂ¬ÃÂ¡ÃÆÃ¢â¬Å¡ÃâÃÂ·(100-u)/u`.
  - Prefill runs in 512-token chunks (`PYRE_PREFILL_CHUNK`), so no submission runs long. This is gentler
    on the driver watchdog.
  - The process runs at below-normal priority (`PYRE_LOW_PRIORITY=0` turns that off).
- Current limits and usage are in `/api/status` ÃÆÃâÃâÃÂ¢ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃâÃÂ ÃÆÃÂ¢ÃÂ¢Ã¢â¬Å¡ÃÂ¬ÃÂ¢Ã¢â¬Å¾ÃÂ¢ `limits`, on the web UI Status page and on the GPUI Status page.

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

## Streaming resume

A chat completion that dies with the process (crash, VRAM watchdog, `srv.ps1` restart) is
written to `PYRE_RESUME_DIR` (default `{temp}/pyre/generations`) as it runs. The next process
marks it `aborted`. Clients learn that from `finish_reason: "error"` plus `pyre.status`, or
from the socket going quiet: the server sends an SSE comment every `PYRE_SSE_PING_SECS`
(default 5). Resume with the same id:

```json
POST /v1/chat/completions
{ "resume": "chatcmpl-...", "stream": true }
```

`GET /v1/generations/{id}` returns the partial text. Details, timeouts, and the pi setup
(proxy on the client machine, no pi source change) are in
[docs/streaming-resume.md](docs/streaming-resume.md). The proxy is
`tools/pyre_resume_client.py`. Journal and resume behaviour is tested without a GPU:

```bash
cargo test -p pyre-resume
```

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

### Qwen-Next: speculative decoding, 128k context, route prior (Oct 29)
Ideas borrowed from Strata (github.com/Niko1221/Strata, MIT); pyre's own implementations, no Strata code.
- `--spec N` / config `spec` / per-request `spec` or `speculative: {draft_len}`: speculative decoding
  (`src/spec.rs`). This GGUF has no MTP/NextN tensors, so the drafter is prompt lookup (n-gram); the MTP
  hook (`mtp_draft`) is there for GGUFs that carry an MTP layer. Drafts are verified in one multi-token
  pass that shares the CPU expert reads; the state is rolled back past the first rejection. Draft length
  adapts online to measured verify cost and acceptance. Rejected drafts are uncounted from the expert-cache
  statistics and windows never straddle a cache refresh, so the cache schedule matches plain decode.
  Off by default: on the RTX 3060 it is not faster (verify cost grows almost linearly with the window
  because routed experts run on the CPU), and on a 400-token code-edit run greedy output still diverged
  from plain decode at token 294 (64-token story/code runs are identical).
- Context window: `PYRE_QN_CTX` or the chat setting (default 131072). Presets run 8k–512k, and a raw count is accepted up to 1048576. Past the GGUF's trained length (usually 262144) uses YaRN. The dense attention range
  (indexer top-k, 2048) stays exact. Older KV is stored at 4-bit by default (`PYRE_QN_KV_BITS=8` or the KV control selects 8-bit; `PYRE_QN_KV8=0` keeps f32). Both packed widths are lossy. A window that would cross the VRAM floor is refused, and the response offers 4-bit or a shorter preset.
- Route prior (`--route-prior X` / `PYRE_ROUTE_PRIOR` / config `route_prior`): residency-biased routing
  toward GPU-cached experts. It changes outputs, so it is off (0) by default.
- `/api/status` reports `speculative` (draft len, rounds, drafted, accepted, acceptance), `route_prior`
  and `context` (used / max); the dashboard metrics hooks (`crate::metrics`) are fed from the Qwen-Next
  expert path (CPU expert ns, GPU read ns, expert-cache hits/misses).

### Qwen-Next decode profile and fixes (Oct 29, later)
Per-token profile (RTX 3060, `PYRE_QN_PROFILE=1` + nsys): ~1300 kernel launches and 54 readbacks per token;
GPU kernels ~27 ms (Q8 dense GEMVs ~12 ms at ~325 GB/s, i.e. bandwidth-bound; GPU cached experts ~6.6 ms;
DeltaNet 2 ms), ~12 ms of small launch gaps, ~21 ms of large gaps where the GPU waits for the CPU
(routing + CPU experts). The GPU is busy ~45% of the time.
- The GPU routed-expert cache was opt-in in phase 3 (it is default on again since phase 4, see below). Its batched expert kernels reach
  only ~15-70 GB/s and sit on the critical path, so decode is faster without it (interleaved A/B, steady
  tok/s: story 15.3-15.5 vs 14.1, code 14.2-15.3 vs 13.4-13.7) and VRAM use drops from ~8.1 to ~5.4 GB.
- Expert-cache eviction is deterministic (HashMap order used to decide slots and therefore the GPU
  summation order, so outputs could vary run to run). With that, speculative output is greedy-identical
  to plain decode on a 400-token code-edit run, with and without the cache.
- Verify cost: a 4-token verify pass costs ~3.2 single steps, because consecutive tokens pick mostly
  different experts, so the expert union saves little CPU work (CPU experts ~108 ms for 4 tokens vs ~27 ms
  for one). Spec decoding stays off by default.

### Qwen-Next GPU experts, phase 3 (Oct 29)
- Batched GPU expert kernels rewritten around int8 `dp4a`: activations are packed to the CPU's Q8 grid
  (per-32 scales), the IQ2_S / IQ2_XXS grid sits in shared memory, sign masks and the IQ2_XXS sign parity
  are computed instead of looked up, Q2_0 down weights expand through a 256-entry byte table, and the
  down pass spreads all of a row's (expert, group) pairs across the lanes. `PYRE_QN_XBENCH=1 pyre` times
  them: 10 experts of a layer 225 -> ~122 us (58 -> ~107 GB/s). IQ1_M layers keep the f32 kernels.
  `PYRE_QN_XQ=0` restores the f32 kernels.
- With them the opt-in expert cache (`PYRE_QN_XCACHE_GB=3`) is now break-even with the CPU-only default
  (~14.5-15 tok/s steady either way); the cache path still pays ~5 ms/token of host enqueue work.
- `PYRE_QN_KICK=1` submits queued GPU work before the CPU expert pass (stream query; local cubecl branch
  `pyre-wddm-kick`). No measurable effect on HYTE, so it is off.
- Same-machine Strata 0.1.48 on this IQ2_XS GGUF (Oct 27, with its --spec 2): 25.9 tok/s (short prompt),
  15.4 (chat prompt B). Strata's published 46-62 tok/s are v0.2.7 on IQ3_XXS with MTP drafting.

### Phase 4: low-overhead GPU expert path

- One expert arena (split into even arenas of at most 2.5 GiB, since one buffer over ~3.1 GB fails to allocate),
  4 launches per MoE layer, reused scratch buffers, router output read in place (`PYRE_QN_XFAST`, default on).
- The routed-expert cache is **default on** with a 2.5 GiB budget (`PYRE_QN_XCACHE_GB`, 0 = off).
  RTX 3060, quiet GPU, 96 tokens, greedy output unchanged: cache off ~14.4 story / ~14.7 code tok/s;
  2.5 GiB ~19.5 / ~17.6 tok/s with >=2.6 GB VRAM still free. 3 GiB is slower (~17.3 / ~16.8).- Multi-token verify uses the same fast expert path; up to two arenas run in one launch (arena bit in the offsets).
- `PYRE_QN_NATIVE=2`: dp4a IQ4_XS dense GEMV (int8 activations, ~2.3 G of the dense weights stay 4-bit).
  Greedy output unchanged, but no decode gain on the RTX 3060 (~-0.2..+0.3 tok/s), so it stays opt-in.
- `PYRE_QN_GPUHEAD=1`: LM head on the GPU, ~+0.3 tok/s; opt-in.
- Larger caches (4-5 GiB) give at most +2% even over 400 tokens: the cache fills 64 experts per refill
  and the hit rate stays ~59%.
- Profile per token at ~19 tok/s: ~24 ms waiting on GPU readbacks (54 syncs, ~1,300 launches, while dense
  bytes would take only ~8-10 ms), ~16 ms CPU routed experts, ~5 ms enqueue. The next wins are fewer
  launches and syncs (kernel fusion, GPU-side routing), not dense bandwidth.- The expert cache now fills completely right after prefill (it used to start from 64 experts and add 64 every
  16 tokens). A 2.5 GiB fill takes ~2 s once; hit rate ~60% on a story prompt, ~38% on a code prompt
  (~51% at 5 GiB), with little tok/s change.
- Nsight Systems (CUDA trace only, 40 tokens, RTX 3060, Windows WDDM) per decode token: ~900 kernel launches
  (~1,200 with `PYRE_QN_NATIVE=2`) at ~7.4 us of host time each (6.6-8.9 ms/token), ~185 host-to-device copies
  at 17-25 us each (3-4.6 ms), ~100 event syncs; GPU kernel time ~23 ms (dense Q8 GEMV ~12 ms of it,
  ~10 ms with the native IQ4 kernel). Host and GPU mostly take turns, so the next steps are fewer launches and
  uploads (CUDA graph replay of the dense part, fused per-layer kernels, one upload per layer) rather than
  faster kernels.- `PYRE_QN_GRAPH=1` (opt-in): the dense part of each DeltaNet layer (36 of 48; ~12 launches each) is
  captured once into a CUDA graph with cubecl's own capture API and replayed on later tokens. Its buffers,
  the DeltaNet state and the residual are persistent (the residual is rewritten in place).
  Nsight: kernel launches 36,176 -> 20,216 over the same 40-token run, plus 36 graph launches per token.
  Decode ~19.2 -> ~19.9 story, ~17.2 -> ~18.2 code; greedy output unchanged. It costs ~2.7 GB more VRAM
  (pages a sealed graph touched stay guarded), leaving ~1.6 GB free, so it is opt-in for now.
  The attention layers (position-dependent arguments) are not captured yet.
- `PYRE_QN_NATIVE=3`: the IQ4_XS dp4a GEMV with x quantised inside each cube. Every cube re-quantises
  x, so it is slower (~17 tok/s); kept as an experiment.
- Speculative decoding on a no-thinking copy task (1,247-token prompt, 232 tokens out): steady 15.0 tok/s
  plain, 16.9 with `--spec 4`, 17.8 with `--spec 8`; greedy output identical. It uses ~2.5 GB more VRAM.
- Prefill is slow: ~10.7 tok/s at 931-1,247 prompt tokens (116 s for 1,247). The likely cause is that the
  expert cache is empty during the first prefill (it fills afterwards) and the routed experts for the
  prompt run on the CPU. To look at later: fill the cache from the first prompt chunk, and use GPU
  batched expert GEMMs for prefill.- Dense GEMV experiments (Nsight with `--cuda-graph-trace=node`, graphs on). Per decode token: ~21.7 ms GPU
  kernel time, ~11.6 ms host CUDA API time, ~505 launches + 34 graph launches, ~106 syncs.
  - Q8_0 `k_gemv` ~43 us/call, ~12 ms/token at ~325 GB/s, close to the RTX 3060's ~360 GB/s.
  - Native IQ4_XS dp4a (`NATIVE=2`) moves half the bytes but runs ~34-44 us/call (~210 GB/s effective):
    each 4 weight bytes need ~11 loads, 4 of them IQ4_NL table lookups. An unrolled x4 version and a
    shared-memory table were no faster, so neither was kept. Without a byte-permute intrinsic (CUDA
    `__byte_perm`, not exposed by CubeCL) the table lookup bounds a 4-bit kernel here.
  - Q8_0 with int8 activations and dp4a: slower (~56 us/call) and changed greedy output on one prompt.
    Not kept.
- CPU experts: 8 pinned P-cores is best (story / code steady tok/s): 18.3 / 16.8 with 8 P-cores, 18.2 / 16.8
  with 16 P-core threads, 17.7 / 16.5 with 8 P + 8 E, 15.7 / 13.8 with 8 P + 16 E.
  Gate/up on IQ2_S dominates (~12 ms/token at ~17 GB/s), so it is decode-bound, not memory-bound.
- CPU experts already overlap the GPU experts and the shared expert (`run_side`). The next layer's dense
  part needs this layer's full MoE output, so there is no further GPU work to hide them under without
  changing the math.
### Strata comparison (from its public docs and v0.2.7 release notes; ideas only, no code)

- **What "46.2 tok/s" is:** Strata v0.2.7, RTX 3060 12 GB, Qwen3.8-Flash-Next **IQ3_XXS** (ISTA-DASLab; 512 experts
  x 48 layers = 24,576, top-10), story answer, route prior off. Same model family as ours, but its experts are 3-bit,
  more bytes than our IQ2_XS. With the 12 GB route prior on by default, it reaches 53.1.
- **It includes speculative decoding.** Setup passes `--mtp`: the model's own MTP draft layer (a separate ~6 GB
  download, not in the GGUF we use) drafts up to 3 tokens, 2.4-3.2 tokens per verify pass, which Strata's docs put
  at 1.6-1.8x. `STRATA_ROUTE_TAIL_SKIP` is also on by default: missed experts at rank 7 or lower in every token of
  a window are skipped. It changes outputs (KL ~0.006) and gave +15.6% decode on a 3060. Strata's docs give no
  no-MTP 3060 figure. From their own factors, plain exact decode is about 46.2 / 1.7 / 1.15, roughly
  **~24 tok/s**, against our 19.9 (`PYRE_QN_GRAPH=1`).
- **GPU vs CPU experts:**
  - Placement: a shipped per-expert-pair profile ranks experts, an adaptive VRAM expert cache follows the
    conversation, and all experts sit pinned in RAM.
  - Missed experts on the CPU: AVX2/AVX-512 kernels (ggml's for the i-quants) at 32-37 us per expert on a 3060 PC,
    against ~80 us for ours. Some misses are streamed over PCIe instead.
  - Result: the CPU share falls from 0.55 to 0.20 of experts (x1.30 decode on a 3060).
  - "Every extra GB of VRAM holds ~700 more experts". The dense weights run in the GGUF's own quants (IQ4_XS etc.),
    not repacked to Q8, so far more VRAM is left for experts than our 7.1 GiB of Q8 dense weights allow.
- **Other ideas named in 0.2.7:** residency-biased routing (route prior), the 28.8 GB per-token lookup table kept in
  RAM, hyper-connection reads split over kernels (on only for RTX 50; slower on sm_86), int8 prompt attention,
  tuned decode split, slots, layer-major prefill.
- **Biggest levers for pyre, in order:** (1) the MTP draft layer (needs the separate MTP file; the user's call),
  (2) a CPU expert kernel ~2x faster, (3) more experts on the GPU by keeping dense weights 4-bit, plus streaming
  some misses over PCIe, (4) the opt-in, output-changing tricks (tail skip, route prior).
### After the Strata study (quiet GPU, graphs on, 96 tokens, two runs each; story pA / code pB)

- **CPU kernel is not the gap.** `PYRE_QN_KBENCH=1` times the IQ2_S gate+up kernel alone: 190 us per expert on one
  thread, 32 us on 8 threads, the same cache-hot or streamed (compute-bound). That is already Strata's 32-37 us. In
  the model the per-layer jobs are small, so the measured ~50 us per miss is pool and sync overhead, not SIMD unpacking.
- **Pool spin** (`PYRE_QN_SPIN_US` 100 / 1000 / 5000 us): steady 19.1-19.8 / 17.3-18.1, within noise. Left at 100.
- **4-bit dense to free VRAM for experts:** `PYRE_QN_NATIVE=3` saves ~1.3 GiB. With the cache at 4 GiB, VRAM matches
  the default (~1.37 GB free): 17.3-17.6 / 17.0-17.1, against the baseline 19.2 / 17.6 (cache 2.5 GiB, Q8).
  Asking for 5-6 GiB tripped the VRAM watchdog at 612 MiB free.
  So the extra experts do not pay for the slower table-lookup kernel. A 4-bit kernel as fast as Q8 (byte-permute
  decode) would turn this into a win: it gives the bytes and the VRAM. It is the next step.
### Exact-output experiments after the Strata study (quiet GPU, graphs on, 96 tokens; story pA / code pB)

- **byte_perm intrinsic:** local cubecl branch `pyre-byte-perm` (worktree `../cubecl-bperm`, commit 33495479). It adds a
  `math.byte_perm` op: CUDA `__byte_perm`, with a portable polyfill on every other backend. A kernel built on it,
  k_gemv_iq4p (IQ4_XS, f32 x in shared memory, levels decoded by prmt instead of table reads), took 70 us per call
  against 56 us for the table version and 47 us for Q8_0. Greedy output also changed, so the kernel has a bug or a
  numerics difference. It is reverted and pyre still builds against the clean fork. The real limit of the 4-bit
  kernels looks like the shared-memory reads of x, not the level lookup.
- **Expert-ranking profile** (`PYRE_QN_XPROFILE=<file>`, opt-in; build one with `PYRE_QN_XPROFILE_SAVE=<file>`):
  counts from two other prompts (story and Python), weighted as 256 tokens of routing, seed and steer the cache.
  It changes placement only, and greedy output is identical. Result: story 17.1-17.3 vs 18.3-19.3, code 17.0-17.5
  vs 17.6, so it is off by default. A lighter prior, or seeding only the very first (pre-prefill) fill, are untested.- **Prompt-free 4-bit kernel, global-x variant** (k_gemv_iq4g: byte_perm, x from L1 like k_gemv): 85 us per call, so
  slower still, and greedy differed. Reverted.
- **PYRE_QN_NATIVE=1** (the shared-memory IQ4_XS kernel) matches the Q8 greedy output and can now be graph-captured.
  With graphs on, VRAM use is the same as Q8, because the cache takes what is free. With the cache cap at 4 GiB:
  story 18.8-19.3 vs 19.3-19.4, code 18.2-18.5 vs 17.9-18.0. That is too small to change the default.
- **Env snapshot:** decode-path `PYRE_*` reads come from a snapshot instead of `std::env::var` (the process
  environment lock in every pool task). Steady 19.0-19.6 / 17.4-18.0, unchanged: not the pool overhead.
- **Prefill on the int8 pool path** (`PYRE_QN_PF_POOL=1`, opt-in): 931-token copy prompt, **10.8 -> 15.3 tok/s**
  (85.9 s -> 60.8 s), with the same 32 greedy tokens. It is opt-in only because the prompt's experts then use the
  decode path's int8 rounding instead of f32. Profile per 256-token chunk (~16 s): hyper-connections 3.3-4.0 s and
  DeltaNet 4.6 s still run on the CPU, routed experts 5.8 s (gate/up 4.0, down 1.25), attention 1.1 s. The next
  prefill steps are hc and DeltaNet on the GPU for chunks, then GPU experts streamed per layer (Strata's approach).
### Decode timeline of one token (PYRE_QN_PROFILE=1, graphs on, story prompt, ~50 ms/token)

Per layer the host does: enqueue the dense segment (graph replay or kernels), then **wait** for its readback
(router logits + FFN input), then route. The pool runs the missed experts on the CPU while the same thread
enqueues the cached experts and the shared expert on the GPU, then uploads the CPU sum and enqueues the combine.
The next layer cannot start before that combine, so GPU and CPU alternate:

| per token | ms |
|---|---|
| host enqueue (graph launches, attention layers' kernels, comb3, y upload) | ~4 |
| waiting for the GPU dense segment (`reads`: ~18.5 ms dense kernels + ~2.5 ms sync/launch latency) | ~21 |
| CPU experts (gate/up ~11, quantize ~1, down ~3.5-4) with GPU experts + shared expert overlapped (~5 ms GPU) | ~16 |
| LM head on the CPU | ~3 |
| PLE layer (1 of 48, host round trip) + embedding, sampling, final readback | ~5 |

GPU kernel time per token (Nsight): k_gemv 13.1 ms, k_xgu_s 2.9, k_delta 2.2, k_gate 1.8 (94 launches),
k_xdown_q 1.6, rms 0.5, attention 0.4, the rest < 0.3 each. So the ~15 ms of GPU idle time per token is the
CPU expert phase (~11 ms of it, after the GPU experts finish) plus head and host gaps, and the CPU idles during
the ~21 ms dense wait. That chain is a true data dependency (each layer's experts need its router output, and
the next layer needs their sum), so it cannot overlap without changing the math. What can shrink it:
fewer CPU misses, faster dense kernels, and faster CPU experts.

Tried on that critical path (all greedy-identical):
- **PCIe share** (`PYRE_QN_XSTREAM=K`, opt-in): K misses per layer are uploaded into reserved cache slots and run
  with the GPU experts. 1/2/4 streamed experts per layer: 7.8 / 4.5 / 2.7 tok/s. The cost is the host repack in
  `xupload`, 1.1 ms per expert (54 ms of repack per token at K=1), plus ~0.33 ms per expert of pageable PCIe
  transfer (~4.5 GB/s). It needs experts kept pre-repacked in pinned memory to be viable; that does not fit next
  to the 33 GB resident copy.
- **Pool layout:** SBLK 16, SDBLK 32, 9 threads (caller plus 8 pinned workers): no gain (within noise).

### Prefill on the GPU (`PYRE_QN_PF_MULTI=K`, opt-in)

The prompt goes through the decode verify path K tokens at a time: hyper-connections, projections and
DeltaNet run on the GPU, without rollback checkpoints, and the expert cache fills after the first piece.
CPU gate/up for tokens that share an expert decode each row once (`qdot::row_dot_k_multi`, bit-identical
to the one-token kernel). 931-token copy prompt:

| setting | prefill tok/s | greedy vs default f32 prefill |
|---|---|---|
| default (CPU dense f32) | 10.8 | - |
| PF_POOL=1 | 15.3 | identical |
| PF_MULTI=4 | 16.7-17.9 | identical |
| PF_MULTI=8 | 19.4-20.5 | differs at token 24 |
| PF_MULTI=32 | 22.2 | differs |
| PF_MULTI=128 | 23.9 | differs |

K >= 8 diverges at a near-tie: the prompt's experts then run partly on the GPU (cache) with different rounding.
The checkpoint-free path gives the same tokens as the checkpointed one. Per 128-token piece (~5 s): host
enqueue 0.95 s (per-token kernel launches), dense wait 1.35 s, CPU experts 2.0 s (gate/up 1.4, down 0.45).
Streaming a layer's experts to the GPU per piece is the next step, but at the current pageable ~4.5 GB/s plus
1.1 ms host repack per expert (~770 MB per layer) it is slower than the CPU. It needs pinned, pre-repacked
staging and pieces of several thousand tokens, as in Strata, to amortize.
**Not made the default.** The gated check had PF_MULTI=4 on by default, with a quiet GPU and graphs on.
- The 931-token prefill was again greedy-identical, at 17.8 tok/s.
- But decode after a short prompt was slower and less stable than with `PF_MULTI=0` (same session):
  story 18.2 / code 15.1 steady, against 19.2 / 16.8.
- The cause: the expert cache was filled after the first 4-token piece, so it was ranked on 4 tokens.
- Moving that fill to after the whole prompt gave 18.2 tok/s prefill, but greedy then differed on both the
  931-token prompt and the code prompt (cache placement changes GPU vs CPU rounding), and decode was still lower
  (18.1 / 15.6).
- So PF_MULTI stays opt-in (default 0). Making it the default first needs the decode cache-fill and rounding
  sensitivity sorted out.
### MTP drafting and multi-row verify GEMV (Oct 29, after reading Strata's source)

Strata (github.com/Niko1221/Strata, MIT) source was read for this round. The files whose design follows it say
so in their headers; Strata's draft-vocabulary files are copied under `data/` (see THIRD_PARTY_NOTICES). No Strata
C++/CUDA is linked or vendored.

* **MTP weights.** The GSQ-RCO GGUF has no MTP head. Like Strata's `tools/mtp_fetch.py`, `scripts/mtp_fetch.py`
  range-reads only the 31 `mtp.*` tensors (5.21 GB, BF16, SHA256 of each checked against Strata's pinned list) from
  `Qwen/Qwen3.8-Flash-Next@de4b8e4d` into one file in the HF cache:
  `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d.../mtp.safetensors`.
  pyre finds it there (or `PYRE_MTP=<file>`, `PYRE_MTP=0` off).
* **Drafter** (`src/model/mtp.rs`, CPU). It is Strata's MTP layer: cell i pairs the main model's final 4-stream
  residual R_i with token i+1 at rope position i. The path is fc_embedding / fc_hidden ? attention
  hyper-connection ? gated attention over the drafter's own K/V (dense, sliding window `PYRE_MTP_WINDOW`, default
  256) ? MLP hyper-connection ? MoE (512 experts, top-10, gated shared expert) ? final mixer ? the main head. The
  drafter's residual feeds the next draft step, and K/V catch-up for verified cells runs batched.
  * Weights are converted to int8 at load (1.4 s; all 512 experts up front, `PYRE_MTP_LAZY=1` on first use).
  * The draft head is the main GPU head (full vocabulary). `PYRE_MTP_HEAD_CPU=1` uses a CPU head over a token
    subset (`PYRE_MTP_DRAFT_VOCAB=en` default | `cjk` | `full` | file, Strata's lists).
  * `PYRE_MTP_MIN_P` stops a chain below a draft probability (Strata's `--spec-min-p`, default 0).
  * About 7 ms per draft step.
  * Acceptance of greedy drafts: 6877% on story and code (Strata reports 0.850.89 with its GPU drafter).
* **Greedy identity.** With `--spec 2/3/4` the output matches plain decode token for token on every run
  (96 tokens story, the full code answer). Wave mode and tail skip match too.
* **Verify GEMV** (`k_gemv4r`, default; `PYRE_QN_GEMV4R=0` gives the old kernel). Nsight showed the multi-token
  verify's 4-column GEMV costing 3.6× a single GEMV, limited by activation loads, not by weights. Each warp now
  takes 4 rows, so each activation value is loaded once for 4 rows, with every (row, column) summed in exactly
  `k_gemv`'s order.
  * GEMV time per T=4 verify: 43 ? 27 ms.
  * T=4 verify: ~147 ? ~120 ms.
  * 8 rows per warp (`PYRE_QN_GEMV4R=8`) is slower because of register pressure.
* **`PYRE_QN_WAVE=1`** (opt-in): the verify window in two token groups, pipelined so the GPU runs one group's dense
  half while the CPU runs the other's experts. It is greedy-identical but no faster: the GPU wait fell 69 ? 40 ms
  while CPU expert time rose 52 ? 80 ms because the expert work is no longer shared across the whole window.

Measurements: quiet GPU, graphs on, 96 tokens, steady tok/s. VRAM is unchanged, with ~1.4 GB free.

| | story (pA) | code (pB) | output |
|---|---|---|---|
| plain decode | 18.620.8 | 16.719.0 | reference |
| `--spec 2` (MTP) | 18.119.3 (77% accepted) | 17.4 (75%) | identical |
| `--spec 3` (MTP) | 18.418.7 (69%) | 15.8 (75%) | identical |
| `--spec 4` (MTP) | 16.3 (66%) |  | identical |
| `PYRE_QN_TAIL_SKIP=7` | 23.0 | 20.7 | story same 96 tokens; code differs from token 4 |
| tail skip 7 + `--spec 2` | 21.7 (80%) |  | identical to tail skip alone |
| `PYRE_ROUTE_PRIOR=1.5` | 23.3 |  | differs from token 40 |

So MTP speculation is correct but **not faster here**, and it stays opt-in (`--spec`, default 0). The reason is
how verify scales on this machine. A window of T tokens costs about T plain steps:
* the CPU experts scale with the distinct experts in the window (~3.3× for T=4);
* DeltaNet and the per-token kernels run T times.

A round with 2.52.9 accepted tokens therefore costs about as much as the tokens it saves. Strata gets its MTP
speedup with a larger share of the experts on the GPU and the whole drafter on the GPU.

`PYRE_QN_TAIL_SKIP=R` (opt-in, changes outputs, default off) is new. It skips an expert that missed the GPU cache
when the expert ranks R or lower among that token's top-k. Strata turns its version (rank 7, per verify window)
on by default on CUDA, which accounts for part of the gap between its decode speed and our exact decode.

### CPU expert rows: AVX-VNNI and hybrid-core workers

Following Strata's AVX2 i-quant expert kernels (iq_avx2), which take an AVX-VNNI copy of the row loops on Alder Lake and
later, pyre's IQ2_S / IQ2_XXS / IQ1_M Q8_K rows use `vpdpwssd` for each madd + add where the CPU has AVX-VNNI (the same
integer sums, so identical output; `PYRE_QN_VNNI=0` turns it off). The expert pool now also puts one pinned worker on
each E-core after the P-cores (`PYRE_QN_ECORES=0` restores P-cores only). On an i9-13900K + RTX 3060 these took plain
decode from ~20.3 to ~22.3 tok/s, greedy-identical. `PYRE_QN_GATHER=1` (vpgatherdq grid decode, also after Strata)
is opt-in: no measurable gain here.

### Host expert arena, PCIe share, refills and draft head

- Strata keeps missed experts pre-packed in the GPU layout in a host arena (pinned in 1 GiB steps above a RAM reserve) and
  sends a 15-20% share of each verify window's distinct misses over PCIe to run beside the CPU pool. pyre has the same as
  opt-in: `PYRE_QN_XARENA_GB=<GiB>` fills a pre-packed host arena at load, hottest first by the routing profile, and stops
  while less than `PYRE_QN_RAM_RESERVE_GB` (default 16) would stay free; `PYRE_QN_XSHARE=<0..1>` sends that share of the
  misses (T=1: the lowest-weight ones; verify: the distinct experts most tokens use, uploaded once per window) to the GPU;
  `PYRE_QN_XPIN=1` stages uploads through CubeCL's page-locked pool. On an RTX 3060 + i9-13900K this is slower
  (22 -> 15-17 tok/s): CubeCL uses one stream, so every upload is serialized with compute, and the host copy per
  expert costs more than the 24-worker CPU pool needs for it. The GPU math also changes the output, so it stays opt-in.
- Expert cache refills repack their experts in parallel (the same experts and slots as before, so the same output).
- `k_gemv4r` (verify GEMV) compiles per column count, so T=2/3 windows no longer compute a padded 4th column, and
  decodes bytes with the 2^23 mantissa trick (exact).
- The MTP draft head runs on the GPU over the draft vocabulary (~100 MB VRAM, 2.2 -> 0.75 ms per draft; same drafts).
  `PYRE_MTP_HEAD_GPU=0` turns it off.
- The drafter's CPU matmuls run on their own 8-thread rayon pool (`PYRE_MTP_THREADS`, 0 = global pool): ~2-3 ms less per 3-draft round. GPU calls stay on the main thread (a CubeCL call from another thread opens a second stream and ~200 MB memory pool).
- `PYRE_QN_HPIN=1` (opt-in): small host-to-device buffers go through CubeCL's page-locked staging pool; within noise here (~+0.3 tok/s).
- `PYRE_QN_XS2=1` (opt-in): PCIe-share uploads are issued from a dedicated thread (CubeCL gives it its own CUDA stream). Same output as the single-stream share, no faster here: the host-side copy (pageable staging of each 1.4 MB expert) is what costs, and it still sits on the enqueue thread.


### Fused DeltaNet verify recurrence (PYRE_QN_DELTAM, default on)
During speculative verify (2-4 tokens), each GDN layer runs one k_delta_m launch that keeps the
state column in registers across the window and writes the per-token checkpoints, instead of one
k_delta launch per token. Greedy-identical (same per-token arithmetic). Measured on RTX 3060:
spec2 pA 22.9 -> 24.2 tok/s, spec3 pA 21.2 -> 22.3, spec2 pB 18.8 -> 20.0. PYRE_QN_DELTAM=0 disables.


### MTP spec2 is the default
`qwen-next` and the server now default to speculative decoding with draft length 2 (MTP drafter) when
the MTP file is found and multi-token verify can run; otherwise they fall back to plain decode. Greedy
output is identical. `--spec 0` (or PYRE_SPEC=0 / config `spec: 0`) disables; `--spec N` sets the length.


### Opt-in experiments (greedy-identical, no measured gain yet)
- PYRE_QN_ETAIL=N: the last N tasks of each expert-pool job go to P-core workers only (E-cores are ~2.3x
  slower and would finish last). Default 0; N=8/16 measured within noise of off.
- PYRE_QN_GEMV4R=v: k_gemv4v, the multi-column verify gemv with 16-byte vector activation loads (same
  lane mapping and summation order as k_gemv4r). Same output; no clear speedup in a noisy session.
- Measured and not kept: 32 pool workers incl. P-core hyper-threads (-5%), longer pool spin (3 ms) and
  4/16/24 MTP drafter threads (8 stays best).


### Windows power throttling opt-out (default on)
With the user at the desktop, Windows' EcoQoS throttled pyre (a non-foreground console job) to ~15 tok/s.
`qwen-next`/`serve` now ask Windows not to throttle this process (SetProcessInformation
ProcessPowerThrottling, per-process; no system setting changes). Measured on HYTE with the desktop in use:
plain pA 15.0 -> 22.2 tok/s, spec2 pA 16.3 -> 23.6. PYRE_ECOQOS=1 restores the old behaviour.


### Expert-pool pre-wake (PYRE_QN_PREWAKE, default on)
Just before blocking on each layer's router readback, the main thread wakes parked pool workers and restarts
their spin window, so the layer's expert job starts on already-spinning workers. Greedy-identical (timing
only). Steady decode, alternating runs: plain pA 23.7 -> 24.5 tok/s, spec2 pB 22.8 -> 24.1.
PYRE_QN_GEMV4R=v re-measured on a quiet machine: no gain (stays opt-in).


### PCIe share re-measured (arena on, single-copy uploads)
CubeCL's write already stages pageable data through its pinned pool and copies asynchronously, so uploads
cost one host copy plus the DMA. With PYRE_QN_XARENA_GB=8 the share is still a large loss on HYTE (plain
pA: off 24.7 tok/s, XSHARE=0.25 12.1, XSHARE=0.5 7.2; XPIN makes no difference). Each shared expert moves
~2 MB over PCIe, which takes longer than the CPU pool needs to compute it, so a second stream or
zero-copy host reads (also PCIe-bound) cannot make it pay here. It stays opt-in; no cubecl fork change.

### Expert cache aging (opt-in, PYRE_QN_XDECAY=k)

Before each cache refill every expert's routing count loses 1/2^k, so recently routed experts weigh more (LFU with aging). Output stays greedy-identical. Measured on HYTE (3 alternating pairs, steady tok/s): plain A 25.18/25.48/24.77 with k=5 vs 24.47/24.33/24.77 default (+2.5%); spec2 B 24.60/23.09/24.47 vs 24.07/23.68/24.45 (no change). Not consistent on both prompts, so it stays opt-in. PYRE_QN_XREFILL=8 was neutral; PYRE_QN_XSWAP=128 changed output on prompt B.
### Gate+up pair kernel (opt-in, PYRE_QN_PAIR=1) and draft length

- PYRE_QN_PAIR=1 runs a CPU expert's gate and up rows through one interleaved IQ2 kernel. Results are bit-identical to two row_dot_k calls (checked in PYRE_QN_KBENCH). KBENCH gate+up time per expert: 1 thread 198 -> 195 us; 8 threads streamed 33.2 -> 29.5 us. In the model (4 alternating pairs, steady tok/s) it averaged 24.54 vs 24.50 on plain A and 24.05 vs 23.95 on spec2 B, which is noise, so it stays opt-in.
- Merging the down step into the gate/up tasks was not built. Down needs an expert's whole hidden vector (quantised per 64 values) before any output row, so a merged task would cover a whole expert. With ~5 misses per layer that is too coarse for 16 threads.
- Draft length (adaptive length on, as before): --spec 3 gave 22.1 (A, 70.7% accepted) / 22.6 (B, 70.6%); --spec 4 gave 23.4 (A, 65.2%) / 19.7 (B, 60.0%). The spec2 default is ~24-24.5, so it stays.
### Verify cost and draft length 1 default

PYRE_SPEC_TIME=1 prints wall time per phase (draft, verify by window size, keep, plain steps). Measured on prompt A with PYRE_QN_PROFILE, one plain token against one 3-token verify:
- Plain token: ~43 ms. GPU wait 20 ms, CPU misses ~11 ms (~170 expert misses).
- 3-token verify: ~78 ms (1.8x). GPU wait 36-39 ms (1.8x), CPU routed 25-28 ms for 450-610 distinct experts (630-700 expert-token entries; the window shares only ~15%), enqueue 3 ms.
- Drafter (CPU): ~4.5 ms for one draft, ~7.7 ms for two (plus head ~0.8 and catch-up ~1.0 ms).
- The tokens of a window route to mostly different experts, so CPU work grows almost linearly with the window. A 2-token verify costs less per accepted token than a 3-token one.

Default draft length is now 1 (3 alternating pairs, steady tok/s, all greedy-identical): A 26.24/26.15/26.03 vs 25.41/25.39/25.53 with k=2; B 23.70/22.68/24.08 vs 22.79/23.02/23.92. Acceptance is 87.5% / 86.4% at k=1. --spec 2 still works.
### Multi-token gate kernel and confidence-gated drafts (default)

- **Per-kernel nsys, plain token vs 3-token verify (prompt A, per round):** gemv 11.8 -> ~16.4 ms (k_gemv4r reads the weights once, 1.4x). Cached-expert kernels k_xgu_s / k_xdown ~3.3x (GPU hits grow with the window's distinct experts). k_gate 1.6 -> 5.4 ms (3.4x: it ran once per token).
- **k_gate_m** (default on, PYRE_QN_GATEM=0 off) runs the hyper-connection gate for 2-4 window tokens in one launch. Each weight word is read once; each token's arithmetic is unchanged, so output is identical. Measured +0.5-1%.
- **Confidence-gated draft length** (after Strata's draft policy, src/spec/draft_policy.cpp, MIT): the default is draft length 2 again, but the second draft is kept only when the drafter's probability is at least 0.9 (PYRE_MTP_MIN_P, 0 = always draft the full length). The sweep (2 runs each, steady tok/s, all identical):

| | A | B |
|---|---|---|
| k=1 | 27.81 / 27.72 | 24.32 / 23.97 |
| k=2, p>=0.6 | 28.70 / 28.31 | 23.85 / 23.71 |
| k=2, p>=0.8 | 28.79 / 29.01 | 23.61 / 24.41 |
| k=2, p>=0.9 | 29.52 / 29.38 | 23.37 / 24.64 |
| k=3, p>=0.8 | 27.52 / 27.87 | 24.84 / 24.40 |

  New defaults: A 29.0, B 23.9-24.5, plain A 24.2 (A is 1.2x over plain).
- **Prompt-B VRAM gap:** the expert cache does fill (1691 experts, 2.50 GiB, 44% hits). The lower peak reading is a sampling artifact in the bench script, not a missing cache.
### Lookup drafting and GPU drafter matmuls (both opt-in)

- **PYRE_SPEC_LOOKUP=n** (PYRE_SPEC_LOOKUP_K=3): when the last n+ tokens occurred earlier in the context, their continuation is drafted instead of running the MTP drafter (lookup mixed with MTP, after Strata's draft policy). On prompts A/B it rarely fires (0 rounds on A, 2 on B) and those drafts were rejected. Steady A 29.86/28.64 (n=6) and 29.75/28.99 (n=10) vs 27.39 (cold)/29.76 default; B 23.18/22.93 and 24.33/23.81 vs 23.68/24.29. Opt-in until tested on repetitive text.
- **PYRE_MTP_GPU=1** uploads the drafter's 14 dense matrices (94 MB Q8) and runs them as GPU GEMVs from the main thread; experts stay on the CPU. Draft numerics differ, output does not. Steady A 29.88/30.05 vs 27.00 (cold)/28.63; B 24.12/24.60 vs 24.21/24.70. Total draft time did not drop measurably, because each matrix is one upload + readback. Opt-in; a fused GPU drafter step (no per-matrix readback) is the next step.
- The calibrated confidence table was not ported. At p >= 0.9, A already accepts 100% of second drafts and B rarely drafts two.
### Prompt C (code rename), lookup drafting and drafter top-k

- Prompt C (C:\Temp\qnx\pC.txt, raw text: a Python function, then "the same function with items renamed to values"). Plain decode 22.6 tok/s; default spec 27.0-28.3 (97-98% accepted).
- PYRE_SPEC_LOOKUP on C, 2 alternating runs, steady tok/s: default 27.72/28.28; n=6 K=3 27.71/28.58; n=8 K=6 28.84/28.10; n=4 K=4 27.78/28.68. Lookup fires in 10-15 of 34 rounds and cuts draft time per round from 15.7 to ~12 ms. The window stays capped by the adaptive length, and the MTP drafter already hits ~98% here, so the round count does not drop. It stays opt-in.
- PYRE_MTP_TOPK=k (opt-in, draft-only) lowers the drafter's expert top-k. Warm averages, A/B/C: default 29.2/24.5/27.3; k=4 28.1/23.8/27.5; k=2 28.6/24.0/27.7; k=4 with PYRE_MTP_GPU=1 29.1/24.8/27.9. Acceptance moves slightly (A 100 -> 95-97%). Speed is all within noise, so it is not a lever here.
### Strata-equivalent output-changing settings, PYRE_QN_FAST, high-acceptance drafts

Strata 0.2.7 on a 12 GB NVIDIA card turns on STRATA_ROUTE_PRIOR=1.5 by itself. STRATA_ROUTE_TAIL_SKIP=7 is its CUDA default, but its setup turns it off when every expert is in RAM, as on HYTE. pyre on top of its defaults, steady tok/s (2 runs each) and the first token that differs from greedy:

| | A | B | C |
|---|---|---|---|
| defaults | 28.9 (identical) | 24.2 (identical) | 27.9 (identical) |
| PYRE_ROUTE_PRIOR=1.5 | 28.8 (from token 40) | 25.8 (from 4) | 29.9 (identical) |
| PYRE_QN_TAIL_SKIP=7 | 30.5 (identical) | 23.9 (from 4) | 29.9 (identical) |
| both = PYRE_QN_FAST=1 | 29.4 / 28.1 (from 40) | 26.4 / 26.5 (from 4) | 29.8 / 30.3 (identical) |

PYRE_QN_FAST=1 (opt-in) sets both; explicit PYRE_ROUTE_PRIOR / PYRE_QN_TAIL_SKIP values win.

PYRE_SPEC_HI=acc (opt-in): while recent acceptance is at least acc, drafts may run to PYRE_SPEC_HI_K (4) with the later-draft gate at PYRE_SPEC_HI_P (0.6). With 0.95 / 0.9: A 26.3-27.9, B 23.0-24.2, C 25.8-26.5, against default A 28.7-28.8, B 22.7-23.4, C 26.3-27.2. Longer windows cost more than their extra accepted tokens, so it stays off.

Drafter time per round (PYRE_MTP_DEBUG): ~9.5 ms (head 1.3, catch-up 1.0, 1.5-1.9 steps; experts ~2.1, attention ~1.5, input ~1.3, rest ~3.3) of a ~89 ms round on A, about 11%. A full GPU drafter would save perhaps 6-7 ms (<10%), so it was not built.
### Opt-in: Strata's adaptive VRAM tier (PYRE_QN_ADAPT=1)

Ported from Strata (MIT), src/program/generate.cpp `adapt`: every PYRE_QN_ADAPT_EVERY tokens (default 12), per layer the most-routed missing experts (decayed count >= 2) replace the least-routed resident ones when they beat them by more than PYRE_QN_ADAPT_GAIN (1.5); at most PYRE_QN_ADAPT_SWAPS (96) a round; counts then decay by PYRE_QN_ADAPT_DECAY (0.7). Quick runs (RTX 3060, busy desktop): A 26.1 vs 27.0, B 22.3 vs 21.6, C 25.0 vs 24.6 steady tok/s; B and C greedy-identical, A diverges (cache contents change GPU/CPU expert placement), so it stays opt-in.

PYRE_VRAMLOG=1 prints NVML VRAM at each load stage and during decode. Finding: with PYRE_QN_GRAPH=1, VRAM grows ~2.7 GiB during decode (37 recaptures; old graphs' persistent buffers are kept), which is what left only ~1.2 GiB free; without graphs pyre peaks at ~8.2 GiB with the 2.5 GiB cache. PYRE_QN_GPUHEAD=1 no longer also uploads the burn LM head copy and no longer panics on the empty drafter subset head.

### Overnight Oct 30: NATIVE=2 profile (VRAM-light, fastest; output differs from Q8 on prompt B at token 32)
- With PYRE_QN_NATIVE=2, long drafts are on by default (PYRE_SPEC_HI 0.5, up to 6 drafts; windows 5..8 run as 4+rest waves).
- The wave verify no longer blocks on a flush after enqueueing the next group (PYRE_QN_WFLUSH=1 restores); same output.
- With PYRE_QN_NATIVE=2 the LM head is Q8 on the GPU by default (PYRE_QN_GPUHEAD=0 off), and the verify head runs as one
  batched gemv_m per 4 tokens (the burn f16 head GEMV took 5-11 ms per round). Same output on A/B/C. The MTP drafter keeps its
  draft-vocabulary GPU head (PYRE_MTP_HEAD_FULL=1 drafts with the full head instead).
- Steady tok/s, 64 tokens, single runs (A/B/C): Q8 default ~30/22/26; NATIVE=2 now 32-33/25/33-35 (all identical to their own baselines).
- Tried, no gain (opt-in or reverted): PYRE_QN_KICKQ / PYRE_QN_READNB (non-blocking queue hand-off and readback request via new
  cubecl ComputeClient::flush_queue / read_async_nb: the wait just moves to the next blocking call), CUBECL_DROPQ_ASYNC, 3+2 wave
  split, PYRE_QN_NATR_MIN, a 3.5 GiB expert cache, PCIe share with the pinned arena (rarely fires: type-limited), draft cap 3 vs 6.
  GPUHEAD on the Q8 dense profile hits a ~10.3 GB VRAM transient on A, so it stays NATIVE=2-only.
- nsys (NATIVE=2, A): GPU busy ~46% of decode wall; k_gemv_iq4d_r is ~25% of GPU time; ~23 launches/ms at ~9 us each on the device thread.

### Oct 30 (early morning) follow-up
- k_gemv_iq4d_r now 4 warps per cube by default (PYRE_QN_NATR_Y; sweep 2/4/8/16, 4 best by ~1 tok/s on A/C).
- Tried, no gain (all within run-to-run noise of +-2 tok/s): expert pool ETAIL/PAIR/ECORES=0 (E-cores off is -4 tok/s), bigger expert cache (PYRE_QN_XCACHE_GB 3.0/3.5), PYRE_SPEC_HI_K 4/5 and PYRE_SPEC_HI 0.3/0.7, PYRE_MTP_THREADS 2/4/16, PYRE_QN_SPIN_US=20, fused q/k/v GEMV in the drafter step.
- Profile (NATIVE=2, prompt A): drafting is ~27 ms per round (C ~41 ms), ~4 ms per draft step independent of thread count or CPU-vs-GPU dense matrices; this is now the largest non-verify cost. Next lever: whole drafter step on the GPU as one submission (no per-matmul readbacks/CPU glue).
- NATIVE=2 not made default: its output matches the Q8 references on A/C (64 tokens) and B (42) in most runs, but one B run diverged at token 32 (verify batch size changes numerics), so it is not strictly greedy-identical.

### Oct 30 (morning) results
Steady tok/s, n=96, one run each (Q8 default / PYRE_QN_NATIVE=2): A 31.8 / 35.0, B 25.4 / 27.6, C 28.5 / 36.3 (overall 30.4/23.0/27.4 and 27.3/25.2/34.5). Q8 default identical to the references on A/B/C; NATIVE=2 differs at near-ties (B token 32, C token 76).
- Drafter: GPU copies uploaded at load (d3f78e2), attention window 128 (c8cf071), GEMV kernels compiled at load (b2b9d5a; removed a ~180 ms mid-decode JIT stall).
- PYRE_QN_NATX=1 (opt-in): native IQ4_XS GEMVs with Strata's / ggml's vec_dot_iq4_xs_q8_1 arithmetic. Strata's exact mode computes dense layers natively in IQ4 (its pack serves quantized dense tensors from the GGUF), so it is a different computation from pyre's Q8_0 path; near-tie tokens move with any rounding change, so NATIVE=2 cannot be made reference-identical by precision fixes.
- PYRE_QN_PREWARM=1 (opt-in): decode/verify kernels compiled at load; no clear gain.
- Q8 with 3-token drafts needs a smaller expert cache to stay inside the VRAM reserve (2.0 GiB fits) and is then slower than the default.
- Remaining lever (not started): Strata's verify pipeline waits on host-mapped flags on the GPU (no per-layer host readback) and streams ~55% of expert misses over PCIe with a copy kernel; see C:\Temp\qnx\overnight.md for the design notes.

### Verify MoE glue as a graph (opt-in, PYRE_QN_MOEGRAPH=1, parked)

With VGRAPH on, the GPU side of each verify layer's MoE glue becomes one captured graph per (layer, window size). That covers the window q8pack, the cached experts' gate/up and down, and the shared expert's silu_mul and down GEMV. The routing goes in as a fixed, padded table (t * top_k entries) plus per-token counts in a 4-word device buffer, written before each replay. The kernels k_xgu_stg and k_xdown_qtg skip the padding entries. The idea comes from Strata (MIT, github.com/Niko1221/Strata); none of its code is used. Greedy output is identical on Q8 A/B/C (n=96). Launches per run drop about 40% (CUBECL_LAUNCH_LOG: ~25k to ~15k), but tok/s does not change (A 31.9/32.8 off vs 31.6/32.7 on). Launches are not the bottleneck, so the flag stays opt-in. comb3_m is not in the graph yet.

### Expert cache policy checks (Oct 30, Q8 defaults, n=96, PYRE_QN_XSTAT=1 logs hit rate and misses/layer/token)

There are 1691 uniform slots of 1.587 MB in 2.5 GiB. Expert sizes per layer are 1.28 MB (3 layers), 1.38 MB (11) and 1.587 MB (34), so slots sized per class would fit about 6% more. Baseline hits: A 58.9% (4.1 misses per layer per token, top-10 routing), B 38.5% (6.2), C 43.1% (5.7). XDECAY=5 or 3 leaves hits within 0.1%. PYRE_QN_XPERLAYER=1 (equal per-layer shares, like Strata's static per-layer slot ranges) is neutral (58.3/38.6/42.9%). Both keep greedy output identical. ADAPT=1 raises A to 66.1% but changes greedy output on A: which experts run on the GPU and which on the CPU can change results, so every placement change needs a greedy check.

### pyre-chat (native GPUI chat client)
`gui/` also builds `pyre-chat`, a chat-only desktop client: `cargo build --release --manifest-path gui/Cargo.toml --bin pyre-chat`
(on HYTE: `cd gui; C:\Temp\qnx\build.ps1 --bin pyre-chat`). Run `pyre-chat.exe [server-url]` (default http://127.0.0.1:8099,
or set it in Settings; e.g. http://192.168.0.30:8099 from another machine). Features: conversation sidebar (new/rename/delete,
saved as JSON in %APPDATA%\pyre-chat or ~/.local/share/pyre-chat, override PYRE_CHAT_DIR), reasoning effort picker
(none/low/medium/high/very high/auto, default medium) sent as `reasoning_effort`, collapsible thinking with thinking tokens and
auto's choice/rating, Markdown with code blocks + Copy, Stop (Esc), system prompt / temperature / max tokens, tok/s and context
usage. Keys: Enter send, Shift+Enter newline, Ctrl+N new chat, Ctrl+V paste, Ctrl+Backspace delete word.
Scripted demo env: PYRE_CHAT_SEND, PYRE_CHAT_EFFORT, PYRE_CHAT_OPEN_THINK, PYRE_CHAT_SETTINGS, PYRE_CHAT_STOP_AFTER.
