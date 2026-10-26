# pyre

A llama.cpp `llama-server`-style inference server written in Rust where **[Burn](https://github.com/tracel-ai/burn) 0.22** hosts and runs the model. No llama.cpp, candle or ggml does any of the compute: every matmul, attention and norm is a Burn tensor op on CUDA (CubeCL), wgpu (Vulkan/DX12/Metal) or the CPU (Flex / NdArray).

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

- **Dependency:** `whisburn-core`, `whisburn-audio`, `whisburn-engine` and `whisburn-models` come in as
  **git** dependencies on the local repo, pinned to commit `d27f1fc`
  (`git = "file:///C:/Build/machine-learning/whisburn"`). That keeps pyre building against committed
  whisburn code rather than its working tree. To bump it, change `rev`.
  Cargo feature `stt` (on by default) can be dropped with `--no-default-features --features cuda,wgpu,flex`.
- **Burn versions:** whisburn is on **burn 0.21** and pyre is on **burn 0.22.1**. Cargo treats them as
  separate crates, so both are compiled and linked (it builds fine, but the first build is longer and
  the binary is bigger). The STT model runs on burn 0.21's wgpu/ndarray runtime and the LLMs on
  0.22's. They share no tensors, so this works, but there are two GPU runtimes and two autotune caches.
  The long-term fix is for whisburn to move to burn 0.22 (see below).
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

`response_format`: `json` (default), `text`, `srt`, `vtt`, `verbose_json` (segments + duration + language).
`/v1/audio/translations` runs whisburn's translate task. `language` is honoured; `prompt`, `temperature`
and `timestamp_granularities` are accepted and ignored. Input is decoded by whisburn-audio (symphonia:
wav/mp3/flac/ogg-vorbis/m4a-aac). Browser mic recordings (webm/opus) may not decode.

Measured on an RTX 3060 with a 7.5 s clip (Windows SAPI voice) and `tiny_en`: wgpu 0.8-1.2 s per request
(6-9x realtime; the first request is about 4.9 s while kernels compile), ndarray CPU 11.2 s (0.7x). The
transcript was exact except "Pyre" was heard as "Fire".

**whisburn changes that would help (not made; whisburn is untouched):**
1. Move to burn 0.22 so pyre shares one Burn/cubecl (and gets CUDA for STT).
2. Accept a models dir per `InferenceRuntime`/`load_model` call instead of the global `WHISBURN_MODELS_DIR`.
3. Load Whisper straight from HF safetensors (like the qwen3/parakeet paths) so pyre can reuse the HF cache with zero extra disk.
4. With `include_timestamps = false`, Whisper decoding stops after the first sentence/window; pyre works around it by always requesting timestamps.
5. Stop forcing `fancy-regex` on `tokenizers` (pyre uses `onig`; it builds, but both regex engines get compiled in).

## Image generation (Stable Diffusion 1.5)

Burn runs every step: the CLIP text encoder, the UNet, the VAE and DDIM. pyre uses
[diffusers-burn](https://github.com/oxiglade/diffusers-burn) (MIT/Apache, a port of diffusers-rs),
pinned to rev `9061882` as a git dependency. It runs on **burn 0.20**, so a pyre build with every
feature links three Burn versions (0.20 for SD, 0.21 for whisburn, 0.22 for LLMs). The `image`
feature is on by default.

- Weights load straight from the diffusers-format HF snapshot (`text_encoder/model.safetensors`,
  `unet/…`, `vae/…`, `tokenizer/merges.txt`), using zero extra disk. `pyre models pull
  stable-diffusion-v1-5/stable-diffusion-v1-5` fetches only those files (about 4.3 GB, f32). The repo is not gated.
- Backends: `cuda` (tested) and `wgpu` (built in, not benchmarked). CPU is refused because it would be too slow.
- Lockfile note: `burn-store 0.20.1` doesn't build with `zip >= 7.3`, so `Cargo.lock` pins `zip 7.2.0`.
  `src/main.rs` also sets `#![recursion_limit = "512"]` because of the wgpu 26 + 28 type depth.

```
pyre serve -m qwen --image-model stable-diffusion-v1-5/stable-diffusion-v1-5
curl http://127.0.0.1:8080/v1/images/generations -H "content-type: application/json" \
  -d '{"prompt":"an ancient mossy stone","size":"512x512","steps":20,"seed":42}'
```

`/v1/images/generations` accepts `prompt`, `n` (≤8), `size` (multiples of 8, 64–1024) and
`response_format` (`b64_json`, or `url`, which returns a data: URL), plus the extensions
`negative_prompt`, `steps` (default 30), `guidance_scale` (7.5) and `seed`. It also returns `timings`.
`/sdapi/v1/txt2img` (the AUTOMATIC1111 subset) accepts `prompt`, `negative_prompt`, `steps`,
`cfg_scale`, `width`, `height`, `seed`, `batch_size` and `n_iter`, and returns `images` (base64),
`parameters` and `info`. The only sampler is DDIM.

Measured on an RTX 3060 12 GB (cuda, f32, 512×512, 20 steps): **about 9.4 s per image (2.1 it/s)**
in steady state. The **first image after startup takes about 9 minutes** while CubeCL JIT-compiles and
autotunes the UNet kernels. Later images and runs reuse the cache. Samples are in
`samples/sd_cuda_*.png` and `samples/sdapi_lighthouse.png`.

## Text to speech (stub)

`/v1/audio/speech` returns **501** with a clear message, and `--tts-model` registers a placeholder.
The UI's TTS card stays disabled. I looked at these ports and didn't do any of them yet:

- **Kokoro-82M via burn-onnx** (burn 0.22, Apache weights): tracel's model check compiles the
  2,464-node graph (about 7 minutes of codegen), but its output still diverges from ONNX Runtime
  (Pearson r ≈ 0.69, peaks about 1.3× too large; burn-onnx#371). It also needs a phonemizer
  (espeak-ng is a system library and GPL, which would mean installing something; a pure-Rust G2P
  would be needed instead).
- **A hand port of Kokoro (StyleTTS2 + ISTFTNet)**: about 2–3k lines (BERT/ALBERT, LSTM, AdaIN
  decoders, iSTFT). It's feasible on Burn 0.22 but is a multi-day job.
- **Piper (VITS)**: also needs espeak-ng phonemes.

Plan: Kokoro through burn-onnx once #371 lands, plus a Rust G2P (misaki-style lexicon), behind a
`tts` feature, emitting 24 kHz wav/mp3/opus.

## Roadmap / known gaps

- GGUF and quantized weights (Q4/Q8) are skipped; only safetensors f32/f16/bf16 load.
- Continuous batching: each model has a single worker and a FIFO queue.
- MoE Qwen3 (A3B), Gemma, vision-language and SDXL (diffusers-burn has SDXL; it isn't wired up) are not done.
- TTS (see above).
- Unify Burn versions. SD is on 0.20 (via diffusers-burn), whisburn on 0.21 and pyre on 0.22. Porting
  diffusers-burn and whisburn to 0.22 would drop two cubecl stacks and give STT CUDA support.
- SD warm-up: persist and prewarm the CubeCL autotune cache so the first image isn't about 9 minutes.
- Browser mic recordings (webm/opus) may not decode in whisburn-audio. Uploads of wav/mp3/flac work.
