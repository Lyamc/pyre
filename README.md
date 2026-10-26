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
