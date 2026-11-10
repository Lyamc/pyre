# pyre

A llama.cpp `llama-server`-style inference server in Rust. [Burn](https://github.com/tracel-ai/burn) (Lyamc/burn) runs the model: matmul, attention, and norms are Burn tensor ops on CUDA, wgpu, or CPU. There is no Python or ONNX runtime on the launch or inference path.

## Launch

On Windows, from a clone or from an install, `pyre.ps1` is the way to start and stop the server. It calls `installer/pyre-launch.ps1`. An MSI is not required.

```powershell
git clone --recurse-submodules https://github.com/Lyamc/pyre.git
cd pyre
git checkout qwen-int
git submodule update --init --recursive
cargo build --release

# Local GGUF directory (one file, or split shards such as *-00001-of-*.gguf).
# The path below is an example. Use your own directory.
.\pyre.ps1 -Model D:\models\Qwen3.8-Flash-Next-IQ3_S -Quant IQ3_S -Foreground
```

That listens on `0.0.0.0:8099` (every interface) and serves the web UI at `http://127.0.0.1:8099/` on this machine. `-Quant IQ3_S` sets `PYRE_QN_QUANT` for this process. Set it when the directory tree contains more than one quant; a folder that only contains IQ3_S shards is found without it. `-Save` writes the model and quant into the config file. Without an API key, anything that can reach the port can call the server. `.\pyre.ps1 -Host 127.0.0.1` (or `host = '127.0.0.1'` in the config) keeps it on this machine.

```powershell
.\pyre.ps1 -Stop      # graceful stop, then kill if it does not exit
.\pyre.ps1 -DryRun    # print the command and environment; starts nothing
.\pyre.ps1 -Watch     # restart after a crash (a stop does not restart)
.\pyre.ps1 -Web       # start if needed, then open the web UI
```

The same command the script runs, on any OS:

```text
pyre serve -m D:\models\Qwen3.8-Flash-Next-IQ3_S --port 8099 -b cuda
pyre serve -m D:\models\Qwen3.8-Flash-Next-IQ3_S --host 127.0.0.1 --port 8099
```

`PYRE_QN_QUANT=IQ3_S` when you need to pick a shard. On Linux or macOS, `./scripts/pyre.sh -m /path/to/IQ3_S --quant IQ3_S` is a thin wrapper around that command (no config file). `--stop` sends SIGTERM.

If `pyre.exe` sits next to `installer/pyre-launch.ps1` (an install under `C:\Program Files\pyre`), the script uses `%ProgramData%\pyre\config.toml` and starts the server hidden, which is what the Start-menu shortcuts do. From a checkout it uses the release binary under `target\` and `%LOCALAPPDATA%\pyre\config.toml`. The first run copies `installer/config.template.toml`.

A script you already run outside this repo, such as `C:\Temp\qnx\int\srv.ps1`, is unchanged. `pyre.ps1 -Watch` is the in-repo restart loop if you want one.

### Where the model comes from

`-m` / `[serve] model` is a folder, a `.gguf` shard, a pyre-store name, or a Hugging Face id `org/name`. For a split GGUF, point it at the directory; pyre opens the first shard (`-00001-of-` when the file is split). `PYRE_QN_QUANT` (for example `IQ3_S`) selects a shard whose path contains that text when several quants are present.

Qwen-Next also needs `tokenizer.json` and the chat template. Pyre looks next to the GGUF (up to three parent directories), then `PYRE_QN_TOKENIZER`, then the Hugging Face cache for the base repo named in the file (otherwise `Qwen/Qwen3.8-Flash-Next`).

With no `-Model` and an empty `model` in the config, the launcher opens `installer/pyre-firstrun.ps1`: download the IQ2_XS GGUF of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, point at a folder you already have, or skip. Pass `-Model` or `-NoFirstRun` to skip that dialog. Nothing in the installer bundles GGUF weights.

### Stop

`.\pyre.ps1 -Stop` writes `%TEMP%\pyre\stop.<pid>` (and the same file under the SYSTEM temp directories, for a scheduled-task server). Pyre drains GPU work at the next step and exits 87. The script waits `PYRE_STOP_GRACE_S` (default 10) plus 5 seconds, then kills the pid. It also writes `pyre-watch.stop` next to the config so a `-Watch` loop does not start the server again.

On Unix, `./scripts/pyre.sh --stop` signals the pid it recorded.

### Config, port, context

| | |
|---|---|
| Launcher config | Install: `%ProgramData%\pyre\config.toml`. Checkout: `%LOCALAPPDATA%\pyre\config.toml`. Override with `-Config` or `PYRE_CONFIG`. The launcher sets `PYRE_CONFIG` for the process. |
| `pyre` itself, if you do not set `PYRE_CONFIG` | `%APPDATA%\pyre\config.toml` on Windows, `~/.config/pyre/config.toml` elsewhere. Top-level keys only: `models_dir`, `offline`, `search_order`, `spec`, `route_prior`. |
| Port | Launcher and the config template use **8099**. `pyre serve` with no `--port` uses **8080**. |
| Host | Default **`0.0.0.0`** (all interfaces), in `pyre serve`, a new config, `pyre.ps1`, and `scripts/pyre.sh`. `--host 127.0.0.1`, `-Host 127.0.0.1`, or `PYRE_HOST=127.0.0.1` is this machine only. A config that already sets `host` is left as it is. Open the firewall yourself. With no API key the server logs a warning. |
| Backend | `-b cuda` is pyre's default. `wgpu`, `vulkan`, `cpu`, `ndarray` are the other names. The launcher passes `-b` only when `[serve] backend` is set. |
| Qwen-Next context | `PYRE_QN_CTX` (`8k` … `512k`, or a token count) or `%LOCALAPPDATA%\pyre\qwen-context.json`. Default 131072. `-c` / `ctx_size` applies to other models, not this one. |
| KV cache | 4-bit by default. `PYRE_QN_KV_BITS=8` selects 8-bit. `PYRE_QN_KV8=0` keeps f32. |

`[serve]` in the template maps to `pyre serve` flags (`model`, `host`, `port`, `ctx_size`, `n_predict`, `backend`, image/stt/tts models, `vram_reserve`, `gpu_util_max`) and to `PYRE_QN_FAST`, `PYRE_VRAM_FLOOR_MB`, `PYRE_API_KEY`, `PYRE_LOG_FILE`. `[env]` is passed through. A new config sets `PYRE_QN_QUANT` to `IQ3_S`. A config that already exists is left alone.

### VRAM watchdog

`PYRE_VRAM_FLOOR_MB` defaults to 1100 (`0` turns it off). A watcher polls free VRAM. Under the floor it shrinks the expert cache when `PYRE_VRAM_ADAPT` is on (the default), then asks for a graceful stop (exit 87). If free VRAM is still under the floor it exits 86. `PYRE_GPU_DUTY` defaults to 85 and paces steps so a desktop session stays usable (`off`, `0`, or `100` disables it). `PYRE_STOP_GRACE_S` is how long a stop request waits for a step boundary.

Logs for a hidden launch: `<config-dir>\logs\pyre-stdout.log` and `pyre-stderr.log`. The template also sets `PYRE_LOG_FILE` to a jsonl file next to that, or under `%ProgramData%\pyre\logs` for an install.

## Build

```powershell
cargo build --release
cargo build --release --no-default-features --features wgpu
cargo test --release
```

Submodules: `vendor/burn-pyre` (Lyamc/burn, `pyre-fused-quant`), `vendor/burn-gguf` (path dependency), `vendor/cubecl-pipe`. Cargo pins Burn and CubeCL to git revisions that match those submodules. `rust-toolchain.toml` pins `stable-x86_64-pc-windows-msvc`, and `.cargo/config.toml` pins the MSVC linker. Build scripts have to be the msvc host. The CUDA backend JIT-compiles through CubeCL's bundled LLVM, so `nvcc` is not required at build time. An NVIDIA driver is required to run on CUDA.

## Desktop GUI, tray, MSI

`gui/` is a separate workspace (`pyre-gui`, and `pyre-chat`). It talks to the server over HTTP. Build it with `cargo build --release --manifest-path gui/Cargo.toml`. `.\pyre.ps1 -Gui` starts `pyre-gui.exe` only when that file is next to the launcher or at `gui\target\x86_64-pc-windows-msvc\release\pyre-gui.exe`.

`installer/` contains WiX source, `pyre-tray`, and `build-installer.ps1`, which packages exes you have already built. This repository does not build or install the MSI as part of the launch path above. Treat the tray and the Start-menu entries as available after you have produced an MSI with that script. Details: [installer/README.md](installer/README.md).

`installer/pyre-service.ps1 -Install` (elevated, optional, not run by the MSI) registers a boot-time scheduled task named `pyre-server`.

## More

Longer notes on GGUF, Qwen-Next, speech, and benchmarks are in [docs/development-log.md](docs/development-log.md). Streaming resume after a crash or a watchdog exit is in [docs/streaming-resume.md](docs/streaming-resume.md).
