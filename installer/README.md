# pyre Windows installer (MSI, WiX v5)

```powershell
# Exes must already be built via C:\Temp\qnx\build.ps1: once in the repo root (pyre) and once in gui\ (pyre-gui, own workspace).
.\installer\build-installer.ps1 -CudaMajor 12 -SourceCommit cf84f1d `
   -PyreExe target\x86_64-pc-windows-msvc\release\pyre.exe `
   -GuiExe  gui\target\x86_64-pc-windows-msvc\release\pyre-gui.exe
# -> installer\out\pyre-<ver>-x64.msi   (stage\ and out\ are git-ignored; stage\MANIFEST.txt records sources and hashes)
```

## CUDA is optional
pyre picks its backend (cuda / wgpu / vulkan / cpu), so nothing blocks the install. The MSI and `pyre-firstrun.ps1` only show a notice, with a link to
https://www.nvidia.com/Download/index.aspx, when an NVIDIA GPU (PCI vendor 10DE) is present and `nvcuda.dll` is missing or older than the driver needed for the
bundled NVRTC. That is driver 560.76 for CUDA 12.6 and 580.88 for CUDA 13. Machines without an NVIDIA GPU see no warning.

## NVRTC: `-CudaMajor` must match the exe
cudarc 0.19.10 (`cuda-version-from-build-system`) runs the first `nvcc` on PATH during the build. On HYTE that is `C:\cuda\bin\nvcc` (12.6), so it bakes in
CUDA_MAJOR/MINOR = 12/6, even though `CUDA_PATH` is v13.3. At runtime it tries, in this order: `nvrtc.dll`, `nvrtc64.dll`, `nvrtc64_12.dll`, `nvrtc64_126.dll`,
`nvrtc64_126_0.dll`, `nvrtc64_120_6.dll`, `nvrtc64_10/11/12.dll`, then `nvrtc64_120_0.dll`. The last name is the real 12.x file. `nvrtc64_130_0.dll` is never tried
for a 12.6 build, so a 12.6 build needs the 12.x pair (`nvrtc64_120_0.dll` + `nvrtc-builtins64_126.dll`). `build-installer.ps1` reads
`release\build\cudarc-*\output` next to `-PyreExe` and refuses a mismatched `-CudaMajor`. To ship CUDA 13, put 13.3's nvcc first on PATH (or set
`CUDARC_CUDA_VERSION=13030`) for the build, then pass `-CudaMajor 13`.

**Confirmed at runtime** (read-only module check by Coder on the running `C:\Temp\qnx\int\pyre_int45c.exe`): it loads `C:\cuda\bin\nvrtc64_120_0.dll` +
`nvrtc-builtins64_126.dll`, and MSVCP140/VCRUNTIME140/_1 from System32. That matches the 12.x pair and the app-local runtime the MSI ships.

## Newer-build options and graceful stop (int45+, newer than cf84f1d)
The `[env]` section of the template has commented `PYRE_GPU_DUTY`, `PYRE_GPU_DUTY_MIN`, `PYRE_VRAM_ADAPT` and `PYRE_STOP_GRACE_S` (semantics are taken from `src/limits.rs`).
"Stop pyre server" first creates `%TEMP%\pyre\stop.<pid>` (pyre drains the GPU at the next step and exits 87). It waits up to the grace period + 5 s and then kills the pid.
Builds older than int45 ignore the stop file, so they are killed.

## What gets installed
`C:\Program Files\pyre` (added to PATH) gets:
- `pyre.exe` and `pyre-gui.exe` (the GUI is built `+crt-static`).
- The NVRTC pair.
- The MSVC runtime DLLs (`vcruntime140`, `vcruntime140_1`, `msvcp140`) app-local, because pyre.exe can't be crt-static: the prebuilt LLVM in cubecl-llvm is /MD. So no VC++ redist prereq.
- The launcher, first-run, and service scripts, `config.template.toml`, and `MANIFEST.txt`.

Start-menu entries: Dashboard, Web Chat, Start/Stop server, Model Setup, Edit config, Uninstall. There is also an optional "Start server at logon" feature, off by default.

## Config
`%ProgramData%\pyre\config.toml` (or set `PYRE_CONFIG`). The launcher creates it from the template the first time it runs.
- **Top-level keys** (`models_dir`, `offline`, `search_order`, `spec`, `route_prior`) are read by pyre itself.
- **`[serve]`** is mapped by `pyre-launch.ps1` to flags: `--host --port -m -b -c -n --image-model --stt-model --tts-model --vram-reserve --gpu-util-max`.
  It also sets these env vars: `PYRE_QN_FAST`, `PYRE_VRAM_FLOOR_MB`, `PYRE_API_KEY`, `PYRE_METRICS_BIND`, `PYRE_METRICS_TOKEN`, `PYRE_LOG_FILE`, `PYRE_INSTANCE`.
  `-b` is only passed when `backend` is set.
- **`[env]`** passes any other variable through.

To see the resolved command line without starting anything, run `pyre-launch.ps1 -DryRun`.

## First run
If `[serve] model` is empty, the launcher opens `pyre-firstrun.ps1`, which offers three choices:
1. Download the IQ2_XS shards of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` plus the tokenizer from `Qwen/Qwen3.8-Flash-Next` into `models_dir`
   (curl, resumable; set `HF_TOKEN` for gated repos).
2. Use an existing folder, `.gguf` file, store name, or HF repo id.
3. Skip.

## Background / service
- **Default:** the server starts hidden for the current user, from the Start-menu entry or the logon feature.
- **Boot-time option:** `pyre-service.ps1 -Install` (run by hand, elevated) registers a boot-time SYSTEM Scheduled Task. The installer never registers it, so run `-Uninstall` before you remove pyre.

## Upgrade / uninstall
- **Upgrade:** MajorUpgrade with a fixed UpgradeCode.
- **Uninstall:** the MSI never owns config, logs, or models (`%ProgramData%\pyre`, `%LOCALAPPDATA%\pyre`, the HF cache), so they survive upgrade and uninstall.
- **Opt-in purge:** `msiexec /x pyre.msi PURGEDATA=1` deletes only `%ProgramData%\pyre`.
