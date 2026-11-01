# pyre Windows installer (MSI, WiX v5)

```powershell
# exes must already be built (via C:\Temp\qnx\build.ps1 for pyre; gui\ is built separately)
.\installer\build-installer.ps1 -PyreExe target\x86_64-pc-windows-msvc\release\pyre.exe `
                                -GuiExe  gui\target\x86_64-pc-windows-msvc\release\pyre-gui.exe
# -> installer\out\pyre-<ver>-x64.msi   (stage\ and out\ are git-ignored)
```

**What gets installed** to `C:\Program Files\pyre` (added to PATH): `pyre.exe`, `pyre-gui.exe`, the NVRTC DLLs (`nvrtc64_*.dll`, `nvrtc-builtins64_*.dll`,
redistributable under the CUDA EULA; cubecl compiles kernels at runtime), `pyre-launch.ps1`, `pyre-firstrun.ps1`, `pyre-service.ps1`, `config.template.toml`.
You also get Start-menu entries for Dashboard, Web Chat, Start/Stop server, Model Setup, Edit config, and Uninstall. You can also pick the optional "Start server at logon" feature (off by default).

**Prereqs (launch conditions):** the NVIDIA driver must support CUDA 13 (`nvcuda.dll` >= 32.0.15.8000, which is driver 580). Bypass this check with `SKIPCUDACHECK=1`.
The VC++ 2015-2022 x64 runtime is also required. Bypass this check with `SKIPVCCHECK=1`. The installer doesn't bundle the CUDA toolkit, cudart, or cuBLAS. pyre.exe imports none of them.

**Config:** `%ProgramData%\pyre\config.toml` (or set `PYRE_CONFIG`). The launcher creates it from the template the first time it runs.
pyre reads the top-level keys itself (`models_dir`, `offline`, `search_order`, `spec`, `route_prior`).
`pyre-launch.ps1` maps `[serve]` to `serve` flags and env vars: `--host --port -m -b -c -n --image-model --stt-model --tts-model --vram-reserve --gpu-util-max`,
`PYRE_QN_FAST`, `PYRE_VRAM_FLOOR_MB`, `PYRE_API_KEY`, `PYRE_METRICS_BIND`, `PYRE_METRICS_TOKEN`, `PYRE_LOG_FILE`, `PYRE_INSTANCE`.
`[env]` passes any other variable through. To see the resolved command line without starting anything, run `pyre-launch.ps1 -DryRun`.

**First run:** if `[serve] model` is empty, the launcher opens `pyre-firstrun.ps1`. You can download the IQ2_XS shards of
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` plus the tokenizer from `Qwen/Qwen3.8-Flash-Next` into `models_dir`. You can also point it at an existing
folder, `.gguf` file, store name, or HF repo id, or skip. Use `HF_TOKEN` for gated repos. After that, manage models from the dashboard's Models page.

**Background / service:** the default is per-user hidden start (Start-menu entry or the logon feature). `pyre-service.ps1 -Install` (run elevated, by hand)
registers a boot-time SYSTEM Scheduled Task. The installer never registers it. Run `-Uninstall` before you remove pyre.

**Upgrade/uninstall:** MajorUpgrade with a fixed UpgradeCode. The MSI doesn't own the config, logs, or models (they live in `%ProgramData%\pyre`, `%LOCALAPPDATA%\pyre`, or the HF cache), so upgrade and uninstall keep them.
`msiexec /x pyre.msi PURGEDATA=1` deletes `%ProgramData%\pyre`. Model folders and the HF cache elsewhere are never deleted.
