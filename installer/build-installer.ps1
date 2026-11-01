<#  build-installer.ps1 - package ALREADY-BUILT exes into an MSI with WiX v5 (dotnet tool `wix`). Does not run cargo.
    .\installer\build-installer.ps1 -PyreExe <path\pyre.exe> -GuiExe <path\pyre-gui.exe> [-NvrtcDir <dir>] [-Version 0.2.0] [-OutDir installer\out]
    Requires: wix 5.x + extensions WixToolset.Util.wixext / WixToolset.UI.wixext (wix extension add -g ... once).
#>
param(
  [Parameter(Mandatory)][string]$PyreExe,
  [Parameter(Mandatory)][string]$GuiExe,
  [string]$NvrtcDir = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin\x64' }),
  [string]$Version,
  [string]$OutDir = (Join-Path $PSScriptRoot 'out')
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot
$wixTools = Join-Path $env:USERPROFILE '.dotnet\tools'
if (-not (Get-Command wix -ea 0) -and (Test-Path (Join-Path $wixTools 'wix.exe'))) { $env:PATH += ";$wixTools" }
if (-not (Get-Command wix -ea 0)) { throw 'wix not found: dotnet tool install --global wix --version 5.0.2' }
if (-not $Version) { $Version = ((Select-String -Path (Join-Path $repo 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value) -replace '[-+].*$', '' }

# NVRTC: cubecl-cuda compiles kernels at runtime via cudarc (dynamic loading). Bundle the redistributable nvrtc64_*.dll
# + nvrtc-builtins64_*.dll next to pyre.exe so users need only the NVIDIA driver (nvcuda.dll), not the CUDA toolkit.
$nvrtc = Get-ChildItem $NvrtcDir -Filter 'nvrtc64_*.dll' | Sort-Object Name -Descending | Select-Object -First 1
$nvb = Get-ChildItem $NvrtcDir -Filter 'nvrtc-builtins64_*.dll' | Sort-Object Name -Descending | Select-Object -First 1
if (-not $nvrtc -or -not $nvb) { throw "nvrtc DLLs not found in $NvrtcDir" }

$stage = Join-Path $PSScriptRoot 'stage'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }   # our own staging dir only
New-Item -ItemType Directory -Force $stage, $OutDir | Out-Null
Copy-Item $PyreExe (Join-Path $stage 'pyre.exe'); Copy-Item $GuiExe (Join-Path $stage 'pyre-gui.exe')
Copy-Item $nvrtc.FullName, $nvb.FullName $stage
foreach ($f in 'pyre-launch.ps1', 'pyre-firstrun.ps1', 'pyre-service.ps1', 'config.template.toml') { Copy-Item (Join-Path $PSScriptRoot $f) $stage }
Copy-Item (Join-Path $repo 'THIRD_PARTY_NOTICES') (Join-Path $stage 'THIRD_PARTY_NOTICES.txt')
$manifest = Get-ChildItem $stage | ForEach-Object { '{0,12}  {1}  {2}' -f $_.Length, (Get-FileHash $_.FullName -Algorithm SHA256).Hash.Substring(0, 16), $_.Name }
$manifest += "pyre.exe from: $PyreExe ($((Get-Item $PyreExe).LastWriteTime))", "pyre-gui.exe from: $GuiExe ($((Get-Item $GuiExe).LastWriteTime))", "nvrtc from: $NvrtcDir"
Set-Content (Join-Path $stage 'MANIFEST.txt') $manifest

$msi = Join-Path $OutDir "pyre-$Version-x64.msi"
& wix build (Join-Path $PSScriptRoot 'pyre.wxs') -arch x64 -ext WixToolset.Util.wixext -ext WixToolset.UI.wixext `
  -d "StageDir=$stage" -d "Version=$Version" -d "NvrtcDll=$($nvrtc.Name)" -d "NvrtcBuiltinsDll=$($nvb.Name)" `
  -d "LicenseRtf=$(Join-Path $PSScriptRoot 'license.rtf')" -o $msi 2>&1 | Out-Host
if ($LASTEXITCODE) { throw "wix build failed ($LASTEXITCODE)" }
Get-Item $msi | Select-Object FullName, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }
