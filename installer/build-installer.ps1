<#  build-installer.ps1 - package ALREADY-BUILT exes into an MSI with WiX v5 (dotnet tool `wix`). Does not run cargo.
    .\installer\build-installer.ps1 -PyreExe <path\pyre.exe> -GuiExe <path\pyre-gui.exe> -CudaMajor 12 [-SourceCommit cf84f1d]
    -CudaMajor picks which NVRTC pair is bundled. It MUST match the CUDA version cudarc baked into pyre.exe
      (cudarc's build.rs runs the first `nvcc` on PATH; see target\...\release\build\cudarc-*\output CUDA_MAJOR_VERSION).
      The script reads that file next to -PyreExe when present and refuses a mismatch.
      12 -> nvrtc64_120_0.dll + nvrtc-builtins64_12x.dll (default dir C:\cuda\bin, else CUDA v12.x\bin)
      13 -> nvrtc64_130_0.dll + nvrtc-builtins64_13x.dll (default dir $env:CUDA_PATH\bin\x64)
    MSVC runtime (vcruntime140, vcruntime140_1, msvcp140) is copied app-local from the VS VC143.CRT redist folder.
#>
param(
  [Parameter(Mandatory)][string]$PyreExe,
  [Parameter(Mandatory)][string]$GuiExe,
  [Parameter(Mandatory)][ValidateSet('12', '13')][string]$CudaMajor,
  [string]$NvrtcDir,
  [string]$CrtDir,
  [string]$SourceCommit,
  [string]$Version,
  [string]$OutDir = (Join-Path $PSScriptRoot 'out')
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot
$wixTools = Join-Path $env:USERPROFILE '.dotnet\tools'
if (-not (Get-Command wix -ea 0) -and (Test-Path (Join-Path $wixTools 'wix.exe'))) { $env:PATH += ";$wixTools" }
if (-not (Get-Command wix -ea 0)) { throw 'wix not found: dotnet tool install --global wix --version 5.0.2' }
if (-not $Version) { $Version = ((Select-String -Path (Join-Path $repo 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value) -replace '[-+].*$', '' }

# --- NVRTC: verify against what cudarc compiled into pyre.exe ---
$cudarcOut = Get-ChildItem (Join-Path (Split-Path $PyreExe) 'build\cudarc-*\output') -ea 0 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$baked = $null
if ($cudarcOut) { $m = Select-String -Path $cudarcOut.FullName -Pattern 'CUDA_MAJOR_VERSION=(\d+)' | Select-Object -First 1; if ($m) { $baked = $m.Matches.Groups[1].Value } }
if ($baked -and $baked -ne $CudaMajor) { throw "pyre.exe was built for CUDA $baked (cudarc: $($cudarcOut.FullName)) but -CudaMajor $CudaMajor was given" }
if (-not $baked) { Write-Warning "could not verify CUDA version baked into pyre.exe (no cudarc build output next to it); trusting -CudaMajor $CudaMajor" }
if (-not $NvrtcDir) {
  $cands = if ($CudaMajor -eq '12') { @('C:\cuda\bin') + (Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin' -ea 0 | Sort-Object FullName -Descending).FullName }
           else { @("$env:CUDA_PATH\bin\x64") + (Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*\bin\x64' -ea 0 | Sort-Object FullName -Descending).FullName }
  $NvrtcDir = $cands | Where-Object { $_ -and (Test-Path (Join-Path $_ "nvrtc64_$($CudaMajor)0_0.dll")) } | Select-Object -First 1
}
$nvrtc = Get-Item (Join-Path $NvrtcDir "nvrtc64_$($CudaMajor)0_0.dll") -ea 0
$nvb = Get-ChildItem $NvrtcDir -Filter "nvrtc-builtins64_$($CudaMajor)*.dll" -ea 0 | Sort-Object Name -Descending | Select-Object -First 1
if (-not $nvrtc -or -not $nvb) { throw "nvrtc64_$($CudaMajor)0_0.dll / nvrtc-builtins64_$($CudaMajor)*.dll not found in '$NvrtcDir'" }
# Driver needed for the bundled CUDA major (nvcuda.dll file version; last 5 digits = driver xxx.yy): 12.6 -> 560.76, 13.0 -> 580.88
$drv = @{ '12' = @('32.0.15.6076', '560.76'); '13' = @('32.0.15.8088', '580.88') }[$CudaMajor]

# --- MSVC runtime, app-local ---
if (-not $CrtDir) { $CrtDir = (Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC14*.CRT' -ea 0 | Sort-Object FullName -Descending | Select-Object -First 1).FullName }
$crt = 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll' | ForEach-Object { Join-Path $CrtDir $_ }
$crt | ForEach-Object { if (-not (Test-Path $_)) { throw "MSVC runtime not found: $_ (pass -CrtDir)" } }

$stage = Join-Path $PSScriptRoot 'stage'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }   # our own staging dir only
New-Item -ItemType Directory -Force $stage, $OutDir | Out-Null
Copy-Item $PyreExe (Join-Path $stage 'pyre.exe'); Copy-Item $GuiExe (Join-Path $stage 'pyre-gui.exe')
Copy-Item $nvrtc.FullName, $nvb.FullName $stage; Copy-Item $crt $stage
foreach ($f in 'pyre-launch.ps1', 'pyre-firstrun.ps1', 'pyre-service.ps1', 'config.template.toml') { Copy-Item (Join-Path $PSScriptRoot $f) $stage }
Copy-Item (Join-Path $repo 'THIRD_PARTY_NOTICES') (Join-Path $stage 'THIRD_PARTY_NOTICES.txt')
$manifest = @("pyre $Version  source commit: $(if ($SourceCommit) { $SourceCommit } else { 'unspecified' })",
  "pyre.exe from: $PyreExe ($((Get-Item $PyreExe).LastWriteTime.ToString('s')))",
  "pyre-gui.exe from: $GuiExe ($((Get-Item $GuiExe).LastWriteTime.ToString('s')))",
  "CUDA major: $CudaMajor (cudarc baked: $(if ($baked) { $baked } else { 'unverified' })); NVRTC from: $NvrtcDir; driver notice below $($drv[1])",
  "MSVC runtime (app-local) from: $CrtDir", '')
$manifest += Get-ChildItem $stage | ForEach-Object { '{0,12}  {1}  {2}' -f $_.Length, (Get-FileHash $_.FullName -Algorithm SHA256).Hash.Substring(0, 16), $_.Name }
Set-Content (Join-Path $stage 'MANIFEST.txt') $manifest

$msi = Join-Path $OutDir "pyre-$Version-x64.msi"
& wix build (Join-Path $PSScriptRoot 'pyre.wxs') -arch x64 -ext WixToolset.Util.wixext -ext WixToolset.UI.wixext `
  -d "StageDir=$stage" -d "Version=$Version" -d "NvrtcDll=$($nvrtc.Name)" -d "NvrtcBuiltinsDll=$($nvb.Name)" `
  -d "DriverMinVersion=$($drv[0])" -d "DriverMinLabel=$($drv[1])" `
  -d "LicenseRtf=$(Join-Path $PSScriptRoot 'license.rtf')" -o $msi 2>&1 | Out-Host
if ($LASTEXITCODE) { throw "wix build failed ($LASTEXITCODE)" }
$manifest | Out-Host
Get-Item $msi | Select-Object FullName, Length
