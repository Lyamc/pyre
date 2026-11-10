<#  pyre-firstrun.ps1 - first-run model selection. No GGUFs are bundled in the installer.
    Options: (1) download the recommended Qwen3.8-Flash-Next IQ2_XS GGUF + tokenizer from Hugging Face into
    models_dir, (2) point at an existing model folder / .gguf / store name / HF repo id, (3) skip.
    Writes [serve] model (and models_dir) into config.toml. Downloads use curl.exe (resumable, -C -);
    set HF_TOKEN for gated repos. Nothing here touches the GPU.
#>
param([string]$Config = (Join-Path $env:ProgramData 'pyre\config.toml'), [switch]$NoUi, [ValidateSet('download', 'existing', 'skip')][string]$Mode, [string]$Path)
$ErrorActionPreference = 'Stop'
$launch = Join-Path $PSScriptRoot 'pyre-launch.ps1'
# reuse Read-Toml / Set-TomlValue from the launcher without running it
$src = Get-Content $launch -Raw
foreach ($fn in 'Read-Toml', 'Set-TomlValue') { $m = [regex]::Match($src, "(?ms)^function $fn.*?^}\r?$"); Invoke-Expression $m.Value }

$GgufRepo = 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'; $Quant = 'IQ2_XS'
$TokRepo = 'Qwen/Qwen3.8-Flash-Next'; $TokFiles = 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'
$cfg = Read-Toml $Config

# Non-blocking GPU/driver notice. CUDA is optional: pyre also runs on Vulkan/wgpu or CPU.
# Shown only when an NVIDIA GPU (PCI vendor 10DE) is present and nvcuda.dll is missing or older than the bundled NVRTC needs.
function Get-DriverNotice {
  $nv = Get-CimInstance Win32_VideoController -ea 0 | Where-Object { $_.PNPDeviceID -match 'VEN_10DE' -or $_.Name -match 'NVIDIA' } | Select-Object -First 1
  if (-not $nv) { return $null }
  $min = [version]'32.0.15.6076'   # driver 560.76 (CUDA 12.6); build-installer.ps1 bundles NVRTC 12.x by default
  $nr = Get-ChildItem $PSScriptRoot -Filter 'nvrtc64_*_0.dll' -ea 0 | Select-Object -First 1
  if ($nr -and $nr.Name -match 'nvrtc64_13') { $min = [version]'32.0.15.8088' }
  $dll = Join-Path $env:SystemRoot 'System32\nvcuda.dll'
  $v = if (Test-Path $dll) { try { [version](Get-Item $dll).VersionInfo.FileVersion.Split(' ')[0] } catch { $null } }
  if ($v -and $v -ge $min) { return $null }
  "Your $($nv.Name) has $(if ($v) { 'an older NVIDIA driver' } else { 'no NVIDIA CUDA driver installed' }). pyre will still run (Vulkan/wgpu or CPU), but for full speed install the latest driver: https://www.nvidia.com/Download/index.aspx"
}
$notice = Get-DriverNotice
if ($notice) { Write-Warning $notice }
$modelsDir = if ($cfg[''].models_dir) { $cfg[''].models_dir } else { Join-Path $env:LOCALAPPDATA 'pyre\models' }

function Get-HF([string]$repo, [string]$file, [string]$dest) {
  New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
  $h = @(); if ($env:HF_TOKEN) { $h = @('-H', "Authorization: Bearer $env:HF_TOKEN") }
  & curl.exe -fL --retry 5 -C - @h -o $dest "https://huggingface.co/$repo/resolve/main/$file"
  if ($LASTEXITCODE) { throw "download failed: $repo/$file (curl exit $LASTEXITCODE)" }
}
function Download-Recommended {
  $h = @{}; if ($env:HF_TOKEN) { $h.Authorization = "Bearer $env:HF_TOKEN" }
  $info = Invoke-RestMethod "https://huggingface.co/api/models/$GgufRepo" -Headers $h
  $files = $info.siblings.rfilename | Where-Object { $_ -like '*.gguf' -and $_ -match $Quant -and $_ -notmatch 'mmproj' }
  if (-not $files) { throw "no $Quant GGUF found in $GgufRepo" }
  $root = Join-Path $modelsDir 'qwen3.8-flash-next-iq2xs'
  foreach ($f in $files) { Write-Host "downloading $f"; Get-HF $GgufRepo $f (Join-Path $root ("gguf\" + ($f -replace '/', '\'))) }
  foreach ($f in $TokFiles) { try { Get-HF $TokRepo $f (Join-Path $root $f) } catch { Write-Warning $_ } }  # pyre finds tokenizer.json in the GGUF's ancestors
  $root
}

if (-not $Mode -and -not $NoUi) {
  Add-Type -AssemblyName System.Windows.Forms
  $f = New-Object Windows.Forms.Form -Property @{ Text = 'pyre - model setup'; Width = 560; Height = $(if ($notice) { 360 } else { 300 }); StartPosition = 'CenterScreen'; FormBorderStyle = 'FixedDialog'; MaximizeBox = $false }
  $lbl = New-Object Windows.Forms.Label -Property @{ Left = 12; Top = 10; Width = 520; Height = 40; Text = "Choose the text model pyre serves. Models are stored in:`n$modelsDir (change models_dir in config.toml)" }
  $r1 = New-Object Windows.Forms.RadioButton -Property @{ Left = 12; Top = 60; Width = 520; Checked = $true; Text = "Download recommended: Qwen3.8-Flash-Next $Quant GGUF (several GB, from Hugging Face)" }
  $r2 = New-Object Windows.Forms.RadioButton -Property @{ Left = 12; Top = 90; Width = 520; Text = 'Use a local folder (GGUF or split shards, for example an IQ3_S directory), store name, or HF repo id:' }
  $tb = New-Object Windows.Forms.TextBox -Property @{ Left = 30; Top = 118; Width = 410 }
  $br = New-Object Windows.Forms.Button -Property @{ Left = 448; Top = 116; Width = 80; Text = 'Browse...' }
  $br.Add_Click({ $d = New-Object Windows.Forms.FolderBrowserDialog; if ($d.ShowDialog() -eq 'OK') { $tb.Text = $d.SelectedPath; $r2.Checked = $true } })
  $r3 = New-Object Windows.Forms.RadioButton -Property @{ Left = 12; Top = 150; Width = 520; Text = 'Skip for now (load models later from the dashboard Models page)' }
  $ok = New-Object Windows.Forms.Button -Property @{ Left = 368; Top = 215; Width = 80; Text = 'OK'; DialogResult = 'OK' }
  $cn = New-Object Windows.Forms.Button -Property @{ Left = 452; Top = 215; Width = 80; Text = 'Cancel'; DialogResult = 'Cancel' }
  if ($notice) {
    $nl = New-Object Windows.Forms.LinkLabel -Property @{ Left = 12; Top = 180; Width = 520; Height = 40; Text = $notice }
    $u = 'https://www.nvidia.com/Download/index.aspx'; $nl.LinkArea = New-Object Windows.Forms.LinkArea($notice.IndexOf($u), $u.Length)
    $nl.Add_LinkClicked({ Start-Process 'https://www.nvidia.com/Download/index.aspx' }); $f.Controls.Add($nl)
    $ok.Top = 275; $cn.Top = 275
  }
  $f.Controls.AddRange(@($lbl, $r1, $r2, $tb, $br, $r3, $ok, $cn)); $f.AcceptButton = $ok; $f.CancelButton = $cn
  if ($f.ShowDialog() -ne 'OK') { return }
  $Mode = if ($r1.Checked) { 'download' } elseif ($r2.Checked) { 'existing' } else { 'skip' }; $Path = $tb.Text.Trim()
}
switch ($Mode) {
  'download' { $m = Download-Recommended; Set-TomlValue $Config 'serve' 'model' $m; Write-Host "model set: $m" }
  'existing' { if (-not $Path) { throw 'no model given' }; Set-TomlValue $Config 'serve' 'model' $Path; Write-Host "model set: $Path" }
  default { Write-Host 'skipped' }
}
