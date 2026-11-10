<#  pyre-launch.ps1 - start or stop `pyre serve` from config.toml.

    From a git checkout (after cargo build --release):
      ..\pyre.ps1 -Model D:\models\Qwen3.8-Flash-Next-IQ3_S -Quant IQ3_S
    From an install (pyre.exe next to this script): the Start-menu shortcuts call this file.
    Config: $env:PYRE_CONFIG, else %ProgramData%\pyre\config.toml when installed,
    else %LOCALAPPDATA%\pyre\config.toml. The process is started with PYRE_CONFIG set to that file.

    -Model PATH   -m for this run (folder of .gguf shards, one .gguf, store name, or HF repo id)
    -Quant NAME   PYRE_QN_QUANT for this run (IQ3_S picks that shard when several are present)
    -Save         write -Model / -Quant into the config file
    -Port N       override [serve] port
    -Foreground   server logs in this window (no redirect)
    -Watch        restart after a crash. A graceful stop (exit 87) or -Stop does not restart.
    -Gui          start if needed, then open pyre-gui.exe at http://127.0.0.1:<port>
    -Web          start if needed, then open the web chat in the browser
    -Stop         graceful stop (stop file), then kill the pid this launcher recorded
    -FirstRun     open model setup (pyre-firstrun.ps1)
    -NoFirstRun   never open model setup (autostart / service / tray)
    -EditConfig   open config.toml in Notepad
    -NoTray       do not start pyre-tray.exe
    -DryRun       print the resolved command line and env, then exit
#>
param(
  [switch]$Gui, [switch]$Web, [switch]$Stop, [switch]$FirstRun, [switch]$NoFirstRun,
  [switch]$EditConfig, [switch]$DryRun, [switch]$NoTray, [string]$Config,
  [string]$Model, [string]$Quant, [switch]$Save, [int]$Port = 0,
  [switch]$Foreground, [switch]$Watch
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = Split-Path $here -Parent
if (-not (Test-Path -LiteralPath (Join-Path $repo 'Cargo.toml'))) { $repo = $here }
$installed = Test-Path -LiteralPath (Join-Path $here 'pyre.exe')
if (-not $Config) {
  if ($env:PYRE_CONFIG) { $Config = $env:PYRE_CONFIG }
  elseif ($installed -and $env:ProgramData) { $Config = Join-Path $env:ProgramData 'pyre\config.toml' }
  else {
    $base = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } elseif ($env:HOME) { Join-Path $env:HOME '.config' } else { $here }
    $Config = Join-Path $base 'pyre\config.toml'
  }
}
$data = Split-Path -Parent $Config

function Read-Toml([string]$path) {   # minimal: [section], key = 'str' | "str" | number | bool, # comments
  $t = @{ '' = @{} }; $sec = ''
  foreach ($raw in Get-Content -LiteralPath $path) {
    $l = $raw.Trim(); if (-not $l -or $l.StartsWith('#')) { continue }
    if ($l -match '^\[([^\]]+)\]') { $sec = $Matches[1].Trim(); if (-not $t[$sec]) { $t[$sec] = @{} }; continue }
    if ($l -match "^([A-Za-z0-9_.-]+)\s*=\s*'([^']*)'") { $t[$sec][$Matches[1]] = $Matches[2]; continue }
    if ($l -match '^([A-Za-z0-9_.-]+)\s*=\s*"((?:[^"\\]|\\.)*)"') { $t[$sec][$Matches[1]] = [regex]::Unescape($Matches[2]); continue }
    if ($l -match '^([A-Za-z0-9_.-]+)\s*=\s*([^#\s]+)') { $t[$sec][$Matches[1]] = $Matches[2] }
  }
  $t
}
function Set-TomlValue([string]$path, [string]$section, [string]$key, [string]$value) {  # literal-string set, keeps comments
  $lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $path); $sec = ''; $done = $false
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $l = $lines[$i].Trim()
    if ($l -match '^\[([^\]]+)\]') { if ($sec -eq $section -and -not $done) { $lines.Insert($i, "$key = '$value'"); $done = $true; break }; $sec = $Matches[1].Trim(); continue }
    if ($sec -eq $section -and $l -match "^$([regex]::Escape($key))\s*=") {
      $c = if ($lines[$i] -match "(\s+#.*)$") { $Matches[1] } else { '' }
      $lines[$i] = "$key = '$value'$c"; $done = $true; break }
  }
  if (-not $done) { if ($sec -ne $section -and $section) { $lines.Add("[$section]") }; $lines.Add("$key = '$value'") }
  Set-Content -LiteralPath $path -Value $lines -Encoding UTF8
}

function Find-PyreExe {
  $cands = @(
    (Join-Path $here 'pyre.exe'),
    (Join-Path $here 'pyre'),
    (Join-Path $repo 'target\release\pyre.exe'),
    (Join-Path $repo 'target\release\pyre'),
    (Join-Path $repo 'target\x86_64-pc-windows-msvc\release\pyre.exe'),
    (Join-Path $repo 'target\x86_64-pc-windows-msvc\release\pyre'),
    (Join-Path $repo 'target\debug\pyre.exe'),
    (Join-Path $repo 'target\debug\pyre'),
    (Join-Path $repo 'target\x86_64-pc-windows-msvc\debug\pyre.exe')
  )
  foreach ($c in $cands) { if (Test-Path -LiteralPath $c) { return (Resolve-Path -LiteralPath $c).Path } }
  $cmd = Get-Command pyre -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return $cmd.Source }
  $null
}

if (-not (Test-Path -LiteralPath $Config)) {
  New-Item -ItemType Directory -Force $data | Out-Null
  Copy-Item (Join-Path $here 'config.template.toml') $Config
  if (-not $installed) {
    # The template's models_dir and log path are the shared install location.
    if ($env:LOCALAPPDATA) { Set-TomlValue $Config '' 'models_dir' (Join-Path $env:LOCALAPPDATA 'pyre\models') }
    Set-TomlValue $Config 'serve' 'log_file' (Join-Path $data 'logs\pyre.jsonl')
  }
}

if ($EditConfig) { Start-Process notepad.exe $Config; return }
if ($Save) {
  if ($Model) { Set-TomlValue $Config 'serve' 'model' $Model }
  if ($Quant) { Set-TomlValue $Config 'env' 'PYRE_QN_QUANT' $Quant }
}
$cfg = Read-Toml $Config; $s = $cfg['serve']; if (-not $s) { $s = @{} }
function V($k, $d = '') { if ($s.ContainsKey($k) -and "$($s[$k])" -ne '') { "$($s[$k])" } else { $d } }
$port = if ($Port) { $Port } else { [int](V 'port' '8099') }
$hostIp = V 'host' '127.0.0.1'
$pidFile = Join-Path $data 'pyre-server.pid'
$watchStop = Join-Path $data 'pyre-watch.stop'
$exe = Find-PyreExe
$modelArg = if ($Model) { $Model } else { V 'model' }

if ($Stop) {
  # Tell a -Watch loop in this data dir to exit, then ask pyre to drain.
  # Graceful stop (builds newer than cf84f1d): pyre polls <its %TEMP%>\pyre\stop.<pid>
  # and exits 87. Older builds ignore the file. A SYSTEM task has a different %TEMP%.
  New-Item -ItemType Directory -Force $data | Out-Null
  Set-Content -LiteralPath $watchStop 'stop'
  $p = Get-Content $pidFile -ea 0
  if ($p -and $exe) {
    $proc = Get-Process -Id ([int]$p) -ea 0
    if ($proc -and ($proc.Path -eq $exe -or (-not $proc.Path -and $proc.ProcessName -eq 'pyre'))) {
      $sf = Join-Path ([IO.Path]::GetTempPath()) "pyre\stop.$($proc.Id)"
      New-Item -ItemType Directory -Force (Split-Path $sf) | Out-Null; Set-Content $sf 'stop'
      foreach ($st in "$env:SystemRoot\SystemTemp", "$env:SystemRoot\Temp") { try { New-Item -ItemType Directory -Force "$st\pyre" -ea Stop | Out-Null; Set-Content "$st\pyre\stop.$($proc.Id)" 'stop' -ea Stop } catch {} }
      $grace = 10; if ($cfg['env'] -and $cfg['env'].PYRE_STOP_GRACE_S) { $grace = [int]$cfg['env'].PYRE_STOP_GRACE_S }
      if (-not $proc.WaitForExit(($grace + 5) * 1000)) { Stop-Process -Id $proc.Id -Force -ea 0 }
      Remove-Item $sf -ea 0
    }
  }
  Remove-Item $pidFile -ea 0
  Write-Host "stop requested ($watchStop)"
  return
}
if ($FirstRun -or (-not $NoFirstRun -and -not $DryRun -and -not $modelArg)) {
  & (Join-Path $here 'pyre-firstrun.ps1') -Config $Config
  $cfg = Read-Toml $Config; $s = $cfg['serve']
  if (-not $Model) { $modelArg = V 'model' }
  if ($FirstRun -and -not $Gui -and -not $Web) { return }
}

$a = @('serve', '--host', $hostIp, '--port', "$port", '-c', (V 'ctx_size' '4096'), '-n', (V 'n_predict' '512'))
if (V 'backend') { $a += @('-b', (V 'backend')) }
if ($modelArg) { $a += @('-m', $modelArg) }
foreach ($m in @(@('image_model', '--image-model'), @('stt_model', '--stt-model'), @('tts_model', '--tts-model'), @('vram_reserve', '--vram-reserve'))) { if (V $m[0]) { $a += @($m[1], (V $m[0])) } }
if ((V 'gpu_util_max' '0') -ne '0') { $a += @('--gpu-util-max', (V 'gpu_util_max')) }
if (V 'extra_args') { $a += (V 'extra_args') -split '\s+' }

$envs = [ordered]@{ PYRE_CONFIG = $Config }
$envs.PYRE_QN_FAST = if ((V 'fast' 'true') -match '^(false|0|off)$') { '0' } else { '1' }
foreach ($m in @(@('vram_floor_mb', 'PYRE_VRAM_FLOOR_MB'), @('api_key', 'PYRE_API_KEY'), @('metrics_bind', 'PYRE_METRICS_BIND'), @('metrics_token', 'PYRE_METRICS_TOKEN'), @('log_file', 'PYRE_LOG_FILE'), @('instance', 'PYRE_INSTANCE'))) { if (V $m[0]) { $envs[$m[1]] = V $m[0] } }
if ($cfg['env']) { foreach ($k in $cfg['env'].Keys) { $envs[$k] = $cfg['env'][$k] } }
if ($Quant) { $envs['PYRE_QN_QUANT'] = $Quant }
if ($envs.PYRE_LOG_FILE -and $envs.PYRE_LOG_FILE -notin 'off', '0') { New-Item -ItemType Directory -Force (Split-Path $envs.PYRE_LOG_FILE) -ea 0 | Out-Null }

$url = "http://127.0.0.1:$port"
Write-Host "pyre: $url"
if ($modelArg) { Write-Host "model: $modelArg" }
Write-Host "config: $Config"
Write-Host "stop: .\pyre.ps1 -Stop"
if ($DryRun) {
  "config : $Config"
  "exe    : $(if ($exe) { $exe } else { '(not found - cargo build --release)' })"
  "args   : $($a -join ' ')"
  foreach ($k in $envs.Keys) { $v = if ($k -match 'KEY|TOKEN') { if ($envs[$k]) { '<set>' } else { '' } } else { $envs[$k] }; "env    : $k=$v" }
  "url    : $url"
  if ($Watch) { "watch  : restart unless exit 87 or -Stop" }
  return
}
if (-not $exe) {
  throw "pyre executable not found. From the repo: cargo build --release`nThen run .\pyre.ps1 again. An install looks for pyre.exe next to this script."
}

function Test-Port { try { $c = [Net.Sockets.TcpClient]::new(); $ok = $c.ConnectAsync('127.0.0.1', $port).Wait(500); $c.Dispose(); $ok } catch { $false } }
function Start-PyreOnce {
  foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
  $out = Join-Path $data 'logs'; New-Item -ItemType Directory -Force $out | Out-Null
  # Windows PowerShell 5.1 joins an argument array and re-splits on spaces, which
  # breaks a model path. Pass one quoted command line there. PowerShell 7 keeps the array.
  $argList = if ($PSVersionTable.PSVersion.Major -ge 6) { $a } else {
    ($a | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
  }
  $sp = @{ FilePath = $exe; ArgumentList = $argList; WorkingDirectory = $here; PassThru = $true }
  if ($Foreground) { $sp.NoNewWindow = $true } else {
    $sp.WindowStyle = 'Hidden'
    $sp.RedirectStandardError = Join-Path $out 'pyre-stderr.log'
    $sp.RedirectStandardOutput = Join-Path $out 'pyre-stdout.log'
  }
  # Stay in this window until the server exits, except when we still have to open a UI.
  if ($Foreground -and ($Watch -or (-not $Gui -and -not $Web))) { $sp.Wait = $true }
  $p = Start-Process @sp
  Set-Content -LiteralPath $pidFile $p.Id
  $p
}

if (-not $Foreground) { Write-Host "logs: $(Join-Path $data 'logs')" }
$tray = Join-Path $here 'pyre-tray.exe'
if (-not $NoTray -and (Test-Path -LiteralPath $tray)) { Start-Process $tray }

if ($Watch) { Remove-Item $watchStop -ea 0 }
$started = $false
do {
  if ($Watch -and (Test-Path -LiteralPath $watchStop)) { break }
  if (-not (Test-Port)) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-PyreOnce
    $started = $true
    if (-not $Watch) { break }
    if (-not $Foreground) { $null = $p.WaitForExit() }
    $code = $p.ExitCode
    if ($code -eq 87 -or $code -eq -1073741510 -or (Test-Path -LiteralPath $watchStop)) { break }
    if ($sw.Elapsed.TotalSeconds -lt 5) { throw "pyre exited $code in under 5s; not restarting. Logs: $(Join-Path $data 'logs')" }
    Write-Host "pyre exited $code; restarting in 2s (.\pyre.ps1 -Stop to end)"
    Start-Sleep -Seconds 2
  } elseif ($Watch) {
    while ((Test-Port) -and -not (Test-Path -LiteralPath $watchStop)) { Start-Sleep -Seconds 2 }
  } else { break }
} while ($Watch)

if ($Gui -or $Web) {
  for ($i = 0; $i -lt 240 -and -not (Test-Port); $i++) { Start-Sleep -Milliseconds 500 }
  if ($Gui) {
    $g = Join-Path $here 'pyre-gui.exe'
    if (-not (Test-Path -LiteralPath $g)) { $g = Join-Path $repo 'gui\target\x86_64-pc-windows-msvc\release\pyre-gui.exe' }
    if (Test-Path -LiteralPath $g) { Start-Process $g -ArgumentList $url } else { Write-Warning "pyre-gui.exe not found. Web UI: $url" }
  }
  if ($Web) { Start-Process $url }
}
