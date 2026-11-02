<#  pyre-launch.ps1 - reads %ProgramData%\pyre\config.toml (or $env:PYRE_CONFIG) and starts `pyre serve` hidden.
    -Gui         start server if not running, then open pyre-gui.exe at http://127.0.0.1:<port>
    -Web         same, but open the web chat UI in the browser
    -Stop        stop the pyre.exe started from THIS install folder (by pid file)
    -FirstRun    open model setup (pyre-firstrun.ps1)
    -NoFirstRun  never open model setup (autostart / service)
    -EditConfig  open config.toml in Notepad
    -NoTray      do not bring up pyre-tray.exe (the tray passes this when it starts the server)
    -DryRun      print the resolved command line + env and exit (safe; starts nothing)
#>
param([switch]$Gui, [switch]$Web, [switch]$Stop, [switch]$FirstRun, [switch]$NoFirstRun, [switch]$EditConfig, [switch]$DryRun, [switch]$NoTray, [string]$Config)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$data = Join-Path $env:ProgramData 'pyre'
if (-not $Config) { $Config = if ($env:PYRE_CONFIG) { $env:PYRE_CONFIG } else { Join-Path $data 'config.toml' } }
if (-not (Test-Path $Config)) {
  New-Item -ItemType Directory -Force (Split-Path $Config) | Out-Null
  Copy-Item (Join-Path $here 'config.template.toml') $Config
}

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

if ($EditConfig) { Start-Process notepad.exe $Config; return }
$cfg = Read-Toml $Config; $s = $cfg['serve']; if (-not $s) { $s = @{} }
function V($k, $d = '') { if ($s.ContainsKey($k) -and "$($s[$k])" -ne '') { "$($s[$k])" } else { $d } }
$port = [int](V 'port' '8099'); $hostIp = V 'host' '127.0.0.1'
$pidFile = Join-Path $data 'pyre-server.pid'
$exe = Join-Path $here 'pyre.exe'

if ($Stop) {
  # Graceful stop (pyre builds newer than cf84f1d, int45+): pyre polls <its %TEMP%>\pyre\stop.<pid>; at the next step boundary
  # it drains the GPU and exits 87 (88 if no step comes within PYRE_STOP_GRACE_S, default 10 s). Older builds ignore the file,
  # and a server started under another account (e.g. the SYSTEM task) has a different %TEMP%, so we fall back to killing the pid.
  $p = Get-Content $pidFile -ea 0
  if ($p) {
    $proc = Get-Process -Id ([int]$p) -ea 0
    if ($proc -and ($proc.Path -eq $exe -or (-not $proc.Path -and $proc.ProcessName -eq 'pyre'))) {
      $sf = Join-Path ([IO.Path]::GetTempPath()) "pyre\stop.$($proc.Id)"
      New-Item -ItemType Directory -Force (Split-Path $sf) | Out-Null; Set-Content $sf 'stop'
      # SYSTEM-task server: its %TEMP% is %SystemRoot%\SystemTemp or %SystemRoot%\Temp (writable only when elevated)
      foreach ($st in "$env:SystemRoot\SystemTemp", "$env:SystemRoot\Temp") { try { New-Item -ItemType Directory -Force "$st\pyre" -ea Stop | Out-Null; Set-Content "$st\pyre\stop.$($proc.Id)" 'stop' -ea Stop } catch {} }
      $grace = 10; if ($cfg['env'] -and $cfg['env'].PYRE_STOP_GRACE_S) { $grace = [int]$cfg['env'].PYRE_STOP_GRACE_S }
      if (-not $proc.WaitForExit(($grace + 5) * 1000)) { Stop-Process -Id $proc.Id -Force -ea 0 }
      Remove-Item $sf -ea 0
    }
  }
  Remove-Item $pidFile -ea 0; return
}
if ($FirstRun -or (-not $NoFirstRun -and -not $DryRun -and -not (V 'model'))) {
  & (Join-Path $here 'pyre-firstrun.ps1') -Config $Config
  $cfg = Read-Toml $Config; $s = $cfg['serve']
  if ($FirstRun -and -not $Gui -and -not $Web) { return }
}

$a = @('serve', '--host', $hostIp, '--port', "$port", '-c', (V 'ctx_size' '4096'), '-n', (V 'n_predict' '512'))
if (V 'backend') { $a += @('-b', (V 'backend')) }
if (V 'model') { $a += @('-m', (V 'model')) }
foreach ($m in @(@('image_model', '--image-model'), @('stt_model', '--stt-model'), @('tts_model', '--tts-model'), @('vram_reserve', '--vram-reserve'))) { if (V $m[0]) { $a += @($m[1], (V $m[0])) } }
if ((V 'gpu_util_max' '0') -ne '0') { $a += @('--gpu-util-max', (V 'gpu_util_max')) }
if (V 'extra_args') { $a += (V 'extra_args') -split '\s+' }

$envs = [ordered]@{ PYRE_CONFIG = $Config }
$envs.PYRE_QN_FAST = if ((V 'fast' 'true') -match '^(false|0|off)$') { '0' } else { '1' }
foreach ($m in @(@('vram_floor_mb', 'PYRE_VRAM_FLOOR_MB'), @('api_key', 'PYRE_API_KEY'), @('metrics_bind', 'PYRE_METRICS_BIND'), @('metrics_token', 'PYRE_METRICS_TOKEN'), @('log_file', 'PYRE_LOG_FILE'), @('instance', 'PYRE_INSTANCE'))) { if (V $m[0]) { $envs[$m[1]] = V $m[0] } }
if ($cfg['env']) { foreach ($k in $cfg['env'].Keys) { $envs[$k] = $cfg['env'][$k] } }
if ($envs.PYRE_LOG_FILE -and $envs.PYRE_LOG_FILE -notin 'off', '0') { New-Item -ItemType Directory -Force (Split-Path $envs.PYRE_LOG_FILE) -ea 0 | Out-Null }

$url = "http://127.0.0.1:$port"
if ($DryRun) {
  "config : $Config"; "exe    : $exe"; "args   : $($a -join ' ')"
  foreach ($k in $envs.Keys) { $v = if ($k -match 'KEY|TOKEN') { if ($envs[$k]) { '<set>' } else { '' } } else { $envs[$k] }; "env    : $k=$v" }
  "url    : $url"; return
}

function Test-Port { try { $c = [Net.Sockets.TcpClient]::new(); $ok = $c.ConnectAsync('127.0.0.1', $port).Wait(500); $c.Dispose(); $ok } catch { $false } }
if (-not (Test-Port)) {
  foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
  $out = Join-Path $data 'logs'; New-Item -ItemType Directory -Force $out | Out-Null
  $p = Start-Process -FilePath $exe -ArgumentList $a -WorkingDirectory $here -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $out 'pyre-stderr.log') -RedirectStandardOutput (Join-Path $out 'pyre-stdout.log')
  Set-Content $pidFile $p.Id
  if ($Gui -or $Web) { for ($i = 0; $i -lt 240 -and -not (Test-Port); $i++) { Start-Sleep -Milliseconds 500 } }
}
$tray = Join-Path $here 'pyre-tray.exe'
if (-not $NoTray -and (Test-Path $tray)) { Start-Process $tray }   # single-instance; brings up the tray icon
if ($Gui) { Start-Process (Join-Path $here 'pyre-gui.exe') -ArgumentList $url }
if ($Web) { Start-Process $url }
