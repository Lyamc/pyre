# Bounded first-request timing: start pyre, time first + second STT (and optionally TTS) request, stop pyre.
# Usage: scripts\first_request.ps1 [-Backend cuda] [-Stt tiny_en] [-Tts ""] [-Port 8099] [-TimeoutSec 600]
param([string]$Backend = "cuda", [string]$Stt = "tiny_en", [string]$Tts = "", [string]$Llm = "", [int]$Port = 8099, [int]$TimeoutSec = 600, [string]$Extra = "")
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root "target\x86_64-pc-windows-msvc\release\pyre.exe"
$a = "serve -b $Backend --port $Port --stt-model $Stt $Extra"
if ($Tts) { $a += " --tts-model $Tts" }
if ($Llm) { $a += " -m $Llm" }
$t0 = Get-Date
$p = Start-Process -FilePath $exe -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardError "$env:TEMP\pyre-first.err" -RedirectStandardOutput "$env:TEMP\pyre-first.out"
$base = "http://127.0.0.1:$Port"
try {
  for ($i = 0; $i -lt 300; $i++) { if ($p.HasExited) { throw "pyre exited" }; try { if ((Invoke-RestMethod "$base/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}; Start-Sleep 1 }
  "startup: {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds
  foreach ($n in 1, 2) {
    $t = Get-Date
    $r = & curl.exe -s -m $TimeoutSec "$base/v1/audio/transcriptions" -F "file=@$root\samples\speech.wav" -F "model=$Stt"
    "stt #$n : {0:N1}s  {1}" -f ((Get-Date) - $t).TotalSeconds, ($r | Out-String).Trim()
  }
  if ($Tts) { foreach ($n in 1, 2) { $t = Get-Date
    & curl.exe -s -m $TimeoutSec -o "$env:TEMP\pyre-first.wav" "$base/v1/audio/speech" -H "Content-Type: application/json" -d '{\"model\":\"tts\",\"input\":\"Hello from pyre.\"}'
    "tts #$n : {0:N1}s" -f ((Get-Date) - $t).TotalSeconds } }
} finally { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; Start-Sleep 1 }
