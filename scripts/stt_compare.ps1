# Head-to-head STT latency benchmark: pyre vs an external faster-whisper server.
# Usage: scripts\stt_compare.ps1 [-PyreUrl http://127.0.0.1:8099] [-FwUrl http://127.0.0.1:8686]
#                                [-Runs 3] [-Files samples\speech.wav,samples\speech-slow3x.wav]
# Both servers must already be running and warm.
param(
  [string]$PyreUrl = "http://127.0.0.1:8099",
  [string]$FwUrl = "http://127.0.0.1:8686",
  [int]$Runs = 3,
  [string]$Files = "samples\speech.wav,samples\speech-slow3x.wav",
  [string]$Language = "en"
)

$root = Split-Path $PSScriptRoot -Parent
$tmp = Join-Path $env:TEMP "pyre-stt-compare"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

function Invoke-Timed {
  param([string]$Url, [string]$Path, [string]$Out)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & curl.exe --noproxy '*' -s -o $Out -w "%{http_code}" -F "file=@$Path" -F "model=whisper-1" -F "language=$Language" -F "response_format=verbose_json" $Url | Out-Null
  $sw.Stop()
  return $sw.Elapsed.TotalSeconds
}

function Get-Text {
  param([string]$Out, [string]$Kind)
  $last = Get-Content $Out -Tail 1
  if (-not $last) { return "" }
  try {
    if ($Kind -eq "pyre") {
      $j = $last | ConvertFrom-Json
      if ($j.text) { return [string]$j.text }
    } else {
      $j = $last | ConvertFrom-Json
      if ($j.event -eq "done") { return [string]$j.data.text }
    }
  } catch {}
  return ""
}

foreach ($rel in $Files.Split(",")) {
  $path = Join-Path $root $rel.Trim()
  if (-not (Test-Path $path)) { Write-Host "MISSING $path"; continue }
  $audioSecs = [double](& ffprobe.exe -v error -show_entries format=duration -of csv=p=0 $path 2>$null)
  Write-Host ""
  Write-Host "=== $rel ($([math]::Round($audioSecs,2)) s audio) ==="

  foreach ($target in @(@("pyre", "$PyreUrl/v1/audio/transcriptions"), @("faster-whisper", "$FwUrl/transcribe"))) {
    $name = $target[0]; $url = $target[1]
    $times = @(); $text = ""
    for ($i = 0; $i -le $Runs; $i++) {
      $out = Join-Path $tmp "$($name -replace '\W','')-$($i).json"
      $t = Invoke-Timed -Url $url -Path $path -Out $out
      if ($i -gt 0) { $times += $t }   # first call is the warm-up
      $text = Get-Text -Out $out -Kind $name
    }
    $warm = $times | Sort-Object
    $med = $warm[[int][math]::Floor($warm.Count / 2)]
    "  {0,-16} median {1,7:N3}s  min {2,7:N3}s  max {3,7:N3}s  ({4,5:N1}x realtime)" -f $name, $med, $warm[0], $warm[-1], ($audioSecs / $med)
    "  {0,-16} text: {1}" -f "", ($text.Substring(0, [math]::Min(110, $text.Length)))
  }
}
