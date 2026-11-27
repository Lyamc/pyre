# Per-model Whisper tuning sweep: latency + WER for each model at several beam sizes.
# Usage: scripts\stt_tune.ps1 -Models tiny_en,base,small,medium,large-v3-turbo -Beams 1,2,3,5
param(
  [string]$Models = "tiny_en,base,small,medium,large-v3-turbo",
  [string]$Beams = "1,2,3,5",
  [string]$File = "samples\speech.wav",
  [string]$Reference = "the quick brown fox jumps over the lazy dog pyre runs speech recognition with burn on the graphics card",
  [int]$Port = 8099,
  [int]$Runs = 3
)
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root "target\x86_64-pc-windows-msvc\release\pyre.exe"
$base = "http://127.0.0.1:$Port"
$audio = Join-Path $root $File

foreach ($model in $Models.Split(",")) {
  $model = $model.Trim()
  $log = Join-Path $env:TEMP "pyre-tune-$model.log"
  $env:PYRE_VRAM_RESERVE = "0"
  $p = Start-Process -FilePath $exe -ArgumentList "serve --stt-model $model --host 127.0.0.1 --port $Port -b cuda --vram-reserve 0" -PassThru -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\pyre-tune-$model.out"
  try {
    $ready = $false
    for ($i = 0; $i -lt 240; $i++) {
      if ($p.HasExited) { Write-Host "${model}: server exited: $(Get-Content $log -Tail 2)"; break }
      try { if ((Invoke-WebRequest -Uri "$base/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) { $ready = $true; break } } catch {}
      Start-Sleep -Seconds 1
    }
    if (-not $ready -or $p.HasExited) { continue }
    # Absorb JIT / clock ramp once.
    $null = & curl.exe --noproxy '*' -s -o NUL -F "file=@$audio" -F "model=whisper-1" -F "language=en" "$base/v1/audio/transcriptions"

    foreach ($beam in $Beams.Split(",")) {
      $beam = $beam.Trim()
      $times = @()
      for ($i = 0; $i -le $Runs; $i++) {
        $out = "$env:TEMP\pyre-tune-$model-$beam-$i.json"
        $beamField = @()
        if ($beam -ne "default") { $beamField = @("-F", "beam_size=$beam") }
        $t = & curl.exe --noproxy '*' -s -o $out -w "%{time_total}" -F "file=@$audio" -F "model=whisper-1" -F "language=en" @beamField -F "response_format=verbose_json" "$base/v1/audio/transcriptions"
        if ($i -gt 0) { $times += [double]$t }
      }
      $sorted = $times | Sort-Object
      $median = $sorted[[int][math]::Floor($sorted.Count / 2)]
      $text = (Get-Content "$env:TEMP\pyre-tune-$model-$beam-$Runs.json" -Raw | ConvertFrom-Json).text
      $wer = & python (Join-Path $PSScriptRoot "stt_wer.py") $Reference $text
      "{0,-16} beam={1,-2} median={2,6:N3}s  {3}  {4}" -f $model, $beam, $median, $wer, $text.Substring(0, [math]::Min(60, $text.Length))
    }
  } finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    Start-Sleep -Seconds 1
  }
}
