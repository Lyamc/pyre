# Compare pyre Parakeet TDT v3 with sherpa-onnx NeMo Parakeet v3 int8 (CPU, 8 threads).
# Usage: scripts\parakeet_compare.ps1 [-File samples\speech.wav] [-Runs 6] [-Reference "..."]
param(
  [string]$File = "samples\speech.wav",
  [int]$Runs = 6,
  [string]$Reference = "the quick brown fox jumps over the lazy dog pyre runs speech recognition with burn on the graphics card"
)
$root = Split-Path $PSScriptRoot -Parent
$py = "C:\Users\Lyam\whisper-server\venv\Scripts\python.exe"
$audio = Join-Path $root $File
$out = "$env:TEMP\parakeet-compare.txt"

# pyre (warm-up request included)
$times = @()
for ($i = 0; $i -le $Runs; $i++) {
  $t = & curl.exe --noproxy '*' -s -o "$env:TEMP\pc-pyre-$i.json" -w "%{time_total}" -F "file=@$audio" -F "model=whisper-1" -F "language=en" -F "experimental=true" -F "response_format=verbose_json" http://127.0.0.1:8099/v1/audio/transcriptions
  if ($i -gt 0) { $times += [double]$t }
}
$sorted = $times | Sort-Object
$median = $sorted[[int][math]::Floor($sorted.Count / 2)]
$text = (Get-Content "$env:TEMP\pc-pyre-$Runs.json" -Raw | ConvertFrom-Json).text
"pyre      median=$([math]::Round($median,3))s runs=$(($times | ForEach-Object { [math]::Round($_,3) }) -join ',')"
"pyre text: $text"
& $py (Join-Path $PSScriptRoot "stt_wer.py") $Reference $text

# sherpa-onnx baseline (own warm-up inside the script)
& $py (Join-Path $PSScriptRoot "parakeet_sherpa_bench.py") $audio $Runs 8 cpu | Tee-Object -FilePath $out
$sherpaText = (Get-Content $out | Select-String 'text:').Line -replace '^text: ', ''
"sherpa WER:"
& $py (Join-Path $PSScriptRoot "stt_wer.py") $Reference $sherpaText
