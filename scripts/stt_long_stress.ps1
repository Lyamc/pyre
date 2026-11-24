# Long-audio OOM stress: start pyre with one STT model, transcribe a long WAV through
# /api/audio/jobs, and sample the server's working set while it runs.
# Usage: scripts\stt_long_stress.ps1 -Model tiny_en -File C:\Temp\long2h.wav
param(
  [string]$Model = "tiny_en",
  [string]$File = "$env:TEMP\long2h.wav",
  [int]$Port = 8099,
  [int]$TimeoutSec = 7200,
  [string]$Log = "$env:TEMP\pyre-long-$Model.log"
)
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root "target\x86_64-pc-windows-msvc\release\pyre.exe"
$base = "http://127.0.0.1:$Port"

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
  $procId = $existing.OwningProcess
  "using existing server pid $procId"
} else {
  $p = Start-Process -FilePath $exe -ArgumentList "serve --stt-model $Model --host 127.0.0.1 --port $Port -b cuda" -PassThru -WindowStyle Hidden -RedirectStandardError $Log -RedirectStandardOutput "$env:TEMP\pyre-long-$Model.out"
  $procId = $p.Id
  for ($i = 0; $i -lt 120; $i++) {
    try { if ((Invoke-WebRequest "$base/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) { break } } catch {}
    Start-Sleep -Seconds 1
  }
  "started server pid $procId"
}

$peak = 0L
$samples = 0
$trend = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$job = Start-Job -ScriptBlock {
  param($base, $file, $sec, $model)
  curl.exe --noproxy '*' -s -N -m $sec -X POST -F "file=@$file" -F "model=$model" -F "response_format=verbose_json" "$base/api/audio/jobs"
} -ArgumentList $base, $File, $TimeoutSec, $Model

while ($job.State -eq 'Running') {
  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  if ($proc) {
    $mb = [math]::Round($proc.WorkingSet64 / 1MB, 1)
    if ($mb -gt $peak) { $peak = $mb }
    if ($samples % 15 -eq 0) { $trend += "$([math]::Round($sw.Elapsed.TotalSeconds))s:${mb}MB" }
    $samples++
  }
  Start-Sleep -Seconds 2
}
$out = Receive-Job $job
Remove-Job $job
$sw.Stop()
"model=$Model audio=$File elapsed=$([math]::Round($sw.Elapsed.TotalSeconds,1))s peak_rss=${peak}MB samples=$samples"
"rss trend: $($trend -join ' ')"
$outFile = "$env:TEMP\pyre-long-$Model.sse"
$out | Set-Content $outFile
"job output: $outFile ($((Get-Item $outFile).Length) bytes)"
$done = $out | Select-String '"stage":"done"' | Select-Object -Last 1
if ($done) { "result: $($done.Line.Substring(0, [math]::Min(300, $done.Line.Length)))" }
else { "no done event; tail: $(($out | Select-Object -Last 3) -join ' ')" }
