# HYTE bench for the IQ3_S Qwen3.8-Flash-Next mix.
# Build:  C:\Temp\qnx\build.ps1
# Model:  C:\AI\models\models\IQ3_S
# Tokens: C:\AI\models\tokenizers\Qwen3.8-Flash-Next
# Prints decode tok/s, expert-cache hit rate, and free VRAM.
# Optional: -Port 8099 -Tokens 64 -Spec 0 (0 = the server default, which is MTP when multi_ok).
param(
    [string]$Model = "C:\AI\models\models\IQ3_S",
    [string]$Tokenizer = "C:\AI\models\tokenizers\Qwen3.8-Flash-Next",
    [string]$Build = "C:\Temp\qnx\build.ps1",
    [int]$Port = 8099,
    [int]$Tokens = 64,
    [int]$Spec = -1
)
$ErrorActionPreference = "Stop"
if (Test-Path $Build) { & $Build } else { Write-Host "build script not found: $Build (using an existing pyre.exe)" }
$exe = @(
    "C:\Build\ai\pyre\target\x86_64-pc-windows-msvc\release\pyre.exe",
    "C:\Temp\qnx\target\x86_64-pc-windows-msvc\release\pyre.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "pyre.exe not found after build" }
$env:PYRE_QN_TOKENIZER = $Tokenizer
$env:PYRE_QN_XSTAT = "1"
$env:PYRE_VRAM_FLOOR_MB = "1024"
if ($Spec -ge 0) { $env:PYRE_SPEC = "$Spec" } else { Remove-Item Env:PYRE_SPEC -ErrorAction SilentlyContinue }
$log = Join-Path $env:TEMP "pyre-iq3s-$Port.log"
$p = Start-Process $exe -ArgumentList "serve -m `"$Model`" -b cuda --port $Port" -RedirectStandardError $log -WindowStyle Hidden -PassThru
try {
    $up = $false
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Seconds 3
        try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" | Out-Null; $up = $true; break } catch {}
    }
    if (-not $up) { Get-Content $log -Tail 40; throw "server did not come up (see $log)" }
    $st = Invoke-RestMethod "http://127.0.0.1:$Port/api/status"
    $body = @{ messages = @(@{ role = "user"; content = "Write a haiku about autumn rain." }); max_tokens = $Tokens; temperature = 0 } | ConvertTo-Json -Depth 5
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600
    $sw.Stop()
    $m = Invoke-RestMethod "http://127.0.0.1:$Port/api/metrics"
    $free = $null
    try { $free = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim() } catch {}
    $n = [int]$r.usage.completion_tokens
    $tps = if ($r.timings.predicted_per_second) { $r.timings.predicted_per_second } elseif ($sw.Elapsed.TotalSeconds -gt 0) { $n / $sw.Elapsed.TotalSeconds } else { 0 }
    $hit = $m.expert_cache.hit_rate
    $spec = $st.speculative
    Write-Host ("tok/s {0:N2}  ({1} tokens)" -f $tps, $n)
    Write-Host ("expert hit rate {0}  hits {1}  misses {2}  resident {3}" -f $hit, $m.expert_cache.hits, $m.expert_cache.misses, $m.expert_cache.resident)
    Write-Host ("free VRAM {0} MiB" -f $free)
    Write-Host ("speculative enabled={0} draft_len={1} drafter={2} acceptance={3}" -f $spec.enabled, $spec.draft_len, $spec.drafter, $spec.acceptance)
    Write-Host "server log: $log"
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
