# End-to-end smoke test: starts pyre, exercises the main endpoints, prints results.
# Usage: scripts\smoke.ps1 [-Model Qwen/Qwen2.5-0.5B-Instruct] [-Backend cuda] [-Port 8091]
param([string]$Model = "Qwen/Qwen2.5-0.5B-Instruct", [string]$Backend = "cuda", [int]$Port = 8091, [string]$ApiKey = "")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root "target\x86_64-pc-windows-msvc\release\pyre.exe"
$args = "-m $Model -b $Backend --port $Port"
if ($ApiKey) { $args += " --api-key $ApiKey" }
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardError "$env:TEMP\pyre-smoke.err" -RedirectStandardOutput "$env:TEMP\pyre-smoke.out"
$base = "http://127.0.0.1:$Port"
$h = @{}; if ($ApiKey) { $h["Authorization"] = "Bearer $ApiKey" }
try {
  for ($i = 0; $i -lt 600; $i++) { try { if ((Invoke-RestMethod "$base/health").status -eq "ok") { break } } catch {}; Start-Sleep 1 }
  Start-Sleep 1
  function post($path, $body) { $tmp = New-TemporaryFile; ($body | ConvertTo-Json -Depth 8 -Compress) | Set-Content -Encoding ascii $tmp
    $a = @("-s", "-N", "$base$path", "-H", "Content-Type: application/json", "-d", "@$tmp"); if ($ApiKey) { $a += @("-H", "Authorization: Bearer $ApiKey") }
    $r = & curl.exe @a; Remove-Item $tmp; $r }
  Write-Host "== /v1/models"; & curl.exe -s "$base/v1/models" -H "Authorization: Bearer $ApiKey"; ""
  Write-Host "== /v1/chat/completions (non-stream)"
  post "/v1/chat/completions" @{ model = "x"; messages = @(@{ role = "user"; content = "What is the capital of France? One sentence." }); temperature = 0; max_tokens = 48 }; ""
  Write-Host "== /v1/chat/completions (stream)"
  post "/v1/chat/completions" @{ messages = @(@{ role = "user"; content = "List three fruits, comma separated." }); stream = $true; max_tokens = 32; seed = 7; stream_options = @{ include_usage = $true } } | Select-Object -Last 6
  Write-Host "== /completion (llama.cpp)"
  post "/completion" @{ prompt = "The quick brown fox"; n_predict = 24; temperature = 0 }; ""
  Write-Host "== /v1/completions"
  post "/v1/completions" @{ prompt = "1, 2, 3, 4,"; max_tokens = 12; temperature = 0; stop = @("9") }; ""
  Write-Host "== /tokenize + /detokenize"
  post "/tokenize" @{ content = "Hello world" }; post "/detokenize" @{ tokens = @(9707, 1879) }; ""
  Write-Host "== /v1/embeddings"; $e = post "/v1/embeddings" @{ input = "hello" } | ConvertFrom-Json; "dim=$($e.data[0].embedding.Count)"
  Write-Host "== /props /slots"; & curl.exe -s "$base/slots" -H "Authorization: Bearer $ApiKey"; ""
} finally { Stop-Process -Id $p.Id -Force }
