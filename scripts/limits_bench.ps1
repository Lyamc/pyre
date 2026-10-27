param($tag, $extra="", $models="-m Qwen/Qwen2.5-0.5B-Instruct", $bench="llm")
$exe="C:\Build\ai\pyre\target\x86_64-pc-windows-msvc\release\pyre.exe"
$p=Start-Process $exe -ArgumentList "serve $models --port 8099 $extra" -RedirectStandardError "C:\Build\ai\pyre\samples\serve-$tag.log" -WindowStyle Hidden -PassThru
try {
  $ok=$false; for($i=0;$i -lt 100;$i++){ Start-Sleep 3; if($p.HasExited){break}; try { Invoke-RestMethod http://127.0.0.1:8099/v1/models | Out-Null; $ok=$true; break } catch {} }
  if(-not $ok){ "[$tag] server did not come up:"; Get-Content "C:\Build\ai\pyre\samples\serve-$tag.log" -Tail 5; return }
  if($bench -eq "none"){ Start-Sleep 12 }
  if($bench -eq "llm"){ & C:\Build\ai\pyre\target\pyvenv\Scripts\python.exe C:\Build\ai\pyre\scripts\bench_concurrency.py 8099 1,8 128 | % {"[$tag] $_"} }
  if($bench -eq "sd"){ $b=@{prompt="a lighthouse at dusk, oil painting";n=1;size="512x512";steps=20} | ConvertTo-Json; foreach($k in 1..2){ $t=Get-Date; Invoke-RestMethod http://127.0.0.1:8099/v1/images/generations -Method Post -ContentType application/json -Body $b -TimeoutSec 600 | Out-Null; "[$tag] image $k : $([math]::Round(((Get-Date)-$t).TotalSeconds,2)) s" } }
  $s=Invoke-RestMethod http://127.0.0.1:8099/api/status; "[$tag] limits: " + ($s.limits | ConvertTo-Json -Compress)
  Select-String "C:\Build\ai\pyre\samples\serve-$tag.log" -Pattern "GPU limits|slots|warm-up" | % {"[$tag] log: " + $_.Line.Substring([Math]::Max(0,$_.Line.Length-140))}
} finally { Stop-Process $p.Id -Force -ErrorAction SilentlyContinue }

