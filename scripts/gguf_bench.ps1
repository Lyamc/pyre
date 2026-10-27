param($model, $tag, $envs="")
$exe="C:\Build\ai\pyre\target\x86_64-pc-windows-msvc\release\pyre.exe"
foreach($kv in ($envs -split ";" | ? {$_})) { $k,$v=$kv -split "=",2; Set-Item "env:$k" $v }
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$p=Start-Process $exe -ArgumentList "serve -m `"$model`" --port 8099" -RedirectStandardError "C:\Build\ai\pyre\samples\serve-$tag.log" -WindowStyle Hidden -PassThru
try {
  $ok=$false; for($i=0;$i -lt 60;$i++){ Start-Sleep 3; try { Invoke-RestMethod http://127.0.0.1:8099/v1/models | Out-Null; $ok=$true; break } catch {} }
  if(-not $ok){ "server did not come up"; return }
  $body=@{messages=@(@{role="user";content="Write a haiku about autumn rain."}); max_tokens=128; temperature=0} | ConvertTo-Json -Depth 5
  $rs=@(); foreach($n in 1..3){ $t=Get-Date; $r=Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -ContentType application/json -Body $body -TimeoutSec 300; $rs+=,@($r, ((Get-Date)-$t).TotalSeconds) }
  $mem=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  $p.Refresh(); $rss=[int]($p.WorkingSet64/1MB)
  $r,$sec=$rs[-1]
  "[$tag] gpu +$($mem-$base) MiB, rss $rss MiB, tokens $($r.usage.completion_tokens) in $([math]::Round($sec,2)) s"
  if($r.timings){ "timings: " + ($r.timings | ConvertTo-Json -Compress) }
  "text: " + $r.choices[0].message.content
} finally { Stop-Process $p.Id -Force -ErrorAction SilentlyContinue }
