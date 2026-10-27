param($model, $tag, $envs="")
# Bounded decode benchmark: tok/s from pyre's own engine log, VRAM delta from nvidia-smi.
$exe="C:\Build\ai\pyre\target\x86_64-pc-windows-msvc\release\pyre.exe"
foreach($kv in ($envs -split ";" | ? {$_})) { $k,$v=$kv -split "=",2; Set-Item "env:$k" $v }
$log="C:\Build\ai\pyre\samples\fqb-$tag.log"
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$p=Start-Process $exe -ArgumentList "serve -m `"$model`" --port 8099" -RedirectStandardError $log -WindowStyle Hidden -PassThru
try {
  $ok=$false; for($i=0;$i -lt 60;$i++){ Start-Sleep 3; try { Invoke-RestMethod http://127.0.0.1:8099/v1/models -TimeoutSec 5 | Out-Null; $ok=$true; break } catch {} }
  if(-not $ok){ "[$tag] server did not come up"; return }
  $body=@{messages=@(@{role="user";content="Count from 1 to 300, separated by commas. Output only the numbers."}); max_tokens=256; temperature=0} | ConvertTo-Json -Depth 5
  $out=$null; foreach($n in 1..3){ $r=Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post -ContentType application/json -Body $body -TimeoutSec 240; if(-not $out){ $out=$r.choices[0].message.content } }
  $mem=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  $g=(Get-Content $log | Select-String "gen (\d+) tok ([\d.]+) tok/s" | Select -Last 3 | % { [double]$_.Matches[0].Groups[2].Value })
  $pp=(Get-Content $log | Select-String "prompt (\d+) tok \(0 cached\) ([\d.]+) tok/s" | Select -Last 1 | % { $_.Matches[0].Groups[2].Value })
  "[$tag] gpu +$($mem-$base) MiB | decode tok/s runs: $($g -join ', ') | first prefill $pp tok/s | out md5 $(([BitConverter]::ToString((New-Object Security.Cryptography.MD5CryptoServiceProvider).ComputeHash([Text.Encoding]::UTF8.GetBytes($out)))).Replace('-','').Substring(0,8)) `"$($out.Substring(0,[Math]::Min(40,$out.Length)))`""
} finally { Stop-Process $p.Id -Force -ErrorAction SilentlyContinue }
