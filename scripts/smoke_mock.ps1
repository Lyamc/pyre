# smoke_mock.ps1: smoke-test a pyre build without a GPU or model weights, safe next to a live production server.
# Starts the exe with PYRE_MOCK_LLM=1 (scripted engine: replays the text after "MOCK_REPLY:" in the last user turn)
# on a spare loopback port, checks the OpenAI chat endpoint (content, tool calls, reasoning, finish_reason, streaming),
# then stops it. Exit code = number of failed checks.
#   .\scripts\smoke_mock.ps1 -exe C:\path\pyre.exe [-tokdir <dir with tokenizer.json + chat template>] [-port 8106]
param([Parameter(Mandatory)][string]$exe, [string]$tokdir='C:\AI\models\tokenizers\Qwen3.8-Flash-Next', [int]$port=8106, [switch]$keep)
$ErrorActionPreference='Stop'
$base="http://127.0.0.1:$port"; $log=Join-Path $env:TEMP "pyre_smoke_$port.err.log"
$env:PYRE_MOCK_LLM='1'
try { $p=Start-Process $exe -ArgumentList "serve -m `"$tokdir`" --host 127.0.0.1 --port $port -c 8192 -b cpu" -RedirectStandardOutput "$log.out" -RedirectStandardError $log -WindowStyle Hidden -PassThru }
finally { Remove-Item Env:PYRE_MOCK_LLM }
$p.PriorityClass='BelowNormal'
$up=$false; for($i=0;$i -lt 60;$i++){ Start-Sleep -Milliseconds 500; if($p.HasExited){ break }; try{ if((Invoke-WebRequest -UseBasicParsing "$base/health" -TimeoutSec 2).Content -match 'ok'){ $up=$true; break } }catch{} }
if(-not $up){ Write-Host "FAIL server did not start (log: $log)"; Get-Content $log -Tail 15; if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force }; exit 99 }
$script:fail=0
function Check($name,$ok,$detail){ if($ok){ Write-Host "PASS $name" } else { $script:fail++; Write-Host "FAIL $name :: $detail" } }
$tools=@(@{type='function';function=@{name='write_file';description='Write a file';parameters=@{type='object';properties=@{path=@{type='string'};content=@{type='string'}};required=@('path','content')}}})
function Chat($reply,[int]$mt=256,[switch]$stream,[switch]$notools){
  $b=@{model='mock';max_tokens=$mt;stream=[bool]$stream;messages=@(@{role='user';content="MOCK_REPLY: $reply"})}; if(-not $notools){ $b.tools=$tools }
  $j=$b | ConvertTo-Json -Depth 10
  if(-not $stream){ return Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($j)) -TimeoutSec 60 }
  $raw=(Invoke-WebRequest -UseBasicParsing -Uri "$base/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($j)) -TimeoutSec 60).Content
  $fin=$null; $content=''; $reason=''; $names=@(); $args=''
  foreach($l in ($raw -split "`n")){ if($l -notmatch '^data: (\{.*)'){ continue }; $o=$Matches[1] | ConvertFrom-Json; foreach($c in $o.choices){ if($c.finish_reason){ $fin=$c.finish_reason }; $d=$c.delta; if($d.content){ $content+=$d.content }; if($d.reasoning_content){ $reason+=$d.reasoning_content }; foreach($t in @($d.tool_calls)){ if($t){ if($t.function.name){ $names+=$t.function.name }; if($t.function.arguments){ $args+=$t.function.arguments } } } } }
  [pscustomobject]@{finish=$fin;content=$content;reasoning=$reason;names=$names;args=$args;raw=$raw}
}
function NTok($s){ (Invoke-RestMethod -Uri "$base/tokenize" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes((@{content=$s} | ConvertTo-Json)))).tokens.Count }
$call1="<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>"
$call2="\n<tool_call>\n<function=write_file>\n<parameter=path>\nb.txt\n</parameter>\n<parameter=content>\na long second file that the cap cuts off\n</parameter>\n</function>\n</tool_call>"
try {
  # 1 plain text
  $r=Chat 'Hello world.' -notools; $c=$r.choices[0]
  Check 'plain: content + stop' ($c.message.content -eq 'Hello world.' -and $c.finish_reason -eq 'stop') "content=[$($c.message.content)] finish=$($c.finish_reason)"
  $s=Chat 'Hello world.' -stream -notools
  Check 'plain stream: content + stop' ($s.content -eq 'Hello world.' -and $s.finish -eq 'stop') "content=[$($s.content)] finish=$($s.finish)"
  # 2 max_tokens cut on plain text
  $r=Chat 'one two three four five six seven eight nine ten' -mt 3 -notools; $c=$r.choices[0]
  Check 'cap: finish length' ($c.finish_reason -eq 'length' -and $r.usage.completion_tokens -eq 3) "finish=$($c.finish_reason) n=$($r.usage.completion_tokens)"
  $s=Chat 'one two three four five six seven eight nine ten' -mt 3 -stream -notools
  Check 'cap stream: finish length' ($s.finish -eq 'length') "finish=$($s.finish)"
  # 3 complete tool call
  $r=Chat $call1; $c=$r.choices[0]; $tc=@($c.message.tool_calls)
  $a=if($tc.Count){ $tc[0].function.arguments | ConvertFrom-Json }
  Check 'tool call: parsed + tool_calls' ($c.finish_reason -eq 'tool_calls' -and $tc.Count -eq 1 -and $tc[0].function.name -eq 'write_file' -and $a.path -eq 'a.txt' -and $a.content -eq 'hi') "finish=$($c.finish_reason) calls=$($tc.Count) args=$($tc[0].function.arguments)"
  $s=Chat $call1 -stream
  Check 'tool call stream: parsed + tool_calls' ($s.finish -eq 'tool_calls' -and ($s.names -contains 'write_file') -and $s.args -match 'a\.txt') "finish=$($s.finish) names=$($s.names -join ',') args=$($s.args)"
  # 4 max_tokens cuts a second tool call: the first still parses, but finish must be length (PR #28)
  $mt=(NTok ($call1.Replace('\n',"`n"))) + 6
  $r=Chat ($call1+$call2) -mt $mt; $c=$r.choices[0]
  Check 'cut tool call: finish length' ($c.finish_reason -eq 'length') "finish=$($c.finish_reason) calls=$(@($c.message.tool_calls).Count) mt=$mt"
  $s=Chat ($call1+$call2) -mt $mt -stream
  Check 'cut tool call stream: finish length' ($s.finish -eq 'length') "finish=$($s.finish) mt=$mt"
  # 5 reasoning split
  $r=Chat '<think>\nplan it\n</think>\n\nAnswer.' -notools; $c=$r.choices[0]
  Check 'reasoning: split' ($c.message.reasoning_content -match 'plan it' -and $c.message.content.Trim() -eq 'Answer.') "reasoning=[$($c.message.reasoning_content)] content=[$($c.message.content)]"
  $s=Chat '<think>\nplan it\n</think>\n\nAnswer.' -stream -notools
  Check 'reasoning stream: split' ($s.reasoning -match 'plan it' -and $s.content.Trim() -eq 'Answer.') "reasoning=[$($s.reasoning)] content=[$($s.content)]"
} catch { $script:fail++; Write-Host "FAIL exception: $_" }
finally { if(-not $keep -and -not $p.HasExited){ Stop-Process -Id $p.Id -Force } }
Write-Host "$(if($fail){'FAILED'}else{'OK'}): $fail failure(s)"
exit $fail