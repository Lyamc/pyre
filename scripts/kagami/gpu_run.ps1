# Bounded, gated pyre GPU run: gate -> run with timeout, sample VRAM -> check System log for display-driver errors.
param([string[]]$PyreArgs, [string]$Name = "run", [int]$TimeoutSec = 600)
$t0 = Get-Date
& C:\Build\ai\pyre-tagger\scripts\gpu_gate.ps1 -MaxMin 5; if ($LASTEXITCODE -ne 0) { "GATE FAILED"; exit 1 }
$base = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$env:RUST_LOG = "info"
$p = Start-Process -FilePath C:\Build\ai\pyre-tagger\target\x86_64-pc-windows-msvc\release\pyre.exe -ArgumentList $PyreArgs -NoNewWindow -PassThru -RedirectStandardError "C:\Temp\kagami\$Name.err" -RedirectStandardOutput "C:\Temp\kagami\$Name.out"
$peak = 0
while (-not $p.HasExited) { $u = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if ($u -gt $peak) { $peak = $u }; if (((Get-Date) - $t0).TotalSeconds -gt $TimeoutSec) { $p.Kill(); "KILLED (timeout)" }; Start-Sleep -Milliseconds 300 }
$p.WaitForExit()
"exit $($p.ExitCode) | baseline VRAM $base MiB | peak VRAM used $peak MiB (pyre ~ $($peak-$base) MiB) | wall $([int]((Get-Date)-$t0).TotalSeconds)s"
$ev = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$t0} -EA 0 | ? { $_.ProviderName -match 'nvlddmkm|Display' -or $_.Message -match 'TDR|nvlddmkm|display driver' }
if ($ev) { "!!! DRIVER EVENTS:"; $ev | select TimeCreated,ProviderName,Id,Message | fl } else { "no display-driver events in System log since start" }
