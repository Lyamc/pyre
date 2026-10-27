# Wait (up to $MaxMin minutes) until no other pyre*/cabin* process runs and the GPU is quiet; exit 1 if it never is.
param([int]$MaxMin = 20)
for ($i = 0; $i -lt $MaxMin * 4; $i++) {
  $p = Get-Process pyre*, cabin*, strata* -EA SilentlyContinue
  $u = [int](nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if (-not $p -and $u -lt 40) { "gpu free (util $u%)"; exit 0 }
  Start-Sleep 15
}
"gpu busy: $($p | % { $_.Name + ':' + $_.Id }) util $u%"; exit 1
