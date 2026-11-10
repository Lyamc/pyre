# Start, stop, or inspect the pyre server. Forwards every argument to installer\pyre-launch.ps1.
# Listens on 0.0.0.0 (all interfaces) unless config or -Host says otherwise.
#   .\pyre.ps1 -Model D:\models\Qwen3.8-Flash-Next-IQ3_S -Quant IQ3_S
#   .\pyre.ps1 -Host 127.0.0.1
#   .\pyre.ps1 -Foreground
#   .\pyre.ps1 -Stop
#   .\pyre.ps1 -DryRun
& (Join-Path $PSScriptRoot 'installer\pyre-launch.ps1') @args
