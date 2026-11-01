<#  pyre-service.ps1 - OPTIONAL. Shipped, never run by the installer. Run elevated to (un)register a boot-time
    background server as a Scheduled Task (runs pyre-launch.ps1 -NoFirstRun as SYSTEM at startup, hidden, restarts on failure).
    pyre.exe is not a Windows-service binary, so a task is used instead of a wrapper like WinSW/nssm.
    All paths come from %ProgramData%\pyre\config.toml (SYSTEM has its own %LOCALAPPDATA%, so keep models_dir explicit).
      pyre-service.ps1 -Install | -Uninstall | -Status
#>
param([switch]$Install, [switch]$Uninstall, [switch]$Status, [string]$Name = 'pyre-server')
$launch = Join-Path $PSScriptRoot 'pyre-launch.ps1'
if ($Install) {
  $act = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launch`" -NoFirstRun" -WorkingDirectory $PSScriptRoot
  $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
  Register-ScheduledTask -TaskName $Name -Action $act -Trigger (New-ScheduledTaskTrigger -AtStartup) -Settings $set `
    -Principal (New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest) -Force | Out-Null
  "registered task $Name (start now: Start-ScheduledTask $Name)"
} elseif ($Uninstall) { Unregister-ScheduledTask -TaskName $Name -Confirm:$false; "removed $Name" }
else { Get-ScheduledTask -TaskName $Name -ea 0 | Select-Object TaskName, State }
