param(
    [string]$irt_model_path
)
# Save current directory
$workingDir = Get-Location
Write-Host "Current Directory: $workingDir"
$process = Start-Process  -FilePath irit64 -ArgumentList $irt_model_path -NoNewWindow -PassThru 
$process.WaitForExit()
Write-Host "CAD model volume calculated"
Write-Host "itd file created"
