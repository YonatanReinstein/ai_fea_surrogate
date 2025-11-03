param(
    [string]$irtfile
)
# Save current directory
$workingDir = Get-Location
$process = Start-Process  -FilePath irit64 -ArgumentList "data\$irtfile.irt" -NoNewWindow -PassThru
$process.WaitForExit()
Write-Host "itd file created"
set-location "C:\irit\irit\ntbin64"
irit2inp64.exe -s 2 2 10 "$workingDir\tmp\$irtfile.itd" > "$workingDir\tmp\$irtfile.inp"
set-location $workingDir
Write-Host "inp file created"