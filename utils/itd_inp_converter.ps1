param(
    [string]$U,
    [string]$V,
    [string]$W
)

$workingDir = Get-Location
set-location "C:\irit\irit\ntbin64"
# irit2inp64.exe -s 2 2 10 "$workingDir\model.itd" > "$workingDir\model.inp"
# irit2inp64.exe -s 10 10 10 "$workingDir\model.itd" > "$workingDir\model.inp"
irit2inp64.exe -s $U $V $W "$workingDir\model.itd" > "$workingDir\model.inp"
set-location $workingDir
Write-Host "inp file created"