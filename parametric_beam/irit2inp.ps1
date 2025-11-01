$process = Start-Process  -FilePath irit64 -ArgumentList "beam.irt" -NoNewWindow -PassThru
$process.WaitForExit()
Write-Host "itd file created"
set-location "C:\irit\irit\ntbin64"
irit2inp64.exe -s 2 2 10 "C:\Users\yoyor\OneDrive - Technion\gnn_serogate\parametric_beam\beam.itd" > "C:\Users\yoyor\OneDrive - Technion\gnn_serogate\parametric_beam\beam.inp"
set-location "C:\Users\yoyor\OneDrive - Technion\gnn_serogate\parametric_beam"
Write-Host "inp file created"