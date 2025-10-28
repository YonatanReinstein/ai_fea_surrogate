$process = Start-Process  -FilePath irit64 -ArgumentList "simple_box.irt" -NoNewWindow -PassThru
$process.WaitForExit()
Write-Host "itd file created"
set-location "C:\irit\irit\ntbin64"
irit2inp64.exe "C:\Users\yoyor\OneDrive - Technion\gnn_serogate\simple_box\simple_box.itd" > "C:\Users\yoyor\OneDrive - Technion\gnn_serogate\simple_box\simple_box.inp"
set-location "C:\Users\yoyor\OneDrive - Technion\gnn_serogate\simple_box"
Write-Host "inp file created"