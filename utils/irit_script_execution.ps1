param(
    [string]$irt_model_path
)
# Save current directory

$process = Start-Process `
    -FilePath "irit64" `
    -ArgumentList "`"$irt_model_path`"" `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput "NUL" `
    -RedirectStandardError "err.txt"
$process.WaitForExit()

