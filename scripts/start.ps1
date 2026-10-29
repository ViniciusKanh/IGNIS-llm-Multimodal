param([switch]$Dev)
. "$PSScriptRoot/common.ps1"
if ($Dev) {
    $NodeExe = (Get-Command node.exe).Source
    $Vite = Join-Path $ProjectRoot 'frontend/node_modules/vite/bin/vite.js'
    if (-not (Test-Path -LiteralPath $Vite)) { throw 'Instale o frontend com bootstrap.ps1.' }
    $FrontendProcess = Start-Process -FilePath $NodeExe -ArgumentList @("`"$Vite`"",'--host','127.0.0.1') -WorkingDirectory (Join-Path $ProjectRoot 'frontend') -WindowStyle Hidden -PassThru
    Write-Host 'Interface: http://127.0.0.1:5173  API: http://127.0.0.1:8000/docs'
    try { Invoke-ProjectPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 }
    finally { if (-not $FrontendProcess.HasExited) { Stop-Process -Id $FrontendProcess.Id } }
} else {
    if (-not (Test-Path -LiteralPath 'frontend/dist/index.html')) { throw 'Frontend não compilado. Execute bootstrap.ps1.' }
    Write-Host 'IGNIS Multimodal: http://127.0.0.1:8000'
    Invoke-ProjectPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
}
