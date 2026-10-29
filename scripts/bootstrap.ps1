param([switch]$SkipFrontend, [switch]$SkipTorch)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$PythonExe = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache/pip'
$env:npm_config_cache = Join-Path $ProjectRoot '.cache/npm'
$env:TEMP = Join-Path $ProjectRoot '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
if (-not (Test-Path -LiteralPath $PythonExe)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar ambiente Python.' }
}
if (-not $SkipTorch) {
    & $PythonExe -m pip install --retries 0 --timeout 20 'torch>=2.7,<3' 'torchvision>=0.22,<1' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Não foi possível instalar PyTorch CUDA. Verifique o acesso a download.pytorch.org.' }
}
& $PythonExe -m pip install --retries 0 --timeout 20 -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependências; consulte a saída do pip.' }
& $PythonExe -m pip freeze | Set-Content -LiteralPath 'logs-dependencies.txt' -Encoding utf8
if (-not $SkipFrontend) {
    Push-Location frontend
    try {
        & npm.cmd install --fetch-retries=0 --fetch-timeout=20000
        if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar frontend.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Falha ao compilar frontend.' }
    } finally { Pop-Location }
}
& $PythonExe -c "import torch; print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available())"
Write-Host 'Ambiente preparado. Baixe os modelos com scripts/download-models.ps1.'
