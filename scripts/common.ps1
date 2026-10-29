$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw 'Ambiente não preparado. Execute .\scripts\bootstrap.ps1.'
}
Set-Location -LiteralPath $ProjectRoot
$env:HF_HOME = Join-Path $ProjectRoot '.cache/huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME 'datasets'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache/torch'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache/pip'
$env:npm_config_cache = Join-Path $ProjectRoot '.cache/npm'
function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falha no comando Python (código $LASTEXITCODE)." }
}
