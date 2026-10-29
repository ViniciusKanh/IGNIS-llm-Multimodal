param(
    [string]$Project = 'C:\dev\Ignis-Multimodal',
    [switch]$Publish,
    [switch]$Private
)
$ErrorActionPreference = 'Stop'
$SourceRoot = Split-Path -Parent $PSScriptRoot
$Project = [IO.Path]::GetFullPath($Project)
$PythonExe = Join-Path $Project '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Instale o ambiente do projeto com bootstrap.ps1 antes de continuar.' }
if ($Project -ne $SourceRoot) {
    # Atualiza apenas os fontes; preserva pesos, dados, checkpoints e configuração do experimento.
    $BackupRoot = Join-Path $Project ('backups/publication-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff'))
    foreach ($Folder in @('backend','scripts','huggingface')) {
        $SourceFolder = Join-Path $SourceRoot $Folder
        foreach ($File in Get-ChildItem -LiteralPath $SourceFolder -Recurse -File) {
            if ($File.FullName -match '[\\/]__pycache__[\\/]') { continue }
            $Relative = $File.FullName.Substring($SourceRoot.Length + 1)
            $Target = Join-Path $Project $Relative
            if (Test-Path -LiteralPath $Target) {
                if ((Get-FileHash -LiteralPath $File.FullName).Hash -eq (Get-FileHash -LiteralPath $Target).Hash) { continue }
                $Backup = Join-Path $BackupRoot $Relative
                New-Item -ItemType Directory -Path (Split-Path -Parent $Backup) -Force | Out-Null
                Copy-Item -LiteralPath $Target -Destination $Backup
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
            Copy-Item -LiteralPath $File.FullName -Destination $Target -Force
        }
    }
    foreach ($Name in @('README.md','requirements.txt')) {
        $Target = Join-Path $Project $Name
        if (Test-Path -LiteralPath $Target) {
            New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
            Copy-Item -LiteralPath $Target -Destination (Join-Path $BackupRoot $Name)
        }
        Copy-Item -LiteralPath (Join-Path $SourceRoot $Name) -Destination $Target -Force
    }
}
Set-Location -LiteralPath $Project
$env:PIP_CACHE_DIR = Join-Path $Project '.cache/pip'
$env:npm_config_cache = Join-Path $Project '.cache/npm'
$env:PYTHONUTF8 = '1'
function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Etapa interrompida (Python exit code $LASTEXITCODE). Consulte a saída acima; os checkpoints foram preservados." }
}
Write-Host '[1/7] Conferindo ambiente CUDA e dependências de visão.'
Invoke-CheckedPython -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponivel no ambiente Python do projeto'; print('PyTorch',torch.__version__,'CUDA',torch.version.cuda)"
$VisionProbe = @'
try:
    import torchvision
    print('Torchvision:', torchvision.__version__)
except Exception as error:
    print('Dependencia de visao requer ajuste:', error)
    raise SystemExit(1)
'@
& $PythonExe -c $VisionProbe
if ($LASTEXITCODE -ne 0) {
    $TorchVersion = (& $PythonExe -c "import torch; print(torch.__version__)").Trim()
    Invoke-CheckedPython -m pip install --retries 1 --timeout 40 "torch==$TorchVersion" 'torchvision>=0.22,<1' --index-url https://download.pytorch.org/whl/cu128
}
Invoke-CheckedPython -c "import torchvision; print('Visao preparada:', torchvision.__version__)"
Write-Host '[2/7] Conferindo modelos e retomando apenas os downloads necessários.'
Invoke-CheckedPython scripts/check_installed_models.py
Write-Host '[3/7] Preparando dataset quando ainda não houver manifesto.'
if (-not (Test-Path -LiteralPath 'datasets/manifest.json')) { Invoke-CheckedPython scripts/prepare_dataset.py }
Write-Host '[4/7] Treinando ou retomando o adaptador.'
Invoke-CheckedPython scripts/train_lora.py --no-evaluate
Write-Host '[5/7] Comparando modelo-base e adaptador final.'
$TrainingConfig = Get-Content -LiteralPath 'configs/train.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$Adapter = "$($TrainingConfig.name)/final"
if (Test-Path -LiteralPath 'training/latest-run.json') {
    $LatestRun = Get-Content -LiteralPath 'training/latest-run.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    $Adapter = $LatestRun.adapter
}
Invoke-CheckedPython scripts/evaluate.py --adapter $Adapter
Write-Host '[6/7] Compilando a interface local.'
Push-Location frontend
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Falha na compilação do frontend.' }
} finally { Pop-Location }
Write-Host '[7/7] Gerando modelo, documentação, amostras e código do Space.'
$ReleaseArguments = @('scripts/prepare_hf_release.py','--adapter',$Adapter)
if ($Publish) { $ReleaseArguments += '--publish' }
if ($Private) { $ReleaseArguments += '--private' }
Invoke-CheckedPython @ReleaseArguments
Write-Host 'Pacotes disponíveis em outputs/hf-release/model e outputs/hf-release/space.'
if (-not $Publish) { Write-Host 'Preparação concluída. Nenhum repositório remoto foi publicado por este comando.' }
