param([string]$Destination = 'C:\dev\Ignis-Multimodal')
$ErrorActionPreference = 'Stop'
$SourceRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $Destination 'backend/app.py'))) {
    throw 'Instale primeiro o projeto base com install-project.ps1.'
}
if ([IO.Path]::GetFullPath($Destination) -eq $SourceRoot) {
    Write-Host 'Os arquivos de publicação já estão nesta cópia do projeto.'
    return
}
Copy-Item -LiteralPath (Join-Path $SourceRoot 'huggingface') -Destination $Destination -Recurse -Force
foreach ($Name in @('prepare_hf_release.py','prepare-hf-release.ps1','sync-publication.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $Name) -Destination (Join-Path $Destination 'scripts') -Force
}
$Readme = Join-Path $Destination 'README.md'
if (Test-Path -LiteralPath $Readme) {
    $Backup = Join-Path $Destination ('README-before-publication-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff') + '.md')
    Copy-Item -LiteralPath $Readme -Destination $Backup
}
Copy-Item -LiteralPath (Join-Path $SourceRoot 'README.md') -Destination $Readme -Force
Write-Host 'Model card, Space e scripts de publicação atualizados. O README anterior foi preservado.'
