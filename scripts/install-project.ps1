param(
    [string]$Destination = 'C:\dev\Ignis-Multimodal',
    [string]$SkillDestination = 'C:\Users\Vinicius Santos\.codex\skills\ignis-multimodal'
)
$ErrorActionPreference = 'Stop'
$SourceRoot = Split-Path -Parent $PSScriptRoot
$Destination = [IO.Path]::GetFullPath($Destination)
$SkillDestination = [IO.Path]::GetFullPath($SkillDestination)
if ($Destination -eq $SourceRoot) { throw 'Origem e destino são iguais; a skill pode ser copiada separadamente.' }
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
foreach ($Entry in Get-ChildItem -LiteralPath $SourceRoot -Force) {
    if ($Entry.Name -in @('.git','.venv','.cache','__pycache__','.pytest_cache','models','datasets','training','outputs','logs')) { continue }
    if ($Entry.Name -in @('AGENTS.md','README.md') -and (Test-Path -LiteralPath (Join-Path $Destination $Entry.Name))) { continue }
    Copy-Item -LiteralPath $Entry.FullName -Destination $Destination -Recurse -Force
}
if (Test-Path -LiteralPath $SkillDestination) { throw "Skill já existe; revise antes de substituir: $SkillDestination" }
Copy-Item -LiteralPath (Join-Path $SourceRoot 'skills/ignis-multimodal') -Destination $SkillDestination -Recurse
& git -C $SkillDestination init
if ($LASTEXITCODE -ne 0) { throw 'Falha ao versionar a skill.' }
& git -C $SkillDestination add .
& git -C $SkillDestination -c user.name='Codex' -c user.email='codex@localhost' commit -m 'Criar skill ignis-multimodal'
if ($LASTEXITCODE -ne 0) { throw 'Falha ao registrar a skill no Git.' }
Write-Host "Projeto instalado em $Destination; skill em $SkillDestination."
