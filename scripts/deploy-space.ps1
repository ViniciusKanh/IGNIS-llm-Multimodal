param([string]$Project = 'C:\dev\Ignis-Multimodal', [switch]$PrepareOnly)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $Project '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Ambiente Python do projeto não encontrado.' }
$env:HF_HOME = Join-Path $Project '.cache/huggingface'
$env:PYTHONUTF8 = '1'
$Parameters = @((Join-Path $PSScriptRoot 'publish_space.py'), '--project-root', $Project)
if ($PrepareOnly) { $Parameters += '--prepare-only' }
& $Python @Parameters
if ($LASTEXITCODE -ne 0) { throw 'A publicação do Space não foi concluída. Confira o erro acima; o modelo publicado permanece preservado.' }
