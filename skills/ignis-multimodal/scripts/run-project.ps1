param(
    [ValidateSet('bootstrap','download-models','train-lora','evaluate','start','status')]
    [string]$Action = 'status',
    [string]$Project = 'C:\dev\Ignis-Multimodal',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
$script = Join-Path $Project "scripts/$Action.ps1"
if (-not (Test-Path -LiteralPath $script)) { throw "Script inexistente: $script" }
& $script @ExtraArgs
