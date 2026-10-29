param([string]$Adapter = 'ignis-image-lora-v1/final')
. "$PSScriptRoot/common.ps1"
Invoke-ProjectPython scripts/evaluate.py --adapter $Adapter
