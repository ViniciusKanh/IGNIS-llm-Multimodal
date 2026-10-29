param([ValidateSet('all','smolvlm','qwen','sd15')][string[]]$Models = @('all'))
. "$PSScriptRoot/common.ps1"
Invoke-ProjectPython scripts/download_models.py --models @Models
