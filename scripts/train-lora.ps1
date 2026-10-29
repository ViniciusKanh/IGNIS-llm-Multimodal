param([int]$Steps = 0, [ValidateSet(0,256,384,512)][int]$Resolution = 0, [string]$Name = '', [switch]$PrepareDataset)
. "$PSScriptRoot/common.ps1"
if ($PrepareDataset -or -not (Test-Path -LiteralPath 'datasets/manifest.json')) {
    Invoke-ProjectPython scripts/prepare_dataset.py
}
$TrainArgs = @('scripts/train_lora.py')
if ($Steps -gt 0) { $TrainArgs += @('--steps', "$Steps") }
if ($Resolution -gt 0) { $TrainArgs += @('--resolution', "$Resolution") }
if ($Name) { $TrainArgs += @('--name', $Name) }
Invoke-ProjectPython @TrainArgs
