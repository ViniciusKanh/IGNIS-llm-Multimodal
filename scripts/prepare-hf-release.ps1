param([string]$Adapter = 'ignis-image-lora-v1/final', [switch]$Publish, [switch]$Private)
. "$PSScriptRoot/common.ps1"
$ReleaseArgs = @('scripts/prepare_hf_release.py','--adapter',$Adapter)
if ($Publish) { $ReleaseArgs += '--publish' }
if ($Private) { $ReleaseArgs += '--private' }
Invoke-ProjectPython @ReleaseArgs
