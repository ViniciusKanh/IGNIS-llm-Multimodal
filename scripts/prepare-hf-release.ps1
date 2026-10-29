param([string]$Adapter = '', [switch]$Publish, [switch]$Private)
. "$PSScriptRoot/common.ps1"
$ReleaseArgs = @('scripts/prepare_hf_release.py')
if ($Adapter) { $ReleaseArgs += @('--adapter',$Adapter) }
if ($Publish) { $ReleaseArgs += '--publish' }
if ($Private) { $ReleaseArgs += '--private' }
Invoke-ProjectPython @ReleaseArgs
