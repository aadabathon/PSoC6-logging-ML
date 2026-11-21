[CmdletBinding(DefaultParameterSetName = 'Setup')]
param(
    [Parameter(ParameterSetName = 'Study', Mandatory = $true)][string]$Study,
    [Parameter(ParameterSetName = 'Snapshot', Mandatory = $true)][string]$Experiment,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$movementPython = Join-Path $PSScriptRoot 'venv/Scripts/pythonw.exe'
if (-not (Test-Path -LiteralPath $movementPython -PathType Leaf)) {
    throw 'Set up the analysis venv first. See README.md for commands.'
}
$movementArguments = @((Join-Path $PSScriptRoot 'movement_study.py'))
if ($PSCmdlet.ParameterSetName -eq 'Study') {
    $movementArguments += @('--study', (Resolve-Path -LiteralPath $Study).Path)
} elseif ($PSCmdlet.ParameterSetName -eq 'Snapshot') {
    $movementArguments += @('--experiment', (Resolve-Path -LiteralPath $Experiment).Path)
}
if ($Check) {
    @{ executable = $movementPython; arguments = $movementArguments } | ConvertTo-Json
    return
}
$quotedArguments = @($movementArguments | ForEach-Object { '"' + $_ + '"' })
Start-Process -FilePath $movementPython -ArgumentList $quotedArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
