param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('collect', 'train', 'train_forever', 'evaluate', 'eval_batch', 'cache_frames',
                 'split_stages', 'validate_rgb', 'validate_shards', 'merge', 'audit', 'summarize',
                 'repro', 'foresight', 'finetune', 'compare', 'test')]
    [string]$Command,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)
# Thin launcher: resolves the interpreter and forwards arguments to `python -m charger.<Command>`.
# No machine-specific paths are hardcoded. Set CHARGER_PYTHON to override the interpreter.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($env:CHARGER_PYTHON) { $env:CHARGER_PYTHON }
          elseif (Test-Path (Join-Path $root '.venv\Scripts\python.exe')) { Join-Path $root '.venv\Scripts\python.exe' }
          else { 'python' }
# SAPIEN's CPU backend is used for simulation; keep CPU thread pools small so several
# evaluation shards can run at once without oversubscribing the machine.
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { '1' }
if ($Command -eq 'test') {
    & $python -m unittest discover -s tests -v
} else {
    & $python -u -m "charger.$Command" @Arguments
}
exit $LASTEXITCODE
