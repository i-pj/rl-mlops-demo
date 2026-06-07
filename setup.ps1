#Requires -Version 5.1
<#
─────────────────────────────────────────────────────────────────
RL MLOps Demo — One-Command Setup

Usage:
  powershell -ExecutionPolicy Bypass -File .\setup.ps1

  or, from PowerShell:
  .\setup.ps1

What it does:
  1. Verifies uv is installed
  2. Installs all Python dependencies via uv
  3. Runs the doctor health check
  4. Prints success or detailed error info

Supports:
  - Windows PowerShell 5.1+
  - PowerShell 7+
  - Offline mode, if model .zip is pre-placed in models/
─────────────────────────────────────────────────────────────────
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Colors ───────────────────────────────────────────────────────
$ESC    = [char]27
$RED    = "${ESC}[0;31m"
$GREEN  = "${ESC}[0;32m"
$YELLOW = "${ESC}[1;33m"
$CYAN   = "${ESC}[0;36m"
$NC     = "${ESC}[0m" # No Color

function Info {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Message)
    Write-Host "$CYANℹ $NC $($Message -join ' ')"
}

function Success {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Message)
    Write-Host "$GREEN✓ $NC $($Message -join ' ')"
}

function Warn {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Message)
    Write-Host "$YELLOW⚠ $NC $($Message -join ' ')"
}

function Fail {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Message)
    Write-Host "$RED✗ $NC $($Message -join ' ')"
    exit 1
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @(),

        [string]$FailureMessage = "Command failed.",

        [switch]$ShowLastLines,

        [int]$LineCount = 5
    )

    if ($ShowLastLines) {
        $output = & $FilePath @Arguments 2>&1
        $exitCode = $LASTEXITCODE

        $output | Select-Object -Last $LineCount

        if ($exitCode -ne 0) {
            Fail $FailureMessage
        }

        return
    }

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Fail $FailureMessage
    }
}

# ── Banner ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "$CYAN╔══════════════════════════════════════════════════════╗$NC"
Write-Host "$CYAN║  🏎️  RL MLOps Demo — Setup                          ║$NC"
Write-Host "$CYAN║  From RL Agent to Production                        ║$NC"
Write-Host "$CYAN╚══════════════════════════════════════════════════════╝$NC"
Write-Host ""

# ── Step 1: Check uv ─────────────────────────────────────────────
Info "Checking for uv …"
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue

if (-not $UvCommand) {
    Fail "uv not found. Install it first:  irm https://astral.sh/uv/install.ps1 | iex"
}

$UvVersion = (& $UvCommand.Source --version 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) {
    Fail "uv was found but could not be executed. Check your PATH or uv installation."
}
Success "uv found: $UvVersion"

# ── Step 2: Navigate to project root ─────────────────────────────
$ScriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}

Set-Location $ScriptDir
Info "Working directory: $ScriptDir"

# ── Step 3: Install dependencies ─────────────────────────────────
Info "Installing dependencies via uv sync …"
Invoke-NativeCommand `
    -FilePath $UvCommand.Source `
    -Arguments @('sync') `
    -FailureMessage "Dependency installation failed. Check uv output above." `
    -ShowLastLines `
    -LineCount 5
Success "Dependencies installed"

# ── Step 4: Verify critical imports ──────────────────────────────
Info "Verifying Python imports …"
$PythonCheck = @'
import torch
import gymnasium
import stable_baselines3
import sb3_contrib
print(f'  torch={torch.__version__}')
print(f'  gymnasium={gymnasium.__version__}')
print(f'  sb3={stable_baselines3.__version__}')

# Check device
if torch.backends.mps.is_available():
    print('  device=MPS (Apple Silicon GPU) ✓')
elif torch.cuda.is_available():
    print(f'  device=CUDA ({torch.cuda.get_device_name()}) ✓')
else:
    print('  device=CPU')
'@

Invoke-NativeCommand `
    -FilePath $UvCommand.Source `
    -Arguments @('run', 'python', '-c', $PythonCheck) `
    -FailureMessage "Import verification failed. Check dependencies."
Success "All imports verified"

# ── Step 5: Doctor check ─────────────────────────────────────────
Info "Running health checks (this may take a moment to load torch) …"
Invoke-NativeCommand `
    -FilePath $UvCommand.Source `
    -Arguments @('run', 'rl-demo', 'doctor') `
    -FailureMessage "Doctor health check failed. Review the output above."

# ── Done ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "$GREEN╔══════════════════════════════════════════════════════╗$NC"
Write-Host "$GREEN║  ✅  Setup complete!                                 ║$NC"
Write-Host "$GREEN╠══════════════════════════════════════════════════════╣$NC"
Write-Host "$GREEN║                                                      ║$NC"
Write-Host "$GREEN║  Quick start:                                        ║$NC"
Write-Host "$GREEN║    uv run rl-demo train --timesteps 50000            ║$NC"
Write-Host "$GREEN║    uv run rl-demo eval -n 5 --clean                  ║$NC"
Write-Host "$GREEN║    uv run rl-demo demo --clean                       ║$NC"
Write-Host "$GREEN║    uv run rl-demo tune --n-trials 5                  ║$NC"
Write-Host "$GREEN║    uv run rl-demo mlflow-ui                          ║$NC"
Write-Host "$GREEN║                                                      ║$NC"
Write-Host "$GREEN╚══════════════════════════════════════════════════════╝$NC"
Write-Host ""
