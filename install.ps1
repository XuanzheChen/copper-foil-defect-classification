# Copper standalone installer (Windows PowerShell 5.1+ / PowerShell 7+).
# git must already be installed and authorized to access the private repository.
[CmdletBinding()]
param(
    [string]$Target,
    [string]$RepoUrl,
    [ValidateSet('auto', 'cpu', 'cu118', 'cu126', 'cu128')]
    [string]$Backend = 'auto',
    [int]$GpuIndex = 0,
    [switch]$Update,
    [switch]$RecreateEnv,
    [switch]$DryRun,
    [switch]$NoUvBootstrap
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -eq 'Core' -and -not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    throw 'install.ps1 supports Windows only. On Linux use install.sh.'
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is required and must be authorized to access the private Copper repository.'
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvPath = if ($uvCommand) { $uvCommand.Source } else { $null }
if (-not $uvPath) {
    $candidate = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $candidate) { $uvPath = $candidate }
}
if (-not $uvPath) {
    if ($NoUvBootstrap -or $env:COPPER_NO_UV_BOOTSTRAP -eq '1') {
        throw 'uv missing. Install it from https://docs.astral.sh/uv/ or allow official uv bootstrap.'
    }
    $tempScript = Join-Path $env:TEMP ("copper-uv-installer-{0}.ps1" -f [guid]::NewGuid())
    try {
        Write-Host 'Downloading official uv installer from astral.sh into a temporary file...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $env:UV_NO_MODIFY_PATH = '1'
        Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile $tempScript -UseBasicParsing
        # The execution-policy bypass applies to this child process only.
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tempScript
        if ($LASTEXITCODE -ne 0) { throw "uv installer failed with exit code $LASTEXITCODE" }
    } finally {
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
    $uvPath = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uvPath)) {
        throw "uv installer did not create the expected executable: $uvPath"
    }
}

& $uvPath python install 3.10
if ($LASTEXITCODE -ne 0) { throw "uv python install failed with exit code $LASTEXITCODE" }
$managedPython = [string](& $uvPath python find --managed-python 3.10 | Select-Object -Last 1)
$managedPython = $managedPython.Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $managedPython)) {
    throw 'Could not locate uv-managed Python 3.10.'
}
$env:COPPER_UV = $uvPath
$arguments = @((Join-Path $PSScriptRoot 'scripts\install\bootstrap.py'), '--backend', $Backend, '--gpu-index', "$GpuIndex")
if ($Target) { $arguments += @('--target', $Target) }
if ($RepoUrl) { $arguments += @('--repo-url', $RepoUrl) }
if ($Update) { $arguments += '--update' }
if ($RecreateEnv) { $arguments += '--recreate-env' }
if ($DryRun) { $arguments += '--dry-run' }
& $managedPython @arguments
if ($LASTEXITCODE -ne 0) { throw "Copper installation failed with exit code $LASTEXITCODE" }
