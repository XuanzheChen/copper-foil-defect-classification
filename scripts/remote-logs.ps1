[CmdletBinding()]
param(
    [ValidateSet("list", "stdout", "stderr", "follow-stdout", "follow-stderr", "download")]
    [string]$Action = "list",
    [string]$RunId,
    [int]$Lines = 200,
    [string]$ConfigPath = (Join-Path $PSScriptRoot "..\.server.json")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Server config not found: $ConfigPath. Copy .server.json.example to .server.json and edit it first."
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
foreach ($field in @("sshTarget", "repositoryPath")) {
    if ([string]::IsNullOrWhiteSpace($config.$field)) {
        throw "Missing required server config field: $field"
    }
}

if ($Action -ne "list" -and [string]::IsNullOrWhiteSpace($RunId)) {
    throw "-RunId is required for action '$Action'."
}
if ($RunId -and $RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId may contain only letters, digits, dot, underscore and hyphen."
}

function Quote-Bash([string]$Value) {
    if ($Value.Contains("'")) {
        throw "Server configuration values may not contain a single quote."
    }
    return "'$Value'"
}

$repoPath = Quote-Bash $config.repositoryPath
$runRoot = "$repoPath/experiments"

switch ($Action) {
    "list" {
        & ssh $config.sshTarget "if [ -d $runRoot ]; then find $runRoot -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -r; fi"
    }
    "stdout" {
        & ssh $config.sshTarget "tail -n $Lines $runRoot/$RunId/stdout.log"
    }
    "stderr" {
        & ssh $config.sshTarget "tail -n $Lines $runRoot/$RunId/stderr.log"
    }
    "follow-stdout" {
        & ssh $config.sshTarget "tail -n $Lines -F $runRoot/$RunId/stdout.log"
    }
    "follow-stderr" {
        & ssh $config.sshTarget "tail -n $Lines -F $runRoot/$RunId/stderr.log"
    }
    "download" {
        $destination = Join-Path (Join-Path $PSScriptRoot "..\downloaded-logs") $RunId
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        & scp "$($config.sshTarget):$($config.repositoryPath)/experiments/$RunId/*" $destination
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Logs downloaded to $destination"
        }
    }
}

if ($LASTEXITCODE -ne 0) {
    throw "Remote log command failed with exit code $LASTEXITCODE"
}
