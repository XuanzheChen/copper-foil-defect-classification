[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot "..\.server.json")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Server config not found: $ConfigPath. Copy .server.json.example to .server.json and edit it first."
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
foreach ($field in @("sshTarget", "repositoryPath", "repositoryUrl", "branch")) {
    if ([string]::IsNullOrWhiteSpace($config.$field)) {
        throw "Missing required server config field: $field"
    }
}

function Quote-Bash([string]$Value) {
    if ($Value.Contains("'")) {
        throw "Server configuration values may not contain a single quote."
    }
    return "'$Value'"
}

$repoPath = Quote-Bash $config.repositoryPath
$repoUrl = Quote-Bash $config.repositoryUrl
$branch = Quote-Bash $config.branch

$remoteScript = @"
set -e
if [ ! -d $repoPath/.git ]; then
  mkdir -p `$(dirname $repoPath)
  git clone --branch $branch $repoUrl $repoPath
fi
cd $repoPath
if [ -n "`$(git status --porcelain)" ]; then
  echo "ERROR: server working tree is dirty; refusing to overwrite local changes" >&2
  git status --short >&2
  exit 2
fi
git fetch origin $branch
git checkout $branch
git pull --ff-only origin $branch
printf 'SERVER_COMMIT='
git rev-parse HEAD
"@

Write-Host "Synchronizing $($config.sshTarget):$($config.repositoryPath)"
& ssh $config.sshTarget $remoteScript
if ($LASTEXITCODE -ne 0) {
    throw "Server synchronization failed with exit code $LASTEXITCODE"
}
