# Git and training server workflow

## Source of truth

The local `main` branch is the development source of truth. The expected flow is:

```text
local edit -> local test -> git commit -> git push origin main -> server git pull --ff-only
```

Do not edit tracked source files directly on the training server. `scripts/sync-server.ps1` refuses to pull when the server working tree has uncommitted changes, preventing accidental loss or divergence.

## One-time server preparation

1. Configure SSH access from the local Windows machine to the training server.
2. Configure the server to read the private GitHub repository. Prefer a repository-scoped GitHub deploy key. Do not store private keys or tokens in this repository.
3. Copy `.server.json.example` to `.server.json` locally and fill in the SSH target and server repository path. `.server.json` is ignored by Git.
4. From the local repository, run:

```powershell
.\scripts\sync-server.ps1
```

On first use the script clones the repository. Later runs fetch and fast-forward the configured branch.

## Starting an experiment on the server

After synchronizing, connect to the server and run commands through the experiment wrapper:

```bash
cd /path/to/copper-foil-defect-classification
./scripts/start-experiment.sh csd-baseline -- bash train_csd.sh
```

Each run is stored under `experiments/<UTC timestamp>-<name>/` and records:

- `git_commit.txt`: exact source revision used by the experiment
- `git_remote.txt`: source repository URL
- `command.sh`: executed command
- `pid.txt`: background process ID
- `started_at.txt` and `finished_at.txt`: UTC timestamps
- `stdout.log` and `stderr.log`: complete process output streams
- `exit_code.txt`: process exit code after completion

The `experiments/latest` symbolic link points to the newest run. The `experiments/` directory is excluded from Git.

The wrapper also sets `OUTPUT_DIR` to that same run directory before starting the command,
so checkpoints, TensorBoard logs, `preprocessing_reports/` and `eval-class-acc.txt` are written
beside the run's own logs rather than into a second output tree. Do not set `OUTPUT_DIR`
yourself when launching through the wrapper.

Training parameters that are expected to be tuned per run are read from the environment by
`train_csd.sh`, so they can be adjusted without editing tracked files on the server:

```bash
BATCH_SIZE=64 EPOCHS=100 ./scripts/start-experiment.sh csd-baseline -- bash train_csd.sh
```

## Viewing logs locally

List experiment IDs:

```powershell
.\scripts\remote-logs.ps1 list
```

Read the last 200 stderr lines:

```powershell
.\scripts\remote-logs.ps1 stderr -RunId 20260914T120000Z-csd-baseline
```

Follow stderr continuously. Press Ctrl+C to stop:

```powershell
.\scripts\remote-logs.ps1 follow-stderr -RunId 20260914T120000Z-csd-baseline
```

Download all metadata and logs for a run:

```powershell
.\scripts\remote-logs.ps1 download -RunId 20260914T120000Z-csd-baseline
```

Downloaded files are placed in `downloaded-logs/`, which is also excluded from Git.

## Recommended iteration discipline

- Give each experiment a meaningful name.
- Commit and push before server synchronization.
- Record dataset version and configuration in the experiment name or an external registry.
- Compare results only after confirming `git_commit.txt` and the dataset version.
- Keep model checkpoints and datasets outside Git; use dedicated artifact storage or server disks.
