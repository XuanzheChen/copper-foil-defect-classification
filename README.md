# Copper Foil Defect Classification (Standalone)

Cross-platform, standalone **200 × 200 grayscale copper-foil image classification** based on ConvNeXt V2. This branch is intended for independent local or server use **without the original experiment archive, SSH synchronization, experiment wrapper, or fixed server paths**.

> **Branch:** `portable/standalone` (maintained separately from `main`). The existing scientific preprocessing and model/checkpoint semantics are preserved. An automatic Git/CUDA-aware installer is included for Windows and Linux; manual installation remains available as a fallback.

## 0. Quick start: automatic installation

The repository is **private**: Git must already be installed and authenticated to GitHub (SSH key or HTTPS credential manager). The launcher can automatically install user-level `uv` and managed Python 3.10, create an isolated project `.venv`, choose PyTorch `cu118`, `cu126`, `cu128`, or CPU according to NVIDIA GPU compute capability and driver requirements, and run an actual model-forward smoke test. It does not modify your system Python or Conda environment.

**Linux (Bash):**

```bash
git clone --depth 1 --single-branch --branch portable/standalone git@github.com:XuanzheChen/copper-foil-defect-classification.git
bash copper-foil-defect-classification/install.sh
```

**Windows (PowerShell):**

```powershell
git clone --depth 1 --single-branch --branch portable/standalone git@github.com:XuanzheChen/copper-foil-defect-classification.git
if ($LASTEXITCODE -eq 0) {
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\copper-foil-defect-classification\install.ps1
}
```

**Already cloned?** In the `portable/standalone` repository root run `bash install.sh` on Linux, or `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1` on Windows.

- For explicit CPU installation: `bash install.sh --backend cpu` / `.\install.ps1 -Backend cpu`
- To clone directly into another directory: `bash install.sh --target ~/copper-new` / `.\install.ps1 -Target D:\copper-new`
- To safely update the standalone branch: `bash install.sh --update` / `.\install.ps1 -Update`
- To preview automatic CUDA selection: `bash install.sh --dry-run` / `.\install.ps1 -DryRun` (the launchers may still provision uv/Python)
- To switch CPU/CUDA wheel profiles: add `--recreate-env` / `-RecreateEnv` (only Copper-managed virtual environments are eligible)
- If GitHub HTTPS authentication is configured instead of SSH, use the HTTPS clone URL and optionally `--repo-url` / `-RepoUrl` when cloning through the installer.

Installation reports are written to `.venv/copper-install-report.json`. The installer does not download datasets or trained checkpoints. See **[INSTALLER.md](INSTALLER.md)** for the full platform support table, driver thresholds, private repository authentication, safety rules and troubleshooting. The manual instructions in section 2 are kept for users who cannot run the automatic launcher.

## 1. Capabilities and entry points

| Task | Cross-platform entry point | Notes |
| --- | --- | --- |
| Supervised training | `python scripts/copper.py train` | Retains existing 200×200 preprocessing, augmentation, and checkpoint selection |
| Evaluation on train/val/test | `python scripts/copper.py eval` | Reads architecture, class count, input size from the checkpoint |
| Folder/batch inference | `python scripts/copper.py infer` | Creates a SHA-256 dataset snapshot, then runs Top-k inference |
| Dataset organization | `python scripts/copper.py organize` | Copies by Top-1, preserves originals, verifies snapshot and predictions |
| Single-image inference | `python infer_topk.py` | Direct low-level entry point |
| Minimal seven-class release | `deployment/minimal/` | Optional model-specific packaging tooling |
| Optional class-loss and hard-example controls | `main_finetune.py` | Advanced training options retained in the core |

The `scripts/copper.py` front end invokes the existing model implementation rather than forking training or inference logic. The core model definitions remain in `models/self/convnextv2_modified.py`.

## 2. Hardware and installation

- **OS:** Windows (PowerShell) or Linux (Bash); x86-64 recommended.
- **Python:** 3.10 is the recommended baseline.
- **GPU:** NVIDIA CUDA capable GPU is recommended for training; CPU inference is supported. The actual PyTorch wheel must support both the GPU architecture and the installed driver.
- **Dependencies:** Install PyTorch/TorchVision first, then `requirements.txt`. The requirements file intentionally **does not pin a universal CUDA build**.
- **Dataset, trained checkpoints, and pretrained weights are NOT included in Git.**

### Linux (Bash)

```bash
git clone --depth 1 --single-branch --branch portable/standalone \
  git@github.com:XuanzheChen/copper-foil-defect-classification.git
cd copper-foil-defect-classification
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For a CUDA 12.8 PyTorch wheel on a compatible NVIDIA system (the original RTX 5080 reference profile):

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For CPU-only mode instead, install the CPU wheel:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
git clone --depth 1 --single-branch --branch portable/standalone git@github.com:XuanzheChen/copper-foil-defect-classification.git
cd copper-foil-defect-classification
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For a compatible NVIDIA/CUDA 12.8 configuration:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Or for CPU-only mode:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If PowerShell refuses script activation, use `.\.venv\Scripts\python.exe` in the commands below instead of changing execution policy.

**Important:** `nvidia-smi`'s "CUDA Version" is the maximum API version exposed by the driver; it is not a requirement to install that exact local CUDA Toolkit. Do not blindly install `cu128` on every NVIDIA card. Select a driver-compatible wheel supported by that GPU, or choose CPU wheels. If installation fails, consult the [official PyTorch installer](https://pytorch.org/get-started/locally/). These are installation templates, not a claim that every Windows/Linux/GPU combination has been validated.

### Verify environment

```bash
python -c "import torch,torchvision,timm; print('torch',torch.__version__,'vision',torchvision.__version__,'timm',timm.__version__,'cuda',torch.cuda.is_available())"
python -m unittest discover -s tests -v
python scripts/copper.py --help
```

On NVIDIA systems, optionally run:

```bash
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(4,device='cuda').sum().item())"
```

**Security:** this is a private repository. Authenticate with an SSH deploy key or Git Credential Manager. Do not embed access tokens in scripts or commit them. Windows users may substitute the authenticated HTTPS clone URL.

## 3. Dataset and class labels

Use a directory with `train/`, `val/`, and `test/` subdirectories. Each split uses exactly the **same classes** and directory names:

```text
data/copper/
  train/
    凹凸点/
      image001.png
    刮伤/
      image002.png
    ...
  val/
    凹凸点/
    刮伤/
    ...
  test/
    凹凸点/
    刮伤/
    ...
configs/class_labels_csd.txt
```

Label files contain one sequential `index:class_name` record per line, starting from zero, e.g.:

```text
0:class_a
1:class_b
```

**Class index order must match Python/ImageFolder's lexicographic folder order.** The wrapper checks the training and requested evaluation splits against the label file, including ordering. The repository has a nine-class example `configs/class_labels_csd.txt`, and another ten-class label file; the minimal deployment template is specific to a historical seven-class checkpoint. **Do not mix these configurations.**

For the current supervised models, eligible images must be exactly **200×200 pixels**. Nonconforming images are excluded and reported; the pipeline does **not resize** them. It decodes grayscale, expands to three channels, and applies ImageNet normalization. Training additionally applies its configured translation and brightness/contrast augmentation. Do not change these transforms casually when using existing checkpoints.

The training/eval wrapper needs `train/` and `val/` for training; evaluation currently also requires `train/` to exist because the underlying `main_finetune.py` initializes a training dataset even in evaluation-only mode. Use a real matching train split, not a fabricated dataset.

## 4. Training

The commands below work in both Bash and PowerShell when run from the repository root; paths use relative examples.

```bash
python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --model convnextv2_tiny --epochs 200 --batch-size 32 --update-freq 4 --output ./outputs/train
```

Fine-tune from **compatible** ImageNet pretrained ConvNeXt V2 weights:

```bash
python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --model convnextv2_tiny --pretrained ./weights/convnextv2_tiny_1k_224_ema.pt --epochs 200 --batch-size 32 --update-freq 4 --output ./outputs/train
```

The `--pretrained` argument is optional; without it, training starts without external pretrained weights. Do not confuse pretraining weights with an already trained Copper checkpoint. Verify compatibility and license of third-party weights before using them.

Resume an interrupted Copper training run:

```bash
python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --model convnextv2_tiny --resume ./outputs/train/checkpoint-199.pth --output ./outputs/train
```

The resume filename is an **example only**: inspect the actual files in your output directory. `--resume` should point to a trusted, compatible training checkpoint.

Key options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | `convnextv2_tiny` | Registered model factory |
| `--epochs` | `200` | Training epochs |
| `--batch-size` | `32` | Per-step/per-GPU microbatch |
| `--update-freq` | `4` | Gradient accumulation factor |
| `--blr` | `3.2e-3` | Base LR, scaled by effective batch by the core trainer |
| `--workers` | `0` | DataLoader processes; zero is a portable Windows-safe baseline |
| `--device` | `auto` | CUDA when available, otherwise CPU |
| `--seed` | `0` | Random seed |
| `--output` | required | Checkpoints and metrics |

`batch-size × update-freq` affects effective batch size and learning rate. For small GPUs, reduce `--batch-size` and (if you want the same effective batch) increase `--update-freq`. CPU training is permitted by the entry point but may be very slow; GPU training is recommended.

The original trainer handles best-checkpoint selection, logs, metrics and saving. Do not assume every epoch's checkpoint is retained. The output directory is under `outputs/`, which Git ignores.

## 5. Evaluation / testing

Point to a trained Copper checkpoint (not the ImageNet pretrained file). `--split test` reads `<data>/test`:

```bash
python scripts/copper.py eval --data ./data/copper --split test --checkpoint ./outputs/train/checkpoint-best.pth --labels ./configs/class_labels_csd.txt --output ./outputs/test
```

Validation instead:

```bash
python scripts/copper.py eval --data ./data/copper --split val --checkpoint ./outputs/train/checkpoint-best.pth --labels ./configs/class_labels_csd.txt --output ./outputs/val
```

The wrapper reads the model name, image size and class count from checkpoint metadata, rejects incompatible input shape/class count, and reuses the original `main_finetune.py --eval true` evaluator. Expected artifacts include `eval-class-acc.txt`, confusion matrices, `eval_predictions.csv`, `eval_misclassified.csv` and preprocessing exclusion reports.

**Label identity matters:** a checkpoint's class count alone is insufficient to prove the label names and order match. Always supply the label file used for that model's training. Only load checkpoints from trusted sources because PyTorch checkpoint deserialization may execute arbitrary code.

## 6. Single-image inference and batch inference

### One image (direct entry point)

```bash
python infer_topk.py --data ./data/example.png --checkpoint ./outputs/train/checkpoint-best.pth --class-labels ./configs/class_labels_csd.txt --k 3 --conf 0.7 --output ./outputs/single --device cpu --amp false --num-workers 0
```

### Entire directory (recommended)

```bash
python scripts/copper.py infer --data ./data/images --checkpoint ./outputs/train/checkpoint-best.pth --labels ./configs/class_labels_csd.txt --k 3 --conf 0.7 --batch-size 32 --output ./outputs/inference
```

The directory can contain unlabeled images or class-named subdirectories. The unified command creates `manifest.json` (file list, sizes, SHA-256 and eligible 200×200 count) **before inference**.

Top-k behavior: every image has a Top-1 prediction; only when its **unrounded** Top-1 confidence is strictly below `--conf` are `k` candidate classes emitted. A confidence exactly equal to threshold does not trigger Top-k. `k` must be between 1 and the number of classes.

Inference outputs:

| File | Purpose |
| --- | --- |
| `manifest.json` | Snapshot used for future data integrity checks |
| `all_predictions.jsonl` | All valid images, Top-1 predictions, optional Top-k and relative paths |
| `low_confidence_topk.jsonl` | Low-confidence only |
| `low_confidence_topk.csv` | Human-readable low-confidence candidates |
| `excluded_non_200x200.csv` | Images excluded by size |
| `summary.json` | Counts, confidence statistics, timing, model/weight/label hashes |

To run on CPU, specify `--device cpu`. For direct `infer_topk.py`, also pass `--amp false`. Windows users should start with `--workers 0`, especially with large image collections.

## 7. Organize an inferred dataset

After the folder inference above, run:

```bash
python scripts/copper.py organize --data ./data/images --inference ./outputs/inference --labels ./configs/class_labels_csd.txt --output ./outputs/organized
```

Equivalent direct command:

```bash
python scripts/organize-inference-results.py --data ./data/images --predictions ./outputs/inference/all_predictions.jsonl --summary ./outputs/inference/summary.json --manifest ./outputs/inference/manifest.json --class-labels ./configs/class_labels_csd.txt --output ./outputs/organized
```

Generated organization layout:

```text
outputs/organized/
  by_top1/
    class_a/...
    class_b/...
  top1_predictions.csv
  low_confidence_top3.json   # name varies with k
  organization_summary.json
```

**Safety guarantees:** original inputs are copied, never moved; each input file is checked against its SHA-256 manifest; prediction coverage must be complete and unique; invalid/mismatched class data and confidence rules are rejected; output collision causes a refusal to overwrite. The organizer supports class counts and `k/conf` values recorded by the inference run, rather than hardcoding the old seven-class Top-3 configuration.

For a **different inference machine**, transfer the entire inference output directory (including `manifest.json`) and ensure the target machine has a byte-identical copy of the original images. Inference results include portable relative paths and support reading Windows or Linux absolute paths; the organizing machine's local data root can differ from the machine where predictions were produced. If any files have been added, removed or modified, the organizer will refuse to proceed rather than silently produce a misleading dataset.

Do not put generated outputs under your inference input directory, as this would invalidate the manifest.

## 8. Advanced entry points

The original `main_finetune.py` remains available for uncommon training settings (e.g. weighted class loss, hard-example resampling, multi-GPU/distributed training), and `scripts/build-hard-example-manifest.py` remains available as a Python utility. These paths are intentionally **optional**; they do not require historical `experiments/` output, repository-clean checks, or an SSH server.

The optional `MinkowskiEngine` submodule and `main_pretrain.py` are for FCMAE self-supervised pretraining only. They are **not needed** for ordinary supervised classification or inference, and their special compiled dependencies are not installed by the normal requirements.

The seven-class `deployment/minimal/` packaging files are tied to a particular historical checkpoint and have their own documentation and integrity checks. They are not a substitute for a general trained checkpoint, and no pretrained/trained model weights are committed to this branch.

## 9. Maintenance and troubleshooting

- **Git permissions:** repository is private; configure GitHub SSH/HTTPS authentication before cloning.
- **GPU unavailable:** inspect `nvidia-smi`, the NVIDIA driver, PyTorch wheel and `torch.cuda.is_available()`; specify `--device cpu` for supported CPU tasks.
- **Out of memory:** decrease batch size and worker count; keep effective batch/lr scaling in mind.
- **Dataset mismatch:** ensure every split contains the same class directories with identical labels/order, and 200×200 images.
- **Checkpoint mismatch:** use a checkpoint from the same architecture and label vocabulary; the wrapper refuses incompatible metadata.
- **Unexpected excluded images:** see `excluded_non_200x200.csv`. No automatic resizing occurs.
- **Organize errors:** compare the original dataset with the saved manifest, the inference `summary.json` and prediction count.
- **Regression tests:** `python -m unittest discover -s tests -v`. They should be executed under the installed environment; passing tests is not a substitute for testing on each target GPU/OS.
- **Upgrading source:** this branch does not automatically merge changes from `main`; selectively port and test desired changes.

## 10. Attribution and license

Based on [Meta's ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) and the existing Copper defect-classification additions. Keep all upstream copyright and license notices. See [LICENSE](LICENSE). Model weights and datasets may have their own license restrictions; check them before redistribution.
