#!/usr/bin/env bash
set -euo pipefail
cd /home/cxz/projects/copper-foil-defect-classification
source scripts/server-env.sh
export DATA_PATH=/share/data/cxz/copper-foil-defect-classification/1017-formal-split-9classes
export CLASS_LABELS="$DATA_PATH/class_labels.txt"
export NB_CLASSES=9 MODEL=convnextv2_tiny EPOCHS=200 BATCH_SIZE=32 UPDATE_FREQ=4
[[ -z "$(git status --porcelain)" ]]
"$HOME/.local/bin/micromamba" run -n "$COPPER_ENV_NAME" python -c 'import hashlib,json,pathlib,os; p=pathlib.Path(os.environ["DATA_PATH"]); m=json.loads((p/"sha256_manifest.json").read_text()); assert len(m)==12407; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in m.items()); print("Verified 12407 image SHA256 hashes")'
x=$(./scripts/start-experiment.sh "$(TZ=Asia/Shanghai date +%Y%m%d-%H%M)-1017-9classes-train" -- "$HOME/.local/bin/micromamba" run -n "$COPPER_ENV_NAME" bash train_csd.sh)
echo "$x"
r=$(printf '%s\n' "$x" | sed -n 's/^RUN_DIR=//p')
cp "$DATA_PATH/README.md" "$r/dataset_README.md"
cp "$CLASS_LABELS" "$r/class_labels.txt"
cp "$DATA_PATH/split_manifest.csv" "$r/split_manifest.csv"
cp "$DATA_PATH/sha256_manifest.json" "$r/dataset_sha256.json"
sha256sum "$PRETRAINED_WEIGHT" > "$r/pretrained_sha256.txt"
printf 'dataset=%s\nsplit_seed=42\ntraining_seed=0\nepochs=200\nmodel=%s\n' "$DATA_PATH" "$MODEL" > "$r/dataset_config.txt"
while [[ ! -f "$r/exit_code.txt" ]]; do sleep 30; done
[[ "$(cat "$r/exit_code.txt")" == 0 ]]
RUN_DIR="$r" "$HOME/.local/bin/micromamba" run -n "$COPPER_ENV_NAME" python -c 'import os,subprocess; allowed=os.path.relpath(os.environ["RUN_DIR"])+"/"; lines=subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],text=True).splitlines(); assert all(x.startswith("?? "+allowed) for x in lines),lines'
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=status.showUntrackedFiles GIT_CONFIG_VALUE_0=no
y=$(bash scripts/eval.sh "$(TZ=Asia/Shanghai date +%Y%m%d-%H%M)-1017-9classes-test" "$(basename "$r")")
echo "$y"
q=$(printf '%s\n' "$y" | sed -n 's/^RUN_DIR=//p')
cp "$r/dataset_config.txt" "$q/dataset_config.txt"
cp "$CLASS_LABELS" "$q/class_labels.txt"
printf '%s\n' "$r/checkpoint-best.pth" > "$q/training_checkpoint.txt"
sha256sum "$r/checkpoint-best.pth" > "$q/checkpoint_sha256.txt"
while [[ ! -f "$q/exit_code.txt" ]]; do sleep 30; done
[[ "$(cat "$q/exit_code.txt")" == 0 ]]
echo SERIES_COMPLETED
