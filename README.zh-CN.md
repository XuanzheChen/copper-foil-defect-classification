# 铜箔表面缺陷分类（独立部署版）

[**简体中文**](README.zh-CN.md) | [**English**](README.md)

本项目基于 ConvNeXt V2，实现 **200×200 铜箔表面灰度图像分类**。`portable/standalone` 分支面向 Windows/Linux 独立使用，不要求原科研仓库中的历史 `experiments/`、三端协作、SSH 服务器同步、固定服务器目录或强制实验封装器。

> **分支定位：**这是从 `main` 派生并单独维护的独立部署分支。核心模型、训练检查点格式和正式图像预处理逻辑保持不变。已经提供自动安装器；本 README 同时保留手动安装作为备选方案。

## 0. 快速开始：自动安装（推荐）

### 前置条件

- 已安装 **Git**，并且当前账号可以访问私有仓库 `XuanzheChen/copper-foil-defect-classification`。使用 GitHub SSH Key，或已配置 Git Credential Manager 的 HTTPS 认证；**不要将 Token 写入命令、URL 或提交到仓库**。
- 安装器支持 **Windows 10/11 x64** 或 **Linux x86-64**（Linux 需要 glibc ≥ 2.28）。macOS、ARM64、ROCm 和 WSL 不属于当前验收范围。
- 第一次安装需要网络连接 GitHub、Astral uv 和 PyTorch 软件包源；CUDA 版依赖体积可能较大。
- 安装器使用用户级 `uv` 管理 **Python 3.10**，在项目目录创建 `.venv/`；不要求管理员权限、不改系统 Python/Conda、不安装系统级 CUDA Toolkit。
- 安装器根据 NVIDIA GPU 计算能力及驱动版本，从 PyTorch `cu118`、`cu126`、`cu128` 或 CPU 配置中选型；验证真实张量计算和 200×200 模型前向。**若已检测到 NVIDIA GPU 却不兼容，不会静默回退 CPU。**

首次运行时，如果未安装 `uv`，启动脚本可能从 `astral.sh` 下载并执行官方安装脚本。如需事先自行审核，可先安装 uv，或通过 `COPPER_NO_UV_BOOTSTRAP=1`（Windows 也可用 `-NoUvBootstrap`）禁止自动引导。

### Linux：首次克隆并安装

在 Bash 终端执行：

```bash
git clone --depth 1 --single-branch --branch portable/standalone \
  git@github.com:XuanzheChen/copper-foil-defect-classification.git
bash copper-foil-defect-classification/install.sh
```

如果已经位于仓库根目录：

```bash
bash install.sh
```

### Windows：首次克隆并安装

在 PowerShell 终端执行：

```powershell
git clone --depth 1 --single-branch --branch portable/standalone git@github.com:XuanzheChen/copper-foil-defect-classification.git
if ($LASTEXITCODE -eq 0) {
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\copper-foil-defect-classification\install.ps1
}
```

如果已经位于仓库根目录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

`-ExecutionPolicy Bypass` 仅用于当前启动的 PowerShell 子进程，不会永久修改系统执行策略。如果使用 HTTPS 凭据管理器而非 SSH，请把克隆地址改为：

```text
https://github.com/XuanzheChen/copper-foil-defect-classification.git
```

### 自动安装器命令速查

下表命令均在已经克隆好的 `portable/standalone` 仓库根目录中执行。

| 操作 | Linux Bash | Windows PowerShell |
| --- | --- | --- |
| 自动选择安装配置 | `bash install.sh` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1` |
| 明确安装 CPU 版 | `bash install.sh --backend cpu` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Backend cpu` |
| 明确选择 CUDA 12.6 | `bash install.sh --backend cu126` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Backend cu126` |
| 选择第 2 张 GPU | `bash install.sh --gpu-index 1` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -GpuIndex 1` |
| 克隆并安装至空目录 | `bash install.sh --target ~/copper-copy` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Target D:\copper-copy` |
| 安全拉取远端新提交 | `bash install.sh --update` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Update` |
| 重建 Copper 管理的虚拟环境 | `bash install.sh --recreate-env` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -RecreateEnv` |
| 预览安装计划 | `bash install.sh --dry-run` | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -DryRun` |

可选配置为 `auto`（默认）、`cpu`、`cu118`、`cu126`、`cu128`。**不能只凭 `nvidia-smi` 的“CUDA Version”数字选择 PyTorch Wheel**；安装器还会检查驱动和 GPU 计算能力。CUDA 档位或显卡不兼容时，应先诊断原因，或显式使用 CPU 选项。

其他注意事项：

- `--update` / `-Update` 仅在工作区无未提交/未跟踪文件时执行 `git fetch` 与 `git merge --ff-only`，不会强制覆盖修改。
- `--recreate-env` / `-RecreateEnv` 只会重建由 Copper 安装器标记管理的 `.venv`，不会删除数据集、权重或其他虚拟环境。
- `--dry-run` / `-DryRun` 不克隆项目、不安装项目依赖，但启动脚本在缺少 uv/Python 时**仍可能先下载 uv 和 Python 3.10**。要完全避免下载，请自行准备好相应环境，再直接运行 `python scripts/install/bootstrap.py --dry-run`。
- 如需采用 GitHub HTTPS 认证安装到另一空目录，可加 `--repo-url https://github.com/XuanzheChen/copper-foil-defect-classification.git`（Windows 对应 `-RepoUrl`）。
- 安装器不会下载数据集、ImageNet 预训练权重或已训练的铜箔分类模型，也不会自动开始训练。
- 安装失败后可以查看错误并在相同配置下重试；只有真实安装及验证通过才会报告成功。

更多驱动兼容性、认证、安全约束及排错细节参阅 **[INSTALLER.md](INSTALLER.md)**。

### 安装后：无需激活环境即可验证

进入项目根目录，然后执行对应系统的命令。

**Linux Bash：**

```bash
cd copper-foil-defect-classification
.venv/bin/python scripts/copper.py --help
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cat .venv/copper-install-report.json
```

**Windows PowerShell：**

```powershell
cd copper-foil-defect-classification
.\.venv\Scripts\python.exe scripts\copper.py --help
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
Get-Content .\.venv\copper-install-report.json
```

安装完成后会生成 `.venv/copper-install-state.json` 和 `.venv/copper-install-report.json`。后续本 README 给出的 `python ...` 训练/测试/推理命令，若未激活虚拟环境，Linux 将 `python` 替换为 `.venv/bin/python`，Windows 替换为 `.\.venv\Scripts\python.exe`。

## 1. 功能及统一运行入口

| 任务 | Windows/Linux 统一 Python 命令 | 说明 |
| --- | --- | --- |
| 监督训练 | `python scripts/copper.py train` | 保留原模型、200×200 正式预处理和检查点选择 |
| 模型测试/验证 | `python scripts/copper.py eval` | 从检查点读取模型架构、输入尺寸和类别数 |
| 文件夹批量推理 | `python scripts/copper.py infer` | 自动生成输入文件 SHA-256 清单并输出 Top-k |
| 推理结果整理 | `python scripts/copper.py organize` | 低置信度按 Top-1 归入 `类别(other)/`，校验清单且不移动原图 |
| 单张图片推理 | `python infer_topk.py` | 直接调用底层推理入口 |
| 七类最小部署包 | `deployment/minimal/` | 特定历史模型的可选打包工具 |
| 高级训练 | `main_finetune.py` | 保留类别损失权重、难例重采样等高级参数 |

`scripts/copper.py` 是跨平台命令包装器，不会重新实现或分叉模型训练算法。模型实现位于 `models/self/convnextv2_modified.py`。

## 2. 手动安装（自动安装器不可用时）

建议优先使用第 0 节的自动安装。以下手动命令适用于 **Python 3.10**，需自己确认 PyTorch Wheel 与 GPU/驱动兼容，不能将 CUDA 12.8 安装例子视为所有 NVIDIA GPU 通用的配置。

### Linux Bash

```bash
git clone --depth 1 --single-branch --branch portable/standalone \
  git@github.com:XuanzheChen/copper-foil-defect-classification.git
cd copper-foil-defect-classification
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如设备已确认兼容 CUDA 12.8 对应的 PyTorch 2.7.1 Wheel（例如原 RTX 5080 参考环境）：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

只需要 CPU 时，改用：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone --depth 1 --single-branch --branch portable/standalone git@github.com:XuanzheChen/copper-foil-defect-classification.git
cd copper-foil-defect-classification
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

已确认支持 CUDA 12.8 的 GPU：

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CPU 环境：

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

如果 PowerShell 拒绝运行 `Activate.ps1`，可以**不激活**，直接使用 `.\.venv\Scripts\python.exe` 作为后续命令的 Python 路径。

### 手动安装验证

在已经激活的环境中执行：

```bash
python -c "import torch,torchvision,timm; print('torch',torch.__version__,'vision',torchvision.__version__,'timm',timm.__version__,'cuda',torch.cuda.is_available())"
python -m unittest discover -s tests -v
python scripts/copper.py --help
```

NVIDIA GPU 可以额外检查：

```bash
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(4,device='cuda').sum().item())"
```

以上安装模板不是已在所有 Windows/Linux/GPU 组合验证通过的保证。其它 PyTorch 兼容版本请参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

## 3. 数据集和标签文件

采用 ImageFolder 格式；训练与测试数据集的类别文件夹应保持一致：

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

标签文件必须从 0 开始、连续编号，每行采用 `index:class_name`，例如：

```text
0:class_a
1:class_b
```

**非常重要：**当前底层 ImageFolder 按文件夹名的字典序（lexicographic order）生成类别索引。标签文件中的类别顺序必须与其一致，且训练/测试使用与训练检查点相同的标签语义；仅类别数相同并不能保证标签一致。封装器会检查相应数据集文件夹与标签文件的名称、数量及顺序。

仓库中的 `configs/class_labels_csd.txt` 为九类示例，`configs/class_labels_csd_10.txt` 为十类示例；`deployment/minimal/` 则是特定历史七类模型的部署方案。**这些模型标签不可混用。**

正式样本仅接受 **200×200 像素**：非 200×200 图像会被跳过并生成排除报告，**不会自动调整尺寸**。图像先按灰度读取、扩成三通道，再按 ImageNet 统计量归一化；训练时还会采用现有的平移及亮度/对比度增强。直接改写预处理可能使旧权重的行为发生变化。

目前底层 `main_finetune.py` 即使处于 `--eval` 模式，也会先初始化训练集，因此测试时仍需要数据目录下真实存在的 `train/`；不要虚构占位训练数据。

## 4. 模型训练

以下所有示例在项目根目录执行，数据路径和标签应替换成真实目录。示例命令在 Bash/PowerShell 下都可使用；如果没有激活虚拟环境，使用第 0 节说明的 Python 完整路径。

### 从头训练

```bash
python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --model convnextv2_tiny --epochs 200 --batch-size 32 --update-freq 4 --output ./outputs/train
```

### 使用兼容的 ImageNet 预训练权重微调

```bash
python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --model convnextv2_tiny --pretrained ./weights/convnextv2_tiny_1k_224_ema.pt --epochs 200 --batch-size 32 --update-freq 4 --output ./outputs/train
```

`--pretrained` 可省略；省略时不会自动下载预训练权重。ImageNet 预训练参数不等于训练完成的 Copper 缺陷分类检查点，使用时需要核对模型架构与许可条件。

### 恢复中断训练

```bash
python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --model convnextv2_tiny --resume ./outputs/train/checkpoint-199.pth --output ./outputs/train
```

其中 `checkpoint-199.pth` **只是文件名模板**；应查看输出目录中实际保存的检查点并使用可信、兼容的文件。

### 常用训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | `convnextv2_tiny` | 模型结构 |
| `--epochs` | `200` | 总训练轮数 |
| `--batch-size` | `32` | 每步/每卡批大小 |
| `--update-freq` | `4` | 梯度累积倍数 |
| `--blr` | `3.2e-3` | 基础学习率，原训练器会结合有效 batch 进行缩放 |
| `--workers` | `0` | DataLoader 工作进程数，0 是兼容 Windows 的稳妥起点 |
| `--device` | `auto` | 有 CUDA 则优先 CUDA，否则 CPU |
| `--seed` | `0` | 随机种子 |
| `--output` | 必填 | 检查点和运行指标保存位置 |

有效 batch 受 `batch-size × update-freq` 影响，也会影响学习率缩放。显存不足时可减小 batch，并按需求调整梯度累积倍数。CPU 训练可使用但往往较慢；推荐 GPU 训练。

现有训练器负责训练日志、模型检查点及最佳权重选择；**并非每一轮检查点都会长期保留**。建议将运行结果保存在 Git 已忽略的 `outputs/` 目录下。

## 5. 模型验证与测试

使用训练好的 Copper 检查点进行测试，不要误把 ImageNet 预训练权重当作完整分类模型。

### 测试集评估

```bash
python scripts/copper.py eval --data ./data/copper --split test --checkpoint ./outputs/train/checkpoint-best.pth --labels ./configs/class_labels_csd.txt --output ./outputs/test
```

### 验证集评估

```bash
python scripts/copper.py eval --data ./data/copper --split val --checkpoint ./outputs/train/checkpoint-best.pth --labels ./configs/class_labels_csd.txt --output ./outputs/val
```

评估器从检查点中读取模型名称、输入尺寸及类别数，对不兼容配置拒绝执行，调用原 `main_finetune.py --eval true`。预期输出包括：

- `eval-class-acc.txt`：总体及分类指标；
- `confusion_matrix.csv`、`confusion_matrix_normalized.csv`：混淆矩阵；
- `eval_predictions.csv`、`eval_misclassified.csv`：逐样本预测及错误样本；
- `preprocessing_reports/`：预处理排除报告。

**注意权重安全：**PyTorch 检查点反序列化可能执行非可信代码，只加载自己训练或来源可信的检查点。验证类别数不代表验证标签语义，因此仍须核实标签文件与训练时一致。

## 6. 单张图像与批量推理

### 单张图片推理（直接入口）

```bash
python infer_topk.py --data ./data/example.png --checkpoint ./outputs/train/checkpoint-best.pth --class-labels ./configs/class_labels_csd.txt --k 3 --conf 0.7 --output ./outputs/single --device cpu --amp false --num-workers 0
```

### 文件夹批量推理（推荐入口）

```bash
python scripts/copper.py infer --data ./data/images --checkpoint ./outputs/train/checkpoint-best.pth --labels ./configs/class_labels_csd.txt --k 3 --conf 0.7 --batch-size 32 --output ./outputs/inference
```

输入可以是无标签图片目录，也可以是以真实类别命名的子文件夹。统一入口会在推理之前生成 `manifest.json`，其中保存相对文件路径、文件大小、SHA-256 和符合 200×200 尺寸的样本数量。

**Top-k 规则：**每张有效图都有 Top-1 预测；只有未四舍五入的 Top-1 softmax 置信度严格 **小于 `--conf`** 时，才附加 `k` 个候选类别；等于阈值不触发。`k` 必须在 1 到类别总数之间。

### 推理输出文件

| 文件 | 作用 |
| --- | --- |
| `manifest.json` | 输入数据集快照，供后续完整性校验 |
| `all_predictions.jsonl` | 全部有效图片的 Top-1 与可选 Top-k 信息 |
| `low_confidence_topk.jsonl` | 仅低置信度图片 |
| `low_confidence_topk.csv` | 低置信度候选类别和概率 CSV |
| `excluded_non_200x200.csv` | 尺寸不符合要求的图像 |
| `summary.json` | 数量、阈值、推理耗时、模型/权重/标签哈希等 |

CPU 模式使用 `--device cpu`；直接调用 `infer_topk.py` 时还需 `--amp false`。Windows 上建议先使用 `--workers 0`，确认稳定后再调整。

## 7. 批量推理后的数据集整理

完成第 6 节的文件夹推理后，可按 **Top-1 预测类别 + 置信度阈值** 将图片复制到不同目录，原始数据保持不变。

### 统一命令

```bash
python scripts/copper.py organize --data ./data/images --inference ./outputs/inference --labels ./configs/class_labels_csd.txt --output ./outputs/organized
```

### 等价的底层命令

```bash
python scripts/organize-inference-results.py --data ./data/images --predictions ./outputs/inference/all_predictions.jsonl --summary ./outputs/inference/summary.json --manifest ./outputs/inference/manifest.json --class-labels ./configs/class_labels_csd.txt --output ./outputs/organized
```

### 按 conf 分流到普通类别与 `(other)` 类别

整理工具直接读取本次推理 `summary.json` 中的 `conf`（如 `0.7`），不要求在 `organize` 命令中再次传入阈值：

- **Top-1 置信度 ≥ conf**：复制到 `by_top1/类别名/`。
- **Top-1 置信度 < conf**：复制到 `by_top1/类别名(other)/`。
- **恰好等于 conf**：属于普通类别，而不是 `(other)`。
- 例如，预测为 `class_a`、置信度 `0.69` 且 `conf=0.7` 的样本进入 `class_a(other)`；同类置信度 `0.70` 的样本进入 `class_a`。

整理后的目录示例：

```text
outputs/organized/
  by_top1/
    class_a/                 # Top-1 是 class_a，置信度 >= conf
    class_a(other)/          # Top-1 是 class_a，置信度 < conf
    class_b/                 # Top-1 是 class_b 的高置信度样本
    class_b(other)/          # class_b 存在低置信度样本时才创建
  top1_predictions.csv       # 保留预测类别，并增加 organized_class 实际归档目录
  low_confidence_top3.json   # 低置信度 Top-k 候选，文件名随 k 变化
  organization_summary.json  # 原预测数、低置信度数、最终各目录数量
```

**注意：**`(other)` 只是样本整理目录，不是新的模型类别，不改变 Top-1 预测结果。汇总中 `per_class` 仍代表所有 Top-1 类别数量，`low_confidence_per_class` 是其中低置信度数量，新增 `organized_per_class` 对应实际归档目录。若真实标签名与自动生成的 `类别(other)` 目录冲突（含 Windows 大小写不敏感的情况），整理器会拒绝执行，避免误混样本。**如需修改 conf，应重新用目标 `--conf` 批量推理，再整理新结果**，而不是直接修改 `summary.json`。


**完整性保护：**

- 只复制，不移动或修改原始图片；
- 使用 `manifest.json` 检查输入文件是否缺失、增加或发生改动（SHA-256）；
- 校验预测结果覆盖全部有效图片，且每张图片对应唯一一条记录；
- 校验类别、置信度和 Top-k 规则一致；
- 目标输出文件冲突时拒绝覆盖；
- 不把新生成的整理目录放在需要被校验的原始输入目录里。

整理器不再强制固定“七类、Top-3、0.7 阈值”，而是读取推理摘要中的 `k/conf` 和实际标签文件。

**跨机器整理：**如在 Linux 服务器推理、回 Windows 本地整理，需要将整个推理输出目录（含 `manifest.json`）以及字节一致的原图数据集带回。原始数据的本地存储位置可以改变；推理信息保留相对路径和 Windows/Linux 路径兼容处理。但只要数据有变化，校验就会拒绝继续，以避免错误整理。

## 8. 高级功能

现有 `main_finetune.py` 保留加权分类损失、难例采样、分布式/多卡训练等可选参数；`scripts/build-hard-example-manifest.py` 可以用于生成难例清单。它们不是一般训练/测试/推理的必需步骤，也不依赖原来的 `experiments/` 归档和服务器 SSH 同步。

`main_pretrain.py` 与可选子模块 `MinkowskiEngine` 仅与 FCMAE 自监督预训练有关；常规监督训练与分类推理**不需要**安装其定制编译依赖。

`deployment/minimal/` 属于特定历史七类分类模型的最小部署包结构，带独立 README 和完整性检查，不应把其中的标签或元数据直接当作其他模型的配置。仓库没有把训练数据或模型权重一并提交。

## 9. 常见问题与维护

- **GitHub clone 提示权限不足：**确认已授权访问私有仓库，并配置好 SSH Key 或 HTTPS 凭据；不要在脚本或 URL 中明文保存 Token。
- **安装器识别不到 GPU：**检查 `nvidia-smi`、驱动版本与 GPU 计算能力；确认使用了支持该硬件的 PyTorch Wheel。需要 CPU 时显式使用 `--backend cpu` / `-Backend cpu`。
- **显存不足：**优先减少 `--batch-size`、`--workers`，需要维持有效 batch 时再调 `--update-freq`。
- **已有虚拟环境冲突：**如果 `.venv` 不是 Copper 安装器管理的环境，安装器会拒绝接管，需自行备份并移走。切换 CUDA 档位时使用 `--recreate-env`。
- **数据集/标签不匹配：**检查各 split 的类别文件夹及其字典序、类别数、标签文件和检查点是否一致。
- **大量图片被排除：**检查 `excluded_non_200x200.csv`，正式流程不会自动 resize。
- **整理失败：**检查输入数据与 `manifest.json` 是否一致、`summary.json` 和预测记录数量是否对应。
- **更新代码：**安装器 `--update` / `-Update` 只对干净的 `portable/standalone` 工作区做快进更新；此分支不会自动合并 `main` 的所有科研改动。
- **自动化测试：**`python -m unittest discover -s tests -v`。GitHub Actions 在 Windows/Linux 执行语法与离线策略测试；**CI 通过不代表所有真实 GPU 安装验证已经通过**。GPU 与模型前向仍应在目标机器实际测试。

## 10. 代码来源与许可

本项目基于 [Meta ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) 和现有 Copper 铜箔缺陷分类扩展代码。保留原项目的版权及许可证声明，具体参见 [LICENSE](LICENSE)。训练数据、第三方预训练权重及其它模型权重可能存在独立授权条款，分发前请另行确认。
