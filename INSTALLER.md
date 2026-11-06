# Copper Standalone 自动安装器（Windows / Linux）

此安装器服务于 `portable/standalone` 分支，不会修改 `main`，也不需要旧三端协作配置、实验记录封装、CUDA Toolkit 或 Conda。

## 0. 前置条件与认证

- **必须有 Git**（Windows 可安装 Git for Windows；Linux 使用发行版包管理器）。
- 仓库现已设为 **Public**：默认使用 **HTTPS Git clone**，无需 SSH Key 或 GitHub 登录。仅自愿使用 SSH 地址时才需要配置 SSH Key。不接受带 token/password 的 URL，不在文件中保存凭据。
- 支持 **Windows 10/11 x64、Linux x86-64**（Linux 使用 glibc 2.28 或更新版本）；macOS/WSL/ARM64/ROCm 不在本安装器的验收范围内。
- 需要能够访问 GitHub、Astral uv 和 PyTorch 包索引。第一次安装将下载 Python、PyTorch 与依赖，可能占用较多网络流量和磁盘空间。
- 默认选用用户级 `uv` 管理的 **Python 3.10** 并在项目目录生成 `.venv`，**不使用管理员权限、不更改系统 Python/Conda**。

> 安全提示：首次启动时如果系统找不到 `uv`，入口脚本会从 `https://astral.sh/uv/install.sh` 或 `install.ps1` 下载并执行 Astral 官方安装脚本。希望自行审核/安装 `uv` 的用户可先安装，或者设置 `COPPER_NO_UV_BOOTSTRAP=1` 禁止自动引导。下载外部脚本本身应受你的网络安全策略约束。

## 1. 第一次使用

**Linux Bash**（已拥有仓库访问权限）：

```bash
git clone --depth 1 --single-branch --branch portable/standalone \
  https://github.com/XuanzheChen/copper-foil-defect-classification.git
bash copper-foil-defect-classification/install.sh
```

**Windows PowerShell**：

```powershell
git clone --depth 1 --single-branch --branch portable/standalone https://github.com/XuanzheChen/copper-foil-defect-classification.git
if ($LASTEXITCODE -eq 0) {
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\copper-foil-defect-classification\install.ps1
}
```

HTTPS 地址是本仓库及安装器的**默认方式**。PowerShell 的 `-ExecutionPolicy Bypass` 仅作用于上述单次子进程；不要为此修改系统级执行策略。**注意：PowerShell 的 `$LASTEXITCODE` 和花括号语法不能直接粘贴到 CMD 中。**

**Windows CMD：**请先执行克隆，确认成功后再运行安装器（CMD 不支持 PowerShell 的 `if ($LASTEXITCODE -eq 0) { ... }` 语法）：

```cmd
git clone --depth 1 --single-branch --branch portable/standalone https://github.com/XuanzheChen/copper-foil-defect-classification.git
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\copper-foil-defect-classification\install.ps1"
```

在现有 `portable/standalone` 仓库根目录内也可直接运行 `bash install.sh`（Linux）或 `powershell.exe -File .\install.ps1`（Windows，须符合本机执行策略）。

**只持有安装脚本，但不想预先 clone 的情况：**先将仓库中的 `install.sh` 或 `install.ps1` 及其 `scripts/install/` 同目录结构复制到临时目录，然后使用 `--target` / `-Target` 指定空目录；安装器将通过 HTTPS 克隆指定 GitHub 分支后安装。项目公开后可从 GitHub 获取安装脚本，但执行前请自行审阅。

## 2. 自动选择 CUDA 配置

安装器使用 `nvidia-smi` 查询 **GPU 计算能力（compute capability）和驱动版本**，并依据 `scripts/install/environment_profiles.json` 中经过整理的 PyTorch 2.7.1/TorchVision 0.22.1 兼容矩阵自动选型。它不会仅凭 `nvidia-smi` 的 “CUDA Version” 字样选包。

| CUDA 档位 | 用途（示例） | Linux 最低驱动（保守） | Windows 最低驱动（保守） |
| --- | --- | --- | --- |
| `cu128` | 新 NVIDIA / Blackwell（计算能力 12.0）及其他适配架构 | 570.26 | 570.65 |
| `cu126` | 较老 CUDA GPU 或驱动不满足 cu128 | 560.28.03 | 560.76 |
| `cu118` | 驱动较旧但符合 PyTorch 2.7.1 wheel 要求 | 520.61.05 | 520.06 |
| `cpu` | 无 NVIDIA GPU，或用户显式选择 | 无要求 | 无要求 |

上述阈值采用 NVIDIA CUDA Toolkit 发布配套驱动的**保守要求**，不代表 CUDA minor-version compatibility 的理论最低驱动。特定 GPU 是否受某 wheel 支持取决于架构；安装后必须通过真实 GPU 运算和 ConvNeXt `200×200` 前向测试。对当前兼容表没有收录的 GPU/驱动组合，工具会解释问题，不会猜测安装。

CPU 自动模式仅在 **没有可用 `nvidia-smi`** 时使用，并明确打印所选 CPU 模式。**检测到 GPU 但驱动或架构不匹配时，不会自动退回 CPU。** 如果确实希望安装 CPU 环境，使用 `--backend cpu` / `-Backend cpu`。

## 3. 常用选项

| 用途 | Linux | Windows PowerShell |
| --- | --- | --- |
| 自动安装 | `bash install.sh` | `powershell.exe -File .\install.ps1` |
| 选择 CPU | `bash install.sh --backend cpu` | `.\install.ps1 -Backend cpu` |
| 强制指定 CUDA 档位 | `bash install.sh --backend cu126` | `.\install.ps1 -Backend cu126` |
| 指定 GPU 索引 | `bash install.sh --gpu-index 1` | `.\install.ps1 -GpuIndex 1` |
| 克隆到新目录 | `bash install.sh --target ~/copper-standalone` | `.\install.ps1 -Target D:\copper-standalone` |
| 安全更新代码 | `bash install.sh --update` | `.\install.ps1 -Update` |
| 重新创建 Copper 虚拟环境 | `bash install.sh --recreate-env` | `.\install.ps1 -RecreateEnv` |
| 查看选型（不克隆或安装项目） | `bash install.sh --dry-run` | `.\install.ps1 -DryRun` |

`--dry-run` 保证不克隆项目、不建立 `.venv`、不安装 PyTorch，但启动脚本仍会在需要时安装 **uv 本身和 Python 3.10**；如果要求绝对零下载，请先自行准备 uv/Python，再直接运行 `python scripts/install/bootstrap.py --dry-run`。

默认克隆地址为无需认证的 HTTPS。若已经配置 GitHub SSH Key，也可以用 `--repo-url` / `-RepoUrl` 改用 SSH；指定 HTTPS 地址同样受支持：

```bash
bash install.sh --target ~/copper-standalone --repo-url https://github.com/XuanzheChen/copper-foil-defect-classification.git
```

```powershell
.\install.ps1 -Target D:\copper-standalone -RepoUrl https://github.com/XuanzheChen/copper-foil-defect-classification.git
```

**非破坏性保护**：如果目标目录非空且不是预期仓库、Git 分支不是 `portable/standalone`、`origin` 不是 Copper 仓库，工具拒绝更改。`--update` 仅在工作区干净时执行 `git fetch` 和 `git merge --ff-only`。没有 `--update` 时不自动同步远程新提交，避免意外改变现有源代码。`--recreate-env` 只删除由 Copper 安装器标记并管理的项目 `.venv`，不会删除权重、数据、日志或其他虚拟环境。如果已有虚拟环境不属于 Copper，请自行备份/移走后再运行。

**幂等性**：第二次运行会识别已管理的相同 CUDA profile，复用 `.venv` 并校验安装；要求切换 CUDA 档位时，明确使用 `--recreate-env`。安装若中途失败，保留 `.venv` 和日志以供排查；重新运行相同命令可继续修复环境。

## 4. 安装完成后

完成后项目树含 `.venv/`（由 `.gitignore` 排除），并生成：

```text
.venv/copper-install-state.json
.venv/copper-install-report.json
```

报告记录 OS、GPU、驱动、Python/PyTorch/TorchVision、CUDA profile、Git 提交、实际前向验证结果及安装时间；不含 GitHub Token。

Linux（项目目录）：

```bash
.venv/bin/python scripts/copper.py --help
.venv/bin/python scripts/copper.py train --data ./data/copper --labels ./configs/class_labels_csd.txt --output ./outputs/train
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts\copper.py --help
.\.venv\Scripts\python.exe scripts\copper.py train --data .\data\copper --labels .\configs\class_labels_csd.txt --output .\outputs\train
```

数据集、**Meta 官方预训练权重下载链接**、训练/测试、单图/批量推理及预测数据整理模板参见根目录 [`README.md`](README.md) 或 [`README.zh-CN.md`](README.zh-CN.md)。安装器**不会下载数据集和训练权重**，也不会隐式开始训练。

## 5. 故障排查与验证边界

- **Git clone 报 `Permission denied (publickey)`**：这是 SSH 认证错误。公开仓库请使用本文的 HTTPS clone 地址，无需 SSH Key。不要在 `--repo-url` 中嵌入 token。
- **找不到 GPU / CUDA 驱动太旧**：检查 `nvidia-smi`；更新驱动或明确选择 CPU。安装器不会帮你修改驱动。
- **已安装 CUDA wheel 但 Smoke Test 失败**：查看错误，检查 GPU Compute Capability、实际设备运算和 wheel；失败不会写入“已验证成功”报告。
- **依赖下载失败**：检查 GitHub、`astral.sh`、PyTorch wheel index、代理和证书。失败后可重复运行，不会清除已有数据。
- **虚拟环境冲突**：必须明确 `--recreate-env` 并且 `.venv` 为 Copper 管理；不要直接删除其他项目的虚拟环境。
- **大文件与磁盘不足**：CUDA 版 PyTorch 依赖下载体积可能较大，检查剩余空间。环境文件与缓存放在用户可写位置。
- **测试范围**：CI 对 Linux/Windows 运行语法与模拟安装策略测试。完整 GPU 依赖安装和实际模型 Smoke Test 必须在目标 GPU 主机上执行；不能将无 GPU 的 CI 通过解释为所有显卡均可运行。

PyTorch 官方档位：[Previous versions](https://pytorch.org/get-started/previous-versions/)；NVIDIA 驱动要求：[CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)；uv 官方安装说明：[Installation](https://docs.astral.sh/uv/getting-started/installation/)。
