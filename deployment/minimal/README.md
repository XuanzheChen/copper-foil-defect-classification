# 铜箔瑕疵分类：七类最小部署包

本包包含本次去重训练的最佳权重、配套七类标签、完整模型定义和纯推理Top-k脚本。解压后，以含本README的目录作为包根目录。

模型为ConvNeXt V2 Tiny，输入保持原始 **200×200** 尺度；先解码为灰度，再扩为三通道并归一化。标签顺序：**凹凸、刮伤、压坑、异物、拖尾、铜瘤、黑点**。

## 环境

推荐Python 3.10。GPU验证环境：PyTorch 2.7.1、TorchVision 0.22.1、CUDA Runtime 12.8。

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python verify_package.py
```

`verify_package.py`仅使用Python标准库，检查全部随包文件、权重与标签的SHA-256。推理输出请放在包目录外，以保留完整性校验的文件集合。无需训练集或额外预训练权重。

## 直接推理

Linux入口自动定位随包权重与标签，默认`k=3`、`conf=0.7`，只需提供图片位置和输出目录：

```bash
bash infer.sh --data /path/to/images --output /path/to/inference-results
```

完整测试的复现参数：

```bash
bash infer.sh --data /path/to/1017-301-plus-formal-split-7classes/test \
  --output /path/to/inference-results --k 3 --conf 0.7 \
  --batch-size 64 --num-workers 10 --amp true --seed 0
```

从其他工作目录执行时，给出入口绝对路径即可：

```bash
bash /path/to/copper-foil-7class/infer.sh --data /path/to/images --output /path/to/inference-results
```

CPU模式加入`--device cpu --amp false --num-workers 0`。Python入口也支持Windows；以下相对路径命令在包根目录执行：

```bash
python infer_topk.py --data /path/to/images --output /path/to/inference-results \
  --checkpoint weights/checkpoint-best.pth --class-labels configs/class_labels_csd.txt \
  --k 3 --conf 0.7 --device cpu --amp false --num-workers 0
```

输入可为ImageFolder、无标签图片目录或单张图像。相对输入根目录的第一层目录名匹配标签时记录真值并统计准确率。非200×200图像写入排除清单并跳过，不做缩放。

## Top-k规则与输出

- `k`必须是`[1,7]`内的整数，拒绝`3.0`；`conf`必须是`[0,1]`内的有限数，拒绝NaN、无穷值及越界值。
- 只有未经四舍五入的 **Top-1置信度 < conf** 时输出k个候选；等于阈值不触发。候选包含Top-1本身，保留原始softmax概率，不重新归一化，也不自动改判。
- CUDA AMP使用与正式测试一致的FP16 softmax；CPU和关闭AMP时使用FP32。硬件、精度和batch变化可能影响临界样本。
- 仅执行`eval()`和`inference_mode()`，完整模型及分类头严格加载。

| 文件 | 内容 |
|---|---|
| `all_predictions.jsonl` | 全部有效图片的Top-1；低置信度图片同时有Top-k候选 |
| `low_confidence_topk.csv` / `.jsonl` | 触发阈值的图片、候选类别及置信度 |
| `summary.json` | 准确率、阈值统计、样本数、权重哈希、参数及速度 |
| `excluded_non_200x200.csv` | 被排除的图片路径和尺寸 |

标准输出为低置信度样本逐行JSON，标准错误为统计汇总，可重定向到包外文件。

## 文件和版本

```text
copper-foil-7class/
  infer.sh
  infer_topk.py
  datasets.py
  models/__init__.py
  models/self/__init__.py
  models/self/convnextv2_modified.py
  models/utils.py
  configs/class_labels_csd.txt
  weights/checkpoint-best.pth
  verify_package.py
  requirements.txt
  model_info.json
  PACKAGE_MANIFEST.json
  README.md
  LICENSE
```

训练运行：`20261023T150120Z-20261023-2301-1017-301-dedup-7classes-train`。200轮训练成功，最佳权重对应零起始epoch29。随包提供原始训练检查点。

- 权重SHA-256：`8b8bce7c740320e5fe46ad82a2e4495a4760f6e6030fd9045123eafaf562b489`
- 标签SHA-256：`5b18a8b194c3d417e97590ca2dd51c0da5e17853fb1e9be158b2a4e94bfa5670`
- 构建源码提交和逐文件SHA-256见`PACKAGE_MANIFEST.json`，训练元数据见`model_info.json`。

此前独立测试：2236张中排除22张非200×200图像，实际2214张，2198正确、16错误，Top-1为99.2773%。CUDA AMP、batch64、workers10、seed0下，`conf=0.7,k=3`输出61张，含12张错分和49张正确，60张的真值在Top-3中。本次整包复验记录随交付提供。
