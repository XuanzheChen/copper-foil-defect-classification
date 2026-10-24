# 1023 图片整合：七类模型推理与整理

已完成上传、七类模型推理及按 Top-1 标签整理。实验运行：`20261024T094716Z-20261024-1747-1023-7classes-inference-organize`，推理源码提交为 `c9eb6d4df6a5b5825432e6e2366e177c034b5e0c`。本地输入为项目根目录的 `Data/data/1023-图片整合/`，服务器原始文件完整保存在 `/share/data/cxz/1023-images-integrated/input/`。

## 输入快照

| 项目 | 数量 |
|---|---:|
| 全部文件 | 31458 |
| 图片 | 31455 |
| 200×200 图片 | 31317 |
| 其他尺寸图片 | 138 |
| XML/XSL 附件 | 3 |
| 原始字节数 | 453960345 |

三个附件为 `productions.xml`、`productions.xsl` 和 `productionsExcel.xml`。所有文件均保留原名和内容，未删除重复图像。按用户确认的规则，138 张非 200×200 图像保留原文件并列入排除清单，不缩放、不产生类别预测。

本地上传归档位于项目根目录的 `temp/20261024-1023-images-upload.tar.gz`，408035799 字节，SHA-256 为 `4b2168b0a96d46565e9f3dc55dde54301c6beb9d69ee7e5e5bda8dfa62dc415f`。内含 `input/` 原始文件及逐文件哈希清单 `dataset_manifest.json`，清单 SHA-256 为 `f3fa8a7bd5671911d5772d86b501336a444612d07a8f25220e07ec6df6a319bb`。

## 整理结果

服务器实验根目录：

```text
/home/cxz/projects/copper-foil-defect-classification/experiments/20261024T094716Z-20261024-1747-1023-7classes-inference-organize/
```

分类图片位于上述目录的 `by_top1/<类别>/<原相对路径>`。按复制方式整理，原图保留；每张图片与上传清单哈希一致。

| Top-1 类别 | 图片数 | 其中置信度 <0.7 |
|---|---:|---:|
| 凹凸 | 94 | 86 |
| 刮伤 | 436 | 435 |
| 压坑 | 6595 | 2532 |
| 异物 | 242 | 153 |
| 拖尾 | 210 | 207 |
| 铜瘤 | 23691 | 3038 |
| 黑点 | 49 | 49 |
| 合计 | 31317 | 6500 |

- [全部 Top-1 预测 CSV](../experiments/20261024T094716Z-20261024-1747-1023-7classes-inference-organize/top1_predictions.csv)：31317 条，包含原相对路径、整理后相对路径、Top-1 标签、置信度和原文件 SHA-256。
- [低置信度 Top-3 JSON](../experiments/20261024T094716Z-20261024-1747-1023-7classes-inference-organize/low_confidence_top3.json)：6500 条，占有效图片的 20.7555%。JSON SHA-256 为 `1a06056fc307daac4ac9f98dc4168ed3f691491e0f253db53039b8ed340c8f95`。
- [非标准尺寸排除清单](../experiments/20261024T094716Z-20261024-1747-1023-7classes-inference-organize/inference/excluded_non_200x200.csv)：138 条，包含原图路径与尺寸。
- [整理统计](../experiments/20261024T094716Z-20261024-1747-1023-7classes-inference-organize/organization_summary.json)与[独立验证记录](../experiments/20261024T094716Z-20261024-1747-1023-7classes-inference-organize/final_verification.json)。

JSON 仅收录未经四舍五入的 Top-1 置信度严格小于 `conf=0.7` 的样本，包含源路径、整理后路径、Top-1 标签与置信度，以及 `top3` 中三个候选的类别 ID、类别名称和原始 softmax 概率。Top-3 包含 Top-1，不重新归一化、不自动改判。该批数据无人工真值，因此不计算分类准确率。

## 模型与执行参数

使用本次去重训练的七类最佳检查点：`experiments/20261023T150120Z-20261023-2301-1017-301-dedup-7classes-train/checkpoint-best.pth`，SHA-256 为 `8b8bce7c740320e5fe46ad82a2e4495a4760f6e6030fd9045123eafaf562b489`。标签取自已验证的七类最小部署包，字节 SHA-256 为 `5b18a8b194c3d417e97590ca2dd51c0da5e17853fb1e9be158b2a4e94bfa5670`。

推理前验证模型定义、预处理代码和 `infer_topk.py` 与最小部署包逐文件哈希一致。参数为 `k=3`、`conf=0.7`、batch64、workers10、seed0、CUDA AMP；严格加载完整权重。推理循环耗时 20.2362 秒，1547.58 张/秒。完整命令与配置见本实验的 `command.sh`、`model_provenance.json` 和 `inference/summary.json`。

整理使用已提交的 `scripts/organize-inference-results.py`，直接在服务器运行，避免再次上传分类图片。该脚本验证全部合规图像恰好有一条预测、类别 ID 与名称一致、阈值为严格比较、复制内容与上传快照相同，并拒绝覆盖已有整理结果；已通过 6 项测试，既有 Top-k 规则的 8 项测试也通过。

## 验证记录

原包装命令在所有推理、整理和校验文件生成后，末尾 Shell here-document 结束符触发 `NameError: name 'PY' is not defined`，因此保留原始 `exit_code.txt = 1` 和错误日志，未改写退出记录。

随后独立复核全部 31458 个原始文件、31317 张分类副本的哈希，逐条核对全部 Top-1 CSV 与推理结果、6500 条 Top-3 JSON 与阈值及候选、138 条排除清单的路径和尺寸，以及权重与标签哈希，全部通过。实际独立验证退出码为 0，记录于 `final_verification_exit_code.txt`；`final_verification.json` 同时记录原包装错误与最终有效性结论。此 Shell 结束符问题未影响任何预测或整理文件，临时启动命令已补齐末尾换行。
