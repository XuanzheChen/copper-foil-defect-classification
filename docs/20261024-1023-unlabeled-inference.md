# 1023 图片整合：七类模型推理与整理

输入位于本地项目根目录的 `Data/data/1023-图片整合/`。当前已完成原图校验、上传快照打包和分类整理程序测试；服务器连接不稳定，上传未完成，尚未启动本次推理，也未生成预测标签。

## 输入快照

| 项目 | 数量 |
|---|---:|
| 全部文件 | 31458 |
| 图片 | 31455 |
| 200×200 图片 | 31317 |
| 其他尺寸图片 | 138 |
| XML/XSL 附件 | 3 |
| 原始字节数 | 453960345 |

三个附件为 `productions.xml`、`productions.xsl` 和 `productionsExcel.xml`。所有文件均保留原名；未删除重复图像。沿用既有模型输入规则时，138 张非 200×200 图像保留原文件并列入排除清单，不缩放、不产生类别预测。若用户另行指定缩放，需单独记录该预处理及结果。

本地上传归档位于项目根目录的 `temp/20261024-1023-images-upload.tar.gz`，408035799 字节，SHA-256 为 `4b2168b0a96d46565e9f3dc55dde54301c6beb9d69ee7e5e5bda8dfa62dc415f`。内含 `input/` 原始文件及逐文件哈希清单 `dataset_manifest.json`，清单 SHA-256 为 `f3fa8a7bd5671911d5772d86b501336a444612d07a8f25220e07ec6df6a319bb`。

## 已准备的处理流程

1. 完成上传并核对归档 SHA-256，检查安全相对路径和逐文件哈希后解压。
2. 通过标准实验包装脚本执行已提交的 `infer_topk.py`，使用本次去重训练的最佳检查点、`deployment/minimal/configs/class_labels_csd.txt` 七类标签以及 `--k 3 --conf 0.7`。
3. 待实验结束，一次性回传文本产物到本地相同的 `experiments/<run>/` 路径。
4. 用本地原图、上传清单及服务器预测运行下面的整理程序；生成的图片数据再上传到同一服务器实验目录，验证每张图片哈希。
5. 最后按仓库约定提交文本产物并完成三端同步。

原始最佳检查点：`experiments/20261023T150120Z-20261023-2301-1017-301-dedup-7classes-train/checkpoint-best.pth`；SHA-256 为 `8b8bce7c740320e5fe46ad82a2e4495a4760f6e6030fd9045123eafaf562b489`。

```bash
python scripts/organize-inference-results.py \
  --data /path/to/local/1023-images \
  --predictions experiments/RUN/inference/all_predictions.jsonl \
  --summary experiments/RUN/inference/summary.json \
  --manifest /path/to/dataset_manifest.json \
  --class-labels deployment/minimal/configs/class_labels_csd.txt \
  --output experiments/RUN
```

整理程序保留原始文件，按 Top-1 标签复制到 `by_top1/<类别>/<原相对路径>`；输出全部 Top-1 的 `top1_predictions.csv` 和单个 `low_confidence_top3.json`。JSON 仅收录 Top-1 置信度严格小于 0.7 的样本，记录源相对路径、整理后路径、Top-1 标签和置信度，以及含类别 ID、名称和置信度的三个候选。未四舍五入后筛选，候选概率不重新归一化，也不自动改判。

整理前会验证推理覆盖全部合规图像且无重复、预测标签顺序正确、低置信度 Top-3 与 Top-1 一致、原图与上传快照一致；整理后核验复制图像 SHA-256。已有输出不会被覆盖。同名不同目录文件保持各自相对路径。

已通过 6 项整理程序测试，以及 8 项既有 Top-k 规则测试。
