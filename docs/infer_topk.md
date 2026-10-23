# 推理时输出低置信度样本的 Top-k

`infer_topk.py` 是独立的纯推理入口。它只执行 `model.eval()` 和 `torch.inference_mode()`，严格加载完整模型权重；不调用训练入口、优化器或训练数据增强。

在最小部署包根目录放置本脚本即可使用。依赖包内已有的 `datasets.py`、`models/self/convnextv2_modified.py`、`models/utils.py`，以及已有的 PyTorch、TorchVision、timm、NumPy、Pillow。

```bash
python infer_topk.py \
  --data /path/to/test \
  --checkpoint /path/to/checkpoint-best.pth \
  --class-labels /path/to/class_labels.txt \
  --output /path/to/output \
  --k 3 --conf 0.7 \
  --batch-size 64 --num-workers 10 --amp true --seed 0
```

- `--k` 和 `--conf` 必填。k 必须为整数，范围为 `[1, 类别数]`；例如七类模型可取 1～7，`3.0` 不被接受。
- conf 为 `[0, 1]` 内的有限数；拒绝 NaN、正负无穷和越界值。
- 只有未经四舍五入的 `top1_conf < conf` 时才输出候选；恰好等于 conf 不触发。conf=0 不触发任何样本，conf=1 仍不包含置信度恰好为1的样本。
- 候选包含 Top-1 本身，总计 k 个类别；置信度使用原始 softmax 概率，不在候选之间重新归一化，不自动改变最终预测。
- 同一次推理按 logits 降序排列类别；相同 logits 按类别序号排序。AMP 下 softmax 保留 FP16，与现有正式评估的概率精度一致；关闭 AMP 使用 FP32。
- 标签文件必须按 `0:类别名` 起依次编号、名称唯一。类别数从该文件推导，并与检查点及完整分类头严格核对，不允许丢弃分类头权重后继续推理。
- 输入支持 ImageFolder、普通图片目录及单张图片。相对输入根目录的第一层目录名匹配标签时记录真值；普通无标签图片仍可推理，不计算准确率。
- 保持原始 200×200 尺度、灰度解码及三通道归一化；其他尺寸写入排除清单，不参与推理。无需训练集目录。

输出文件：

| 文件 | 内容 |
|---|---|
| `all_predictions.jsonl` | 全部有效图像的 Top-1；只有低置信度样本的 `topk` 非空 |
| `low_confidence_topk.jsonl` | 仅触发阈值的样本及 k 个类别与置信度 |
| `low_confidence_topk.csv` | 同上，展开为 `top1_label,top1_conf,...,topk_label,topk_conf` |
| `summary.json` | 统计、类别精确率/召回率、低置信度检出情况、参数和权重哈希 |
| `excluded_non_200x200.csv` | 原始尺寸不合要求的样本 |

标准输出为低置信度样本逐行 JSON；标准错误输出汇总。可用 `> low_confidence.stdout.jsonl 2> summary.stderr.log` 保存。CPU 部署显式指定 `--device cpu`，此时不使用 AMP。

测试：`python -m unittest discover -s tests -p test_infer_topk.py -v`。
