"""Analyze a paired reference evaluation and minimum-package Top-k inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def describe(values):
    if not values:
        return {"count": 0}
    return dict(count=len(values), minimum=min(values), maximum=max(values), mean=float(np.mean(values)),
                **{name: float(np.quantile(values, q)) for name, q in
                   (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-run", required=True)
    args = parser.parse_args()
    root = args.run_dir
    reference = read_csv(root / "reference/eval_predictions.csv")
    predictions = [json.loads(s) for s in (root / "topk/all_predictions.jsonl").read_text().splitlines()]
    summary = json.loads((root / "topk/summary.json").read_text())
    labels = [line.split(":", 1)[1] for line in (root / "class_labels.txt").read_text().splitlines()]
    ref = {r["path"]: r for r in reference}
    candidate = {r["path"]: r for r in predictions}
    assert len(ref) == len(candidate) == len(predictions) == 2214
    assert ref.keys() == candidate.keys()
    label_differences = [p for p, r in ref.items() if r["pred_label"] != candidate[p]["pred_label"]]
    probability_difference = max(abs(float(r["pred_prob"]) - candidate[p]["top1_conf"]) for p, r in ref.items())
    comparison = dict(samples=2214, label_differences=label_differences,
                      max_top1_conf_difference=probability_difference,
                      strict_weights_loaded=summary["strict_weights_loaded"])
    write_json(root / "reference_comparison.json", comparison)
    assert not label_differences and probability_difference == 0, comparison
    assert summary["k"] == 3 and summary["conf"] == .7
    assert summary["checkpoint_sha256"] == "8b8bce7c740320e5fe46ad82a2e4495a4760f6e6030fd9045123eafaf562b489"
    assert summary["excluded_non_200x200"] == 22
    errors = sorted((r for r in predictions if not r["correct"]), key=lambda r: -r["top1_conf"])
    correct = [r for r in predictions if r["correct"]]
    low = [r for r in predictions if r["low_confidence"]]
    low_saved = [json.loads(s) for s in (root / "topk/low_confidence_topk.jsonl").read_text().splitlines()]
    assert low == low_saved
    assert len(low) == sum(float(r["pred_prob"]) < .7 for r in reference)
    for r in low:
        assert len(r["topk"]) == 3 and r["top1_conf"] < .7
        assert r["topk"][0]["class_name"] == r["pred_label"]
        assert r["topk"][0]["confidence"] == r["top1_conf"]
    top5_correct = sum(
        labels.index(r["true_label"]) in [int(i) for i in re.findall(r"(\d+)\([^)]+\)", r["top5"])]
        for r in reference
    )
    matrix = [[sum(r["true_label"] == a and r["pred_label"] == b for r in predictions) for b in labels] for a in labels]
    per_class = []
    for i, name in enumerate(labels):
        support = sum(matrix[i]); predicted = sum(row[i] for row in matrix); tp = matrix[i][i]
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        per_class.append(dict(class_name=name, support=support, correct=tp, errors=support-tp,
                              precision=precision, recall=recall,
                              f1=2*precision*recall/(precision+recall) if precision+recall else 0))
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    histogram = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        inside = lambda x: lo <= x < hi or hi == 1 and x == 1
        histogram.append(dict(interval=f"[{lo:.1f}, {hi:.1f}{']' if hi == 1 else ')'}",
                              errors=sum(inside(r["top1_conf"]) for r in errors),
                              correct=sum(inside(r["top1_conf"]) for r in correct)))
    error_values = [r["top1_conf"] for r in errors]
    correct_values = [r["top1_conf"] for r in correct]
    log = (root / "reference/stdout.log").read_text()
    fps = float(re.findall(r"Evaluation speed: ([\d.]+) images/s", log)[-1])
    result = dict(samples=len(predictions), excluded=22, correct=len(correct), errors=len(errors),
                  accuracy=len(correct)/len(predictions), top5_correct=top5_correct,
                  top5_accuracy=top5_correct/len(predictions),
                  macro_recall=float(np.mean([r["recall"] for r in per_class])),
                  macro_f1=float(np.mean([r["f1"] for r in per_class])), per_class=per_class,
                  reference_fps=fps, script_fps=summary["images_per_second"],
                  low_confidence=summary, incorrect_confidence=describe(error_values),
                  correct_confidence=describe(correct_values), histogram=histogram,
                  high_confidence_errors=sum(r["top1_conf"] >= .7 for r in errors),
                  errors_at_least_09=sum(r["top1_conf"] >= .9 for r in errors),
                  low_confidence_error_recall=sum(r["low_confidence"] for r in errors)/len(errors) if errors else None,
                  reference_comparison=comparison)
    write_json(root / "confidence_analysis.json", result)
    with (root / "error_confidence_distribution.csv").open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["interval","errors","correct"]); writer.writeheader(); writer.writerows(histogram)
    with (root / "misclassified_confidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["path","true_label","pred_label","top1_conf","low_confidence"])
        writer.writeheader(); writer.writerows({k:r[k] for k in writer.fieldnames} for r in errors)
    lines = ["# 1017-301 去重七类模型测试与低置信度 Top-k", "",
             f"训练运行：`{args.training_run}`；训练成功完成200轮，使用epoch29选择的最佳权重。", "",
             f"测试执行代码提交：`{(root/'git_commit.txt').read_text().strip()}`。",
             f"检查点SHA-256：`{summary['checkpoint_sha256']}`。", "",
             "固定独立test划分2236张，排除22张非200×200图像，实际评估2214张。训练/验证/测试内容SHA-256无重复。",
             "标准评估与最小部署依赖下的新脚本均使用batch64、workers10、CUDA AMP、seed0、原生200×200灰度流程。", "",
             f"- 正确{len(correct)}张，错误{len(errors)}张，Top-1准确率{100*result['accuracy']:.6f}%。",
             f"- Top-5准确率{100*result['top5_accuracy']:.6f}%；宏平均召回率{100*result['macro_recall']:.6f}%；宏F1 {100*result['macro_f1']:.6f}%。",
             f"- 标准评估实测{fps:.2f}张/秒；额外脚本实测{summary['images_per_second']:.2f}张/秒。速度含数据加载与首批开销，不是预热后的纯模型吞吐。",
             "- 两次独立推理的逐图Top-1类别及Top-1概率完全一致，分类头严格加载。", "",
             "## 各类别指标", "", "| 类别 | 有效样本 | 正确 | 错误 | 精确率 | 召回率（类内准确率） |", "|---|---:|---:|---:|---:|---:|"]
    for r in per_class:lines.append(f"| {r['class_name']} | {r['support']} | {r['correct']} | {r['errors']} | {100*r['precision']:.4f}% | {100*r['recall']:.4f}% |")
    lines += ["", "## conf=0.7、k=3 实际输出", "",
              f"严格使用Top-1概率<0.7，共输出{len(low)}张：其中错分{summary['low_confidence_errors']}张，正确{summary['low_confidence_correct']}张。",
              f"这些样本中，真实类别进入Top-3的有{summary['low_confidence_topk_contains_truth']}张。候选输出不替换Top-1预测，也不改变模型准确率。", "",
              "完整原始结果：`topk/low_confidence_topk.csv`、`topk/low_confidence_topk.jsonl`；控制台逐行JSON见`topk/stdout.jsonl`。", "",
              "| 样本（相对test） | 真值 | Top-1（conf） | Top-2（conf） | Top-3（conf） |", "|---|---|---|---|---|"]
    for r in low:
        relative=Path(r['path']).relative_to(Path(summary['data'])).as_posix()
        cells=[f"{c['class_name']} ({c['confidence']:.6f})" for c in r['topk']]
        lines.append(f"| {relative} | {r['true_label']} | {' | '.join(cells)} |")
    lines += ["", "## 错分样本Top-1置信度分布", "",
              f"统计：`{json.dumps(result['incorrect_confidence'], ensure_ascii=False)}`。", "",
              f"阈值0.7检出{summary['low_confidence_errors']}/{len(errors)}个错误；仍有{result['high_confidence_errors']}个错误的置信度≥0.7，其中{result['errors_at_least_09']}个≥0.9。softmax高置信度不保证预测正确。", "",
              "| Top-1置信度区间 | 错分数 | 正确数 |", "|---|---:|---:|"]
    for r in histogram:lines.append(f"| {r['interval']} | {r['errors']} | {r['correct']} |")
    lines += ["", "### 错分明细（按置信度降序）", "", "| 样本（相对test） | 真值 | 预测 | Top-1 conf |", "|---|---|---|---:|"]
    for r in errors:
        relative=Path(r['path']).relative_to(Path(summary['data'])).as_posix()
        lines.append(f"| {relative} | {r['true_label']} | {r['pred_label']} | {r['top1_conf']:.6f} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].hist(error_values,bins=bins,color='#d85b50',edgecolor='white')
    axes[0].axvline(.7,color='#333333',linestyle='--',label='conf=0.7')
    axes[0].set(xlim=(0,1),xlabel='Top-1 softmax confidence',ylabel='Misclassified image count',title=f'Errors: {len(errors)} / {len(predictions)}')
    axes[0].legend()
    for values,label,color in ((correct_values,'Correct','#327b92'),(error_values,'Incorrect','#d85b50')):
        if values:axes[1].step(sorted(values),np.arange(1,len(values)+1)/len(values),where='post',label=f'{label} (n={len(values)})',color=color)
    axes[1].axvline(.7,color='#333333',linestyle='--')
    axes[1].set(xlim=(0,1),ylim=(0,1.02),xlabel='Top-1 softmax confidence',ylabel='Cumulative fraction',title='Confidence distribution (ECDF)')
    axes[1].legend();fig.savefig(root/'confidence_distribution.png',dpi=180);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('samples','correct','errors','accuracy','reference_fps','incorrect_confidence','high_confidence_errors','errors_at_least_09')},ensure_ascii=False,indent=2))
    print('LOW_CONFIDENCE_SAMPLES',len(low), 'LOW_CONFIDENCE_ERRORS',summary['low_confidence_errors'])


if __name__ == '__main__':
    main()
