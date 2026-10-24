# 七类最小部署包交付验证

部署包包含去重训练的原始最佳检查点、配套七类标签、纯推理 Top-k 脚本、完整模型定义、依赖清单和 README。运行时无需项目仓库或训练数据。

## 交付文件

- 本地 ZIP：`../docs/copper-foil-7class-20261024.zip`（相对于仓库根目录）。
- 本地解压目录：`../docs/20261024-七类铜箔瑕疵分类最小部署包/`。
- ZIP 大小：334806125 字节。
- ZIP SHA-256：`f6a63a80b51bf099c174c8bac9feeb0cb4671f502e16d3d09e4d6c1c68fe7e23`。
- 构建源码提交：`2392aabf9fbf4099a62cf7513f688657bccdaa7b`。
- 训练运行：`20261023T150120Z-20261023-2301-1017-301-dedup-7classes-train`。
- 最佳权重：零起始 epoch 29，SHA-256 `8b8bce7c740320e5fe46ad82a2e4495a4760f6e6030fd9045123eafaf562b489`。
- 标签顺序：凹凸、刮伤、压坑、异物、拖尾、铜瘤、黑点。

## 使用

先按随包 README 安装依赖，再运行：

```bash
python verify_package.py
bash infer.sh --data /path/to/images --output /path/to/results
```

入口自动定位权重和标签，默认 `k=3, conf=0.7`。支持单图、无标签目录和 ImageFolder；只有 Top-1 置信度严格低于阈值时输出 Top-k。`k` 必须是 1 到 7 的整数，`conf` 必须是 0 到 1 的有限数。输入保持 200×200，其他尺寸跳过并记录。

## 实际验证

验证运行：`20261024T014207Z-20261024-0942-7classes-minimal-deployment-verify`；服务器返回 `exit_code.txt = 0` 和 `deployment_verification.json` 的 `verified = true`。

服务器使用相同源码重建 ZIP，确认与本地交付 ZIP 哈希相同，并从 ZIP 解压后的目录运行。工作目录为 `/tmp`，清除 `PYTHONPATH`，禁用用户 site-packages，不传入权重、标签、k 或 conf 参数。以下结论来自该次实际运行返回的验证记录：

| 检查 | 结果 |
|---|---|
| ZIP | 15 个文件，CRC 和 SHA-256 通过；与本地交付逐字节一致 |
| 文件完整性 | 推理前后均通过全部随包文件校验 |
| 完整测试 | 2214 张，2198 正确、16 错误，准确率 99.2773261% |
| 与此前正式测试比较 | 全部样本的 Top-1 类别、置信度及 Top-k 结果完全相同 |
| 默认 `k=3, conf=0.7` | 61 张低置信度，含 12 张错分和 49 张正确；60 张真值在 Top-3 中 |
| CUDA 默认单图推理 | 通过 |
| CPU 单图推理 | 通过（关闭 AMP） |
| 非法参数 | `k=8`、`k=3.0`、`conf=1.1`、`conf=nan` 均退出码 2，未创建输出目录 |
| 自动化测试 | 5 项部署构建测试和 8 项 Top-k 测试通过 |

本报告在本地编写，依据已读取的服务器终态记录。2026-10-24 按用户要求清理服务器端本次部署验证产物：删除 25 个顶层文件/目录，包括原始日志、测试输出、构建目录和解压副本，以及 `/tmp/20261024-minimal-deployment-text.tar.gz` 临时回传压缩包。验证目录内释放 670400375 字节，仅保留下列已验证的最小部署推理 ZIP：

```text
/home/cxz/projects/copper-foil-defect-classification/experiments/20261024T014207Z-20261024-0942-7classes-minimal-deployment-verify/copper-foil-7class-20261024.zip
```

删除前校验 ZIP 的 15 个文件、CRC、逐文件 SHA-256、七类标签和权重；删除后复核 ZIP 大小与上述哈希一致。原始验证日志未回传入库；本地交付文件和此验证结论保留。
