# RACP / EFC-RAG 论文缺失信息只读审计

审计时间：2026-07-14（Asia/Shanghai）
审计范围：`PAPER_RESULTS_REFERENCE.md` 中的 26 个 FINAL 主实验、`PAPER_EXPERIMENT_PLAN.md`、全部相关输出、当前代码与 Git 历史。
审计约束：本次没有修改实验代码、删除文件、停止或启动实验，也没有重新调用生成模型或检索器；只新增本报告。

> **2026-07-15 补充审计：** HotpotQA `HP-AB-02/03/04/06/07/08/09/10/11/12/13`
> 已按同一验收清单登记为 FINAL。本文第七节、最终缺口分类和优先级已同步更新；
> 文末“当前现场快照”仍保留2026-07-14原始审计现场，不应解读为当前进程状态。

## 审计口径与最重要结论

- 26 个 FINAL 均有 `config.yaml`、`metric_score.txt` 和 `intermediate_data.json`；逐样本 ID 与目标 split 的数量、集合和顺序完全一致，没有缺失预测字段或逐样本 EM/F1。
- 现有名为 `Retrieval Recall` 的正式指标实际是“最终 `retrieval_result[:K]` 是否包含任一 gold answer 字符串”的 **answer-containment hit rate**，不是 supporting-document/title recall。IterRetGen 虽标作 `@10`，实际只评第三轮 5 篇文档；FLARE 评的是最后一次触发的 0/5 篇文档。
- 只有 7 个 FINAL 保存了实际命令。其余 19 个只能恢复生效配置，不能把台账中的模板命令冒充实际命令，因此本报告写为 `UNKNOWN`。
- 所有 FINAL 可证的生成/采样 seed 都是 2024。部分日志另显示 vLLM engine 初始化默认 seed=0；它不是第二个实验采样 seed，缺日志时也不能外推。
- 没有 FINAL 保存进程峰值显存监控；日志中的约 14.99 GiB 权重加载量不是峰值，峰值显存全部为 `UNKNOWN`。
- HotpotQA 的 generation-guided-only 消融已全量完成，但尚未登记为 FINAL。本报告按允许枚举标作 `PARTIAL（FULL-COMPLETE/UNREGISTERED）`，不把它提升为正式结果。
- EFC 的正式参数主要在 HotpotQA dev 上演化，多项改动彼此混杂；现有文件不能证明逐参数的独立选择依据或系统网格搜索。
- 缓存的设计意图是只影响速度；现有对照支持 Hotpot 公共缓存为 raw dense，但没有成对 FINAL 证明 prediction/metric 等价，故结果等价性记为 `UNKNOWN`。详见第一、八节。

记号：`HP`=HotpotQA dev，`2W`=2WikiMultiHopQA dev，`MU`=MuSiQue dev，`NQ`=Natural Questions test；`D` 表示相应的 `racp/output/<目录>`。

# 一、FINAL 实验复现信息

## 1.1 样本覆盖与产物

所有行的 `test_sample_num` 都是 `null`。覆盖审计使用本地 split 原始文件逐 ID、逐顺序核对：HP 7,405/7,405，2W 12,576/12,576，MU 2,417/2,417，NQ 3,610/3,610。

| ID | 数据集 | 方法 | 输出目录 D | 样本数 / split 覆盖 | 配置 / 指标 | 实际命令 |
|---|---|---|---|---|---|---|
| HP-NR-01 | HP dev | No-RAG | [D](output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full/) | 7,405 / 完整 | [config](output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full/config.yaml) / [metric](output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full/metric_score.txt) | `UNKNOWN` |
| HP-NR-02 | HP dev | Standard RAG | [D](output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full/) | 7,405 / 完整 | [config](output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full/config.yaml) / [metric](output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full/metric_score.txt) | `UNKNOWN` |
| HP-NR-03 | HP dev | IterRetGen | [D](output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3/) | 7,405 / 完整 | [config](output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3/config.yaml) / [metric](output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3/metric_score.txt) | `UNKNOWN` |
| HP-NR-04 | HP dev | Full-QD | [D](output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full/) | 7,405 / 完整 | [config](output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full/config.yaml) / [metric](output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full/metric_score.txt) | `UNKNOWN` |
| HP-NR-05 | HP dev | IRCoT | [D](output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full/) | 7,405 / 完整 | [config](output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full/config.yaml) / [metric](output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full/metric_score.txt) | `UNKNOWN` |
| HP-NR-07 | HP dev | EFC-RAG | [D](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full/) | 7,405 / 完整 | [config](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full/config.yaml) / [metric](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full/metric_score.txt) | `UNKNOWN` |
| HP-NR-08 | HP dev | FLARE | [D](output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2/) | 7,405 / 完整 | [config](output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2/config.yaml) / [metric](output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2/metric_score.txt) | [C1](output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2/run_command.txt) |
| 2W-NR-01 | 2W dev | No-RAG | [D](output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full/config.yaml) / [metric](output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full/metric_score.txt) | `UNKNOWN` |
| 2W-NR-02 | 2W dev | Standard RAG | [D](output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full/config.yaml) / [metric](output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full/metric_score.txt) | `UNKNOWN` |
| 2W-NR-03 | 2W dev | IterRetGen | [D](output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full/config.yaml) / [metric](output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full/metric_score.txt) | `UNKNOWN` |
| 2W-NR-04 | 2W dev | Full-QD | [D](output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full/config.yaml) / [metric](output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full/metric_score.txt) | `UNKNOWN` |
| 2W-NR-05 | 2W dev | IRCoT | [D](output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2/config.yaml) / [metric](output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2/metric_score.txt) | `UNKNOWN` |
| 2W-NR-07 | 2W dev | EFC-RAG | [D](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full/config.yaml) / [metric](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full/metric_score.txt) | `UNKNOWN` |
| 2W-NR-08 | 2W dev | FLARE | [D](output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full/) | 12,576 / 完整 | [config](output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full/config.yaml) / [metric](output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full/metric_score.txt) | [C2](output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full/run_command.txt) |
| MU-NR-01 | MU dev | No-RAG | [D](output/musique_2026_07_12_16_14_musique-zero-shot-full/) | 2,417 / 完整 | [config](output/musique_2026_07_12_16_14_musique-zero-shot-full/config.yaml) / [metric](output/musique_2026_07_12_16_14_musique-zero-shot-full/metric_score.txt) | `UNKNOWN` |
| MU-NR-02 | MU dev | Standard RAG | [D](output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full/) | 2,417 / 完整 | [config](output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full/config.yaml) / [metric](output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full/metric_score.txt) | `UNKNOWN` |
| MU-NR-03 | MU dev | IterRetGen | [D](output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full/) | 2,417 / 完整 | [config](output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full/config.yaml) / [metric](output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full/metric_score.txt) | `UNKNOWN` |
| MU-NR-04 | MU dev | Full-QD | [D](output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare/) | 2,417 / 完整 | [config](output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare/config.yaml) / [metric](output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare/metric_score.txt) | `UNKNOWN` |
| MU-NR-05 | MU dev | IRCoT | [D](output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full/) | 2,417 / 完整 | [config](output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full/config.yaml) / [metric](output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full/metric_score.txt) | `UNKNOWN` |
| MU-NR-07 | MU dev | EFC-RAG | [D](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare/) | 2,417 / 完整 | [config](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare/config.yaml) / [metric](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare/metric_score.txt) | `UNKNOWN` |
| MU-NR-08 | MU dev | FLARE | [D](output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full/) | 2,417 / 完整 | [config](output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full/config.yaml) / [metric](output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full/metric_score.txt) | [C3](output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full/run_command.txt) |
| NQ-NR-01 | NQ test | No-RAG | [D](output/nq_2026_07_12_19_21_nq-zero-shot-full/) | 3,610 / 完整 | [config](output/nq_2026_07_12_19_21_nq-zero-shot-full/config.yaml) / [metric](output/nq_2026_07_12_19_21_nq-zero-shot-full/metric_score.txt) | `UNKNOWN` |
| NQ-NR-02 | NQ test | Standard RAG | [D](output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full/) | 3,610 / 完整 | [config](output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full/config.yaml) / [metric](output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full/metric_score.txt) | [C4](output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full/run_command.txt) |
| NQ-NR-03 | NQ test | IterRetGen | [D](output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full/) | 3,610 / 完整 | [config](output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full/config.yaml) / [metric](output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full/metric_score.txt) | [C5](output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full/run_command.txt) |
| NQ-NR-05 | NQ test | EFC-RAG | [D](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full/) | 3,610 / 完整 | [config](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full/config.yaml) / [metric](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full/metric_score.txt) | [C6](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full/run_command.txt) |
| NQ-NR-06 | NQ test | FLARE | [D](output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full/) | 3,610 / 完整 | [config](output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full/config.yaml) / [metric](output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full/metric_score.txt) | [C7](output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full/run_command.txt) |

## 1.2 代码、时间、硬件、缓存与运行完整性

`Git / diff` 中的 commit 多由运行时间附近的 reflog/台账恢复，并非输出内嵌 provenance；diff 指实验运行时 diff。当前工作区的 dirty 状态不能反推早期实验的完整 diff，没有同期记录时必须是 `UNKNOWN`。seed 列是“生成采样 seed / engine-init seed”；前者由 config 证明为 2024，后者只有相应日志存在时才写 0，且不参与每个 request 的采样。时间戳是本地时间；`FS` 表示由文件时间戳恢复而非显式起止 marker。缓存列中的“设计为只提速”只表示预期语义；没有成对 FINAL 时，bitwise prediction/metric 等价性仍为 `UNKNOWN`。

| ID | Git commit / 运行时未提交 diff | 生成 seed / engine-init* | 开始 → 结束；总耗时 | GPU / TP | 缓存及结果语义 | 异常、续跑、合并 |
|---|---|---|---|---|---|---|
| HP-NR-01 | `3b4d780` / `UNKNOWN` | 2024 / 0 | 2026-07-12 14:26:40 → 14:27:53；约 0:01:13 FS | 1 / 1 | 无；N/A | 日志无异常；无续跑/合并证据 |
| HP-NR-02 | `3b4d780` / `UNKNOWN` | 2024 / 0 | 2026-07-12 14:26:41 → 14:49:40；约 0:22:59 FS | 0 / 1 | HP top20 公共缓存；预期只提速，结果等价 `UNKNOWN` | 日志无异常；无续跑/合并证据 |
| HP-NR-03 | `405d9cc` / `UNKNOWN` | 2024 / `UNKNOWN` | 2026-06-05 16:46:31 → 19:06:27；约 2:19:56 FS | 3 / 1 | 无 | 缺完整运行日志；异常/续跑 `UNKNOWN`；结果完整、无合并证据 |
| HP-NR-04 | `58dcbe6` / `UNKNOWN` | 2024 / 0 | 2026-07-10 09:27:20 → 11:02:43；1:35:23 marker | 2 / 1 | 原问题缓存；子问题动态检索；设计为只提速 | `full` 内部 prepare→generate；无异常、无结果合并 |
| HP-NR-05 | `58dcbe6` + 已知 IRCoT compatibility diff（完整 diff `UNKNOWN`） | 2024 / `UNKNOWN` | 2026-07-09 14:27:48 → 15:47:20；约 1:19:32 FS | 0,1 / 2 | 原问题缓存；thought query 动态检索；预期只提速 | 缺完整 run.log；异常/续跑 `UNKNOWN`；结果完整；运行时 dirty=YES |
| HP-NR-07 | `58dcbe6` / `UNKNOWN` | 2024 / 0 | 2026-07-09 15:52:58 → 16:52:00；0:59:02 marker | 2 / 1 | 历史 EFC cache chain，cache-only 重放；结果等价 `UNKNOWN` | `full` 内 prepare→generate；无异常、无合并 |
| HP-NR-08 | `3b4d780` + 已记录 FLARE compatibility diff | 2024 / 0 | 2026-07-13 09:22:59 → 14:14:00；约 4:51:01 FS | 0 / 1 | 加载公共 query cache；只在低置信动态 query 检索，无原问题初始检索 | v2 为独立重跑，不是合并；无异常 |
| 2W-NR-01 | `3b4d780` / 台账未记录 diff | 2024 / 0 | 2026-07-12 14:26:40 → 14:28:18；约 0:01:38 FS | 2 / 1 | 无 | 日志无异常；无续跑/合并证据 |
| 2W-NR-02 | `e6b4456` / 台账未记录 diff | 2024 / `UNKNOWN` | 2026-07-10 15:04:50 → 15:45:44；约 0:40:54 FS | 2 / 1 | 公共 top20 缓存；预期只提速 | 缺运行日志；异常/续跑 `UNKNOWN`；结果完整 |
| 2W-NR-03 | `3e62049` / 台账未记录 diff | 2024 / `UNKNOWN` | 2026-07-10 15:52:01 → 17:47:29；约 1:55:28 FS | 2 / 1 | 公共缓存 + 后续轮动态 miss；预期只提速 | 缺运行日志；异常/续跑 `UNKNOWN`；结果完整 |
| 2W-NR-04 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 14:31:13 → 17:04:26；2:33:13 marker | 1 / 1 | 公共缓存 + QD miss；设计为只提速 | `full` 内 prepare→generate；无异常、无合并 |
| 2W-NR-05 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 17:09:01 → 19:16:07；约 2:07:06 FS | 0,1,2,3 / 4 | 公共缓存 + thought miss；设计为只提速 | v1 失败目录与 v2 分离；v2 独立完成、未合并、日志无异常 |
| 2W-NR-07 | `3e62049` / 台账未记录 diff | 2024 / 0 | 2026-07-12 11:11:13 → 13:42:00；2:30:47 marker | 2 / 1 | 公共缓存 + expansion miss；设计为只提速 | `full` 内 prepare→generate；无异常、无合并 |
| 2W-NR-08 | `3b4d780` + FLARE compatibility diff | 2024 / 0 | 2026-07-13 14:26:27 → 20:41:20；约 6:14:53 FS | 1 / 1 | 加载公共 query cache；只在低置信动态 query 检索，无原问题初始检索 | 无异常、续跑或合并证据 |
| MU-NR-01 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 16:14:36 → 16:15:43；约 0:01:07 FS | 2 / 1 | 无 | 日志无异常；无续跑/合并证据 |
| MU-NR-02 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 19:41:04 → 19:50:29；约 0:09:25 wall | 1 / 1 | 独立 prepare 复用公共缓存，再用 prompt cache generate；只提速 | 明确两阶段；不是异常续跑或结果合并 |
| MU-NR-03 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 21:35:23 → 22:09:49；约 0:34:26 FS | 0 / 1 | 公共缓存 + 动态 miss；设计为只提速 | 日志无异常；无续跑/合并证据 |
| MU-NR-04 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 19:43:12 → 20:22:04；active 0:35:21，wall 0:38:52 | 0 / 1 | 公共缓存 + QD miss + prompt cache；只提速 | 手工 prepare/generate 两阶段；无异常、无结果合并 |
| MU-NR-05 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 22:30:36 → 23:01:40；约 0:31:04 FS | 0,1,2,3 / 4 | 公共缓存 + thought miss；设计为只提速 | 仅退出清理 warning；无失败、续跑或合并证据 |
| MU-NR-07 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 20:15:00 → 21:42:39；active 0:40:26，wall 1:27:39 | 0 / 1 | 公共缓存 + expansion miss + prompt cache；只提速 | 两阶段间 47:13 调度空档；非算法耗时、非异常续跑 |
| MU-NR-08 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-14 09:25:11 → 10:50:13；约 1:25:02 FS | 1 / 1 | 加载公共 query cache；只在低置信动态 query 检索，无原问题初始检索 | 无异常、续跑或合并证据 |
| NQ-NR-01 | `3b4d780` + 运行期 diff | 2024 / 0 | 2026-07-12 19:21:42 → 19:22:29；约 0:00:47 FS | 1 / 1 | 无 | 日志无异常；无续跑/合并证据 |
| NQ-NR-02 | `3b4d780` + compatibility diff | 2024 / 0 | 2026-07-14 09:54:06 → 10:06:10；约 0:12:04 FS | 2 / 1 | NQ top20 缓存；记录为全命中；设计为只提速 | 无异常、续跑或合并证据 |
| NQ-NR-03 | `3b4d780` + compatibility diff | 2024 / 0 | 2026-07-14 10:36:07 → metric 11:15:38；约 0:39:31 FS | 2 / 1 | 公共缓存 + 动态 miss；设计为只提速 | 日志进程约 11:15:44 才 shutdown；无异常、续跑或合并证据 |
| NQ-NR-05 | `3b4d780` + compatibility diff | 2024 / 0 | 2026-07-14 09:54:23 → 10:18:31；0:24:08 marker | 3 / 1 | NQ top20 缓存 + expansion miss；设计为只提速 | `full` 内 prepare→generate；无异常、无合并 |
| NQ-NR-06 | `3b4d780` + compatibility diff | 2024 / 0 | 2026-07-14 10:36:24 → 12:55:39；约 2:19:15 FS | 3 / 1 | 加载公共 query cache；只在低置信动态 query 检索，无原问题初始检索 | 无异常、续跑或合并证据 |

### 缓存结果语义的证据边界

通用 cache manager 可能包在 reranker 外层，不能笼统声称所有缓存都在 rerank 前写入。Hotpot 公共缓存的特殊证据是其来源运行使用自定义 `rs_mhr_raw_batch_search`，绕过自动 rerank 后手工写入 raw dense 文档/分数。与无缓存纯 dense top5 对照时，7,405 条中 top1 全部相同、top5 集合 7,387 条相同、顺序 7,386 条相同，平均重合 4.99757/5；score 也呈 dense cosine 范围。这强烈支持“该缓存是 raw dense”而非 reranker 污染，但 18/19 条仍有集合/顺序差异，且没有 cache/no-cache 成对 FINAL prediction。因此缓存的**设计目标**是只提速，最终逐样本 prediction/metric 是否完全等价仍为 `UNKNOWN`。缓存也没有 index/encoder/query-normalization hash manifest。

## 1.3 可恢复的实际命令

只有下列命令由相应 FINAL 目录中的 `run_command.txt` 直接证明；其余命令均为 `UNKNOWN`。为避免重复，本节保留共同参数后的单行等价写法，原始换行版本以表中 C1–C7 链接为准。

```bash
# C1: HotpotQA FLARE
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_exp.py --method_name flare --dataset_name hotpotqa --split dev --gpu_id 0 --save_note hotpotqa-flare-no-rerank-top5-full-v2 --retrieval_topk 5 --no_reranker --flare_threshold 0.2 --flare_look_ahead_steps 64 --flare_max_generation_length 256 --flare_max_iter 5 --use_retrieval_cache --retrieval_cache_path racp/output/cache/hotpotqa_dev_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache --gpu_memory_utilization 0.75

# C2: 2Wiki FLARE
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_exp.py --method_name flare --dataset_name 2wikimultihopqa --split dev --gpu_id 1 --save_note 2wiki-flare-no-rerank-top5-full --retrieval_topk 5 --no_reranker --flare_threshold 0.2 --flare_look_ahead_steps 64 --flare_max_generation_length 256 --flare_max_iter 5 --use_retrieval_cache --retrieval_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache --gpu_memory_utilization 0.75

# C3: MuSiQue FLARE
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_exp.py --method_name flare --dataset_name musique --split dev --gpu_id 1 --save_note musique-flare-no-rerank-top5-full --retrieval_topk 5 --no_reranker --flare_threshold 0.2 --flare_look_ahead_steps 64 --flare_max_generation_length 256 --flare_max_iter 5 --use_retrieval_cache --retrieval_cache_path racp/output/cache/musique_dev_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache --gpu_memory_utilization 0.75

# C4: NQ Standard RAG
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_exp.py --method_name naive --dataset_name nq --split test --gpu_id 2 --save_note nq-standard-rag-no-rerank-top10-full --retrieval_topk 10 --no_reranker --use_retrieval_cache --retrieval_cache_path racp/output/cache/nq_test_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache --gpu_memory_utilization 0.75

# C5: NQ IterRetGen
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_exp.py --method_name iterretgen --dataset_name nq --split test --gpu_id 2 --save_note nq-iterretgen-no-rerank-full --retrieval_topk 5 --no_reranker --use_retrieval_cache --retrieval_cache_path racp/output/cache/nq_test_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache --gpu_memory_utilization 0.75

# C6: NQ EFC-RAG
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_racp.py --method efc --stage full --dataset_name nq --split test --gpu_id 3 --save_note nq-efc-no-rerank-top10-full --no_reranker --retrieval_batch_size 1024 --initial_topk 20 --probe_topk 5 --final_topk 10 --enable_generation_guided --enable_static_qd_fallback --missing_query_num 1 --missing_query_mode llm --qd_num 2 --qd_topk 5 --gen_topk 10 --rrf_k 60 --rrf_weight 1.0 --role_weight 0.30 --title_weight 0.02 --source_weight 0.05 --redundancy_weight 0.01 --title_dedup_soft --max_same_title 2 --original_seed_count 4 --static_bridge_evidence_topk 5 --static_bridge_original_count 4 --static_bridge_qd_count 2 --probe_max_tokens 128 --planner_model Llama-3.1-8B-Instruct --planner_max_tokens 96 --planner_batch_size 32 --planner_inference_batch_size 8 --planner_gpu_memory_utilization 0.75 --generate_gpu_memory_utilization 0.85 --use_retrieval_cache --retrieval_cache_path racp/output/cache/nq_test_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache

# C7: NQ FLARE
/home/guanjunjie/.conda/envs/flashrag/bin/python racp/run_exp.py --method_name flare --dataset_name nq --split test --gpu_id 3 --save_note nq-flare-no-rerank-top5-full --retrieval_topk 5 --no_reranker --flare_threshold 0.2 --flare_look_ahead_steps 64 --flare_max_generation_length 256 --flare_max_iter 5 --use_retrieval_cache --retrieval_cache_path racp/output/cache/nq_test_bge_large_no_rerank_top20_retrieval_cache.json --save_retrieval_cache --gpu_memory_utilization 0.75
```

# 二、各方法实际预算

## 2.1 逐样本实际调用与文档预算

计数来自完整 `intermediate_data.json`，不是只读配置值。retriever 调用是“逻辑 query 数”；cache hit 可能跳过 BGE 编码和 FAISS 搜索。顺序检索轮/阶段与 query 数分开：Full-QD/static-QD 同一阶段可 batch 两条 QD query，因此一轮可含两个调用。候选格式为“所有 query 原始返回数 / 跨 query 去重池”；EFC 没保存所有 raw 列表，只能恢复去重候选池。

| 数据集 | 方法 | Avg LLM 调用 | Avg retriever query | Avg planner | 平均候选文档 | 最终输入文档平均值与分布 | max / avg 顺序检索阶段 |
|---|---|---:|---:|---:|---|---|---|
| HP | No-RAG | 1.0000 | 0 | 0 | 0 | 0；`0:7405` | 0 / 0 |
| HP | Standard RAG | 1.0000 | 1.0000 | 0 | 10 / 10 | 10；`10:7405` | 1 / 1.0000 |
| HP | IterRetGen | 3.0000 | 3.0000 | 0 | 15 / 7.7203 | 5；`5:7405` | 3 / 3.0000 |
| HP | Full-QD | 2.0000 | 2.9982 | 1.0000 | 29.9912 / 24.4764 | 10；`10:7405` | 2 / 2.0000 |
| HP | IRCoT | 2.8929 | 2.0000 | 0 | 10 / 7.9887 | 7.9887；`5:166, 6:766, 7:1541, 8:2150, 9:2077, 10:705` | 2 / 2.0000 |
| HP | EFC-RAG | 2.8718 | 1.9332 | 0.8718 | raw `UNKNOWN` / pool 25.5246 | 10；`10:7405` | 2 / 1.8710 |
| HP | FLARE | 5.1899 | 0.1899 | 0 | raw 0.9494 / 非累计 | 评测保存的末次检索均值 0.9136；`0:6052, 5:1353` | 5 / 0.1899 |
| 2W | No-RAG | 1.0000 | 0 | 0 | 0 | 0；`0:12576` | 0 / 0 |
| 2W | Standard RAG | 1.0000 | 1.0000 | 0 | 10 / 10 | 10；`10:12576` | 1 / 1.0000 |
| 2W | IterRetGen | 3.0000 | 3.0000 | 0 | 15 / 8.3182 | 5；`5:12576` | 3 / 3.0000 |
| 2W | Full-QD | 2.0000 | 2.9997 | 1.0000 | 29.9984 / 24.3981 | 10；`10:12576` | 2 / 2.0000 |
| 2W | IRCoT | 2.9688 | 2.0000 | 0 | 10 / 8.1566 | 8.1566；`5:159, 6:975, 7:2348, 8:3773, 9:3898, 10:1423` | 2 / 2.0000 |
| 2W | EFC-RAG | 2.9051 | 2.0161 | 0.9051 | raw `UNKNOWN` / pool 26.9002 | 10；`10:12576` | 2 / 1.9040 |
| 2W | FLARE | 5.1280 | 0.1280 | 0 | raw 0.6401 / 非累计 | 评测保存的末次检索均值 0.6059；`0:11052, 5:1524` | 5 / 0.1280 |
| MU | No-RAG | 1.0000 | 0 | 0 | 0 | 0；`0:2417` | 0 / 0 |
| MU | Standard RAG | 1.0000 | 1.0000 | 0 | 10 / 10 | 10；`10:2417` | 1 / 1.0000 |
| MU | IterRetGen | 3.0000 | 3.0000 | 0 | 15 / 9.2354 | 5；`5:2417` | 3 / 3.0000 |
| MU | Full-QD | 2.0000 | 2.9847 | 1.0000 | 29.9235 / 25.9673 | 10；`10:2417` | 2 / 1.9975 |
| MU | IRCoT | 2.9628 | 2.0000 | 0 | 10 / 8.5937 | 8.5937；`5:16, 6:88, 7:311, 8:600, 9:834, 10:568` | 2 / 2.0000 |
| MU | EFC-RAG | 2.9876 | 2.0426 | 0.9876 | raw `UNKNOWN` / pool 27.6719 | 10；`10:2417` | 2 / 1.9855 |
| MU | FLARE | 5.1407 | 0.1407 | 0 | raw 0.7034 / 非累计 | 评测保存的末次检索均值 0.6765；`0:2090, 5:327` | 4 / 0.1407 |
| NQ | No-RAG | 1.0000 | 0 | 0 | 0 | 0；`0:3610` | 0 / 0 |
| NQ | Standard RAG | 1.0000 | 1.0000 | 0 | 10 / 10 | 10；`10:3610` | 1 / 1.0000 |
| NQ | IterRetGen | 3.0000 | 3.0000 | 0 | 15 / 7.8620 | 5；`5:3610` | 3 / 3.0000 |
| NQ | Full-QD | N/A | N/A | N/A | N/A | N/A | N/A |
| NQ | IRCoT | N/A | N/A | N/A | N/A | N/A | N/A |
| NQ | EFC-RAG | 2.1008 | 1.1042 | 0.1008 | raw `UNKNOWN` / pool 20.5006 | 10；`10:3610` | 2 / 1.1008 |
| NQ | FLARE | 5.1582 | 0.1587 | 0 | raw 0.7936 / 非累计 | 评测保存的末次检索均值 0.7078；`0:3099, 5:511` | 5 / 0.1587 |

补充核查：

- IterRetGen 的确执行三轮、每轮 top5；最终字段只保存第三轮 5 篇，配置中的 `retrieval_recall_topk=10` 没有把前三轮累计为 10。
- Full-QD 每条固定有一次 planner 调用；接受的子查询平均数为 HP 1.9982、2W 1.9997、MU 1.9847。MU 分布为 2 条 2,386 例、1 条 25 例、0 条 6 例。
- EFC 的 planner 列已经计入 generation-guided missing-hop planner 和 static-QD planner；probe 属于 LLM 调用但不是 planner。direct 分支依然是 probe + final 两次 LLM 调用，而不是 Standard RAG 的一次。
- Full-QD/EFC 的单样本最大逻辑 query 数仍可为 3（original + 两条 QD），但两条 QD 同批发生在第二顺序阶段，所以最大检索阶段是 2。
- IRCoT 每条均有 2 次 thought；只有未在 thought 中完成的样本追加 finalization：HP 6,612/7,405，2W 12,183/12,576，MU 2,327/2,417。因此 Avg LLM 小于 3。
- IRCoT 最终文档经 doc-id 去重后经常少于 10；只有 HP 9.52%、2W 11.32%、MU 23.50% 的样本正好 10 篇。
- FLARE 没有一个供整条答案共享的单一“最终输入文档集合”：每轮 speculative generation 使用空 reference，触发后才用当轮 top5 重生成。表中 0/5 是保存给评测器的**最后一次触发检索结果**；未触发时 prompt 的 `{reference}` 确实为空，但仍保留“based on the given document”系统指令。

## 2.2 实际时间、token 与额外模型

秒/问题按端到端 wall time 计算；MU 两阶段另在第八节给 active time。`UNKNOWN` 表示本次不能由已保存的实际调用轨迹完整恢复，不使用配置上限冒充实际值。token 使用正式 Llama tokenizer，不含检索文档的向量 token；“partial”明确说明只覆盖哪些已保存调用。“额外模型”是相对共同设置而言：除 No-RAG 外，所有方法都使用共同的 `bge-large-en-v1.5` 检索 checkpoint；表中“无”表示没有超出共同 BGE + Llama 设置的额外 checkpoint。

| 数据集 | 方法 | Avg 输入 token | Avg 输出 token | 总耗时 / 秒·题 | 峰值 GPU 显存 | 额外模型或 checkpoint |
|---|---|---|---|---|---|---|
| HP | No-RAG | 80.3238 exact | 6.3152 exact | 0:01:13 / 0.010 | `UNKNOWN` | 无 |
| HP | Standard RAG | 1634.5889 exact | 5.3002 exact | 0:22:59 / 0.186 | `UNKNOWN` | 无 |
| HP | IterRetGen | 2572.1565 exact | 18.0342 exact | 2:19:56 / 1.134 | `UNKNOWN` | 无；同一 generator 三轮 |
| HP | Full-QD | 1634.5503 partial：仅 final 1/2 | 27.1851 exact：planner+final | 1:35:23 / 0.773 | `UNKNOWN` | planner 为同一 Llama-3.1-8B checkpoint |
| HP | IRCoT | 4489.4023 exact | 40.3292 exact | 1:19:32 / 0.644 | `UNKNOWN` | 无额外 checkpoint；有额外 demonstration |
| HP | EFC-RAG | 1635.3476 partial：仅 final，覆盖 34.82% 调用 | 63.8153 partial：probe+final，覆盖 69.64% 调用 | 0:59:02 / 0.478 | `UNKNOWN` | probe/planner/final 均为同一 Llama checkpoint |
| HP | FLARE | `UNKNOWN` | `UNKNOWN` | 4:51:01 / 2.358 | `UNKNOWN` | 无额外 checkpoint；需要 logprob |
| 2W | No-RAG | 76.8658 exact | 9.6367 exact | 0:01:38 / 0.008 | `UNKNOWN` | 无 |
| 2W | Standard RAG | 1710.6853 exact | 7.4632 exact | 0:40:54 / 0.195 | `UNKNOWN` | 无 |
| 2W | IterRetGen | 2672.5339 exact | 29.8505 exact | 1:55:28 / 0.551 | `UNKNOWN` | 无；同一 generator 三轮 |
| 2W | Full-QD | 1706.6923 partial：仅 final 1/2 | 30.7245 exact：planner+final | 2:33:13 / 0.731 | `UNKNOWN` | planner 为同一 Llama checkpoint |
| 2W | IRCoT | 4828.1145 exact | 41.7779 exact | 2:07:06 / 0.606 | `UNKNOWN` | 无额外 checkpoint；有 demonstration |
| 2W | EFC-RAG | 1705.3839 partial：仅 final，覆盖 34.42% 调用 | 77.2475 partial：probe+final，覆盖 68.85% 调用 | 2:30:47 / 0.719 | `UNKNOWN` | 同一 Llama checkpoint |
| 2W | FLARE | `UNKNOWN` | `UNKNOWN` | 6:14:53 / 1.789 | `UNKNOWN` | 无额外 checkpoint；需要 logprob |
| MU | No-RAG | 81.4340 exact | 10.2714 exact | 0:01:07 / 0.028 | `UNKNOWN` | 无 |
| MU | Standard RAG | 1614.0943 exact | 7.5685 exact | 0:09:25 / 0.234 | `UNKNOWN` | 无 |
| MU | IterRetGen | 2537.5263 exact | 26.0670 exact | 0:34:26 / 0.855 | `UNKNOWN` | 无；同一 generator 三轮 |
| MU | Full-QD | 1613.9015 partial：仅 final 1/2 | 29.0637 exact：planner+final | active 0:35:21 / 0.877；wall 0:38:52 / 0.965 | `UNKNOWN` | planner 为同一 Llama checkpoint |
| MU | IRCoT | 4738.7774 exact | 40.5184 exact | 0:31:04 / 0.771 | `UNKNOWN` | 无额外 checkpoint；有 demonstration |
| MU | EFC-RAG | 1614.0339 partial：仅 final，覆盖 33.47% 调用 | 83.2544 partial：probe+final，覆盖 66.94% 调用 | active 0:40:26 / 1.004；wall 1:27:39 / 2.176 | `UNKNOWN` | 同一 Llama checkpoint |
| MU | FLARE | `UNKNOWN` | `UNKNOWN` | 1:25:02 / 2.111 | `UNKNOWN` | 无额外 checkpoint；需要 logprob |
| NQ | No-RAG | 70.2950 exact | 9.4291 exact | 0:00:47 / 0.013 | `UNKNOWN` | 无 |
| NQ | Standard RAG | 1574.7612 exact | 8.3507 exact | 0:12:04 / 0.201 | `UNKNOWN` | 无 |
| NQ | IterRetGen | 2473.4127 exact | 23.5684 exact | 0:39:31 / 0.657 | `UNKNOWN` | 无；同一 generator 三轮 |
| NQ | Full-QD | N/A | N/A | N/A | N/A | 预定不测试 |
| NQ | IRCoT | N/A | N/A | N/A | N/A | 预定不测试 |
| NQ | EFC-RAG | 1580.3798 partial：仅 final，覆盖 47.60% 调用 | 61.9102 partial：probe+final，覆盖 95.20% 调用 | 0:24:08 / 0.401 | `UNKNOWN` | 同一 Llama checkpoint |
| NQ | FLARE | `UNKNOWN` | `UNKNOWN` | 2:19:15 / 2.314 | `UNKNOWN` | 无额外 checkpoint；需要 logprob |

token 表中的 `exact` 指“对所有持久化可见实际文本重新 tokenize 后调用覆盖完整”，不是 vLLM 原始 token-id 计数。统计使用本地正式 `Llama-3.1-8B-Instruct` tokenizer：输入 `add_special_tokens=true`，输出 `false`；输出不包含未返回的 stop token。No-RAG、Standard、IterRetGen 和 IRCoT 的实际 prompts/outputs 足以离线完整统计；Full-QD 保存 final prompt 和 raw planner output，但 planner input 未持久化；EFC 保存 final prompt、probe output 和最终答案，但 planner raw output 未保留，无法得到严格完整总量；FLARE 没保存全部 speculative prompt、被丢弃 output 与 token-score 轨迹，严格 token 成本必须做带 instrumentation 的补测。

# 三、Retrieval Recall 口径审计

## 3.1 正式评测代码

实现位于 [`flashrag/evaluator/metrics.py`](../flashrag/evaluator/metrics.py) 的 `Retrieval_Recall`（约第 219–248 行），答案规范化位于 [`flashrag/evaluator/utils.py`](../flashrag/evaluator/utils.py)（约第 5 行）。实际算法是：

1. 读取配置的 `retrieval_recall_topk`。
2. 对每条样本取 `retrieval_result[:K]`。
3. 将每篇文档的 `contents` 与每个 gold answer 都做 `normalize_answer`；只要有一篇包含任一规范化 gold 字符串，该样本得 1，否则得 0。
4. 对样本求平均。

规范化只做 lowercase、删除 ASCII 标点、删除 `a/an/the`、合并空白。它不读取 supporting title、supporting sentence、Wikipedia redirect 或别名。文档少于 K 时仅 warning，仍评全部现有文档；空列表合法计 0；evaluator 本身不去重。

| 方法 | 配置标注 K | 实际评测 K | 评测文档集合 | 是否可与 EFC Recall@10 直接比较 |
|---|---:|---:|---|---|
| No-RAG | N/A | 0 | 不计算 retrieval recall | 否 |
| Standard RAG | 10 | 10 | 原问题单次 dense top10 | 指标与 final K 相同；检索总预算不同 |
| IterRetGen | 10 | **5** | 仅第三轮 `retrieval_result_iter_2`；不是三轮累计 | **否；现有 @10 标签错误** |
| Full-QD | 10 | 10 | original + subquery 合并后、title-diverse selector 的 final10 | 最终上下文口径可比；总检索预算不同 |
| IRCoT | 10 | 5–10 | 原问题 top5 + 第一次 thought top5，按 doc-id 累计去重、按最大 dense score排序，cap 10 | 仅能称 cap@10；实际 K 不固定 |
| EFC-RAG | 10 | 10 | `doc_uid` 合并及 selector 后的 final10；不是 original top20 或整个候选池 | 参照方法 |
| FLARE | 5 | 0 或 5 | 最后一次触发检索的 top5；不是历轮累计；未触发为空 | 否，K=5 且多数为空 |

## 3.2 Supporting-title 的另一套辅助逻辑

EFC 的辅助 `support_title_recall` 位于 [`racp/run_racp.py`](run_racp.py) 约第 2097 行，与正式 `Retrieval Recall` 无关：

- 只读取 `metadata.supporting_facts.title`；gold 和 selected title 只 `.lower()`，doc title 额外 strip 空白和外围引号。
- 不做 Unicode 规范化、redirect、别名、大小写之外的标点或括号消歧处理。
- `both_support_title_hit` 实际含义是“所有 supporting titles 都命中”，不保证恰好两个。2Wiki 有 2,751 条样本含 4 个支持标题。
- HotpotQA 和 2Wiki 字段可用；MuSiQue 的证据在 `question_decomposition[*].support_paragraph.title`，当前实现未适配，2,417 条均为 `None`；NQ 没有 supporting-title 字段。

因此，2Wiki、HotpotQA、MuSiQue 并未使用同一个 supporting-evidence 逻辑；论文不得把当前 `retrieval_recall_top10` 写成“支持证据 Recall@10”。

## 3.3 建议的统一重算方案（本次未执行）

无需模型调用，可以从现有逐样本结果离线并列重算：

1. `FinalContextAnswerHit@5`：所有方法统一取最终实际输入列表前 5，空列表计 0。
2. `FinalContextAnswerHit@10`：取最多前 10，同时报告实际文档数分布；IterRetGen 不补伪文档。
3. `CumulativeAnswerHit`：仅作为独立指标，合并 IterRetGen 三轮或 FLARE 所有已保存轮次后按稳定 doc-id 去重。
4. `SupportTitleRecall@5/@10`：HP/2W 读 `supporting_facts.title`，MU 读 decomposition support；统一使用 `casefold + Unicode NFKC + strip quotes + collapse whitespace`。redirect/alias 映射应另列，不应静默混入。
5. 同时报告检索总 query 数和 final-context K，避免把相同 K 误当作 matched retrieval budget。

# 四、EFC 参数来源

## 4.1 Git 演化与 FINAL 实际值

| 参数 | 首次出现 | 后续修改 | 四个当前 FINAL 实际值 |
|---|---|---|---:|
| `initial_topk` | `ce132a8`：20 | `d3f7062` 集中到默认配置；数值未变 | 20 |
| `probe_topk` | `ce132a8`：5 | 未见数值变化 | 5 |
| `final_topk` | `ce132a8`：5 | `0044d6b` 改默认 6；正式 top10 来自运行时 override（生效 config 可证；较早 3 项实际 CLI 未保存），当前默认仍为 6 | 10 |
| `gen_topk` | `ce132a8`：10 | 未见数值变化 | 10 |
| `qd_num` | `ce132a8`：2 | 未见数值变化 | 2 |
| `qd_topk` | `ce132a8`：5 | 未见数值变化 | 5 |
| `rrf_k` | `ce132a8`：60 | 未见数值变化 | 60 |
| `role_weight` | `ce132a8`：0.30 | 未见数值变化 | 0.30 |
| `source_weight` | `ce132a8`：0.05 | 未见数值变化 | 0.05 |
| `title_weight` | `ce132a8`：0.05 | `0044d6b` 改为 0.02 | 0.02 |
| `redundancy_weight` | `ce132a8`：0.05 | `0044d6b` 改为 0.01 | 0.01 |
| `original_seed_count` | `0044d6b`：3 | `58dcbe6` 改为 4 | 4 |
| static `evidence_topk` | `58dcbe6`：5 | 未见变化 | 5 |
| static `original_count` | `58dcbe6`：4 | 未见变化 | 4 |
| static `qd_count` | `58dcbe6`：2 | 未见变化 | 2 |
| `missing_query_mode` | `ce132a8`：heuristic | `0044d6b` 改为 llm | llm |
| `missing_query_num` | `58dcbe6` 集中记录为 1 | 有 Hotpot smoke 测过 2 | 1 |

主要节点：`ce132a8`（首次 EFC，2026-06-06）、`d3f7062`（集中默认参数）、`0044d6b`（Hotpot 路由与证据选择调整）、`58dcbe6`（当前 static bridge、seed reservation=4、当前 router 主干）、`4f4573b`（正式代码基线锁定）。

还有一组容易漏写的硬编码 source-RRF 权重，位于 [`racp/efc.py`](efc.py) 顶部，自 `ce132a8` 起为 original 1.00、QD 1.05、generation-guided 1.15。它们先改变来源 RRF 分数；`source_weight=0.05` 则是 selector 的新增来源覆盖 bonus，二者不是同一参数。

## 4.2 Static-QD 配额和 router 的实际规则

Static-QD 的 top10 不是严格固定 4/2/4：planner evidence 先取 original top5，selector 最多先保留 4 个 original、再最多 2 个 QD，余下最多 4 个位置由全候选池 RRF 排名补齐。`original_seed_count=4` 则属于 role-aware pack 的另一个种子保留机制。direct route 不走该加权 selector，而从 original results 做 title-diverse top10。

当前 auto router 按顺序执行：

1. `force_route != auto` 时强制指定 route。
2. comparison 两侧实体都出现在 top5 文本时 direct。
3. probe 自相矛盾、同时 top5 已含期望答案类型时 direct。
4. 非 multi-hop 时 direct。
5. probe 暴露新实体时 generation-guided。
6. probe bad 或 uncertain 时 static-QD。
7. 仅在启用 centroid router 时，`q_centroid_sim < 0.35` 走 static-QD。
8. 其余 direct。

四个 FINAL 均 `use_centroid_router=false`，所以 0.35 阈值实际不生效。HP、2W、MU 被传入 `assume_multihop=true`，NQ 为 false；数值参数相同，但 NQ 的有效路由行为并非完全同构。

## 4.3 参数选择证据与数据集一致性

HotpotQA dev 上可找到 final5、final6、final10 全量版本，以及 router v2–v6、heuristic/LLM missing query、seed 3/4、missing query 1/2、static bridge 等开发痕迹。然而多轮全量实验同时改变 router、prompt、K、权重和 missing-query 策略，不能从现有文件隔离每一项贡献：

| HotpotQA 历史全量版本 | 主要共同变化 | EM / F1 | 可作为单变量搜索吗 |
|---|---|---:|---|
| `2026_06_06_17_26_efc-hotpotqa-full` | final5、heuristic、title/redundancy=.05/.05、无 seed reservation | 26.12 / 34.12 | 否 |
| `2026_06_06_19_55_efc-hotpotqa-full-v2` | final6、LLM query、.02/.01、seed3、router/prompt 同变 | 33.65 / 44.50 | 否 |
| `2026_06_07_12_41_efc-iterretgen-hotpotqa-full` | seed4 及逻辑变化 | 35.52 / 46.64 | 否 |
| `2026_06_08_13_17_efc-static-bridge-v1-full` | static bridge | 35.57 / 46.69 | 否 |
| `2026_07_09_15_52_efc-static-bridge-final-top10-full` | final10 | 35.85 / 47.21 | 当前 FINAL |

结论：

- HotpotQA dev 明确参与算法开发与参数调整。
- `initial_topk/probe_topk/qd_num/qd_topk/gen_topk/rrf_k/role_weight/source_weight` 没有系统搜索记录；其他参数也没有完整单变量 grid。
- `HOTPOTQA_EXPERIMENT_REPORT.md` 是当前 untracked 的事后汇总，不是可证明为同期保存的 tuning log。
- 四个当前 FINAL 的核心数值参数完全相同；主要执行差异是 HP `retrieval_cache_only=true`，以及 NQ 的 `assume_multihop=false`。
- 存在同名 EFC 但参数不同的旧全量/SMOKE 输出；论文只能引用四个 top10 FINAL。
- **每个参数为何选择该值、是否由 HotpotQA dev 全量结果直接决定，需要在论文中补充说明，不能从现有文件逐项验证。**

# 五、Prompt 与答案解析

## 5.1 最终版本位置和行为

Standard RAG、IterRetGen 每轮、Full-QD final、EFC final，以及 FLARE 的基础模板来自 [`flashrag/prompt/base_prompt.py`](../flashrag/prompt/base_prompt.py) 约第 5 行：系统要求基于给定文档、只输出答案；用户字段为 `Question: {question}`。实际字符串中 `document.Only` 之间缺空格。文档格式是 `Doc N(Title: <首行>) <正文>`。

| 方法 / 阶段 | Prompt 路径与关键内容 | 实际生成上限与 stop | 答案/parser |
|---|---|---|---|
| No-RAG final | [`racp/run_exp.py`](run_exp.py) 约 339：基于 own knowledge，只输出答案 | 32；EOS/eot | 无方法级清洗 |
| Standard RAG final | [`base_prompt.py`](../flashrag/prompt/base_prompt.py)：公共短答案模板 | 32；EOS/eot | 无 |
| IterRetGen | [`active_pipeline.py`](../flashrag/pipeline/active_pipeline.py) 约 13：三轮均用公共短答案模板；下一轮 query=`question + 上轮答案` | 每轮 32；EOS/eot | 无；只保留第三轮原始答案 |
| Full-QD planner | [`racp/run_racp.py`](run_racp.py) 约 779：严格 JSON array，恰好 2 条独立 Wikipedia-style query，不得回答问题 | 96；planner stop | JSON + 容错 parser，约 1019–1057 |
| Full-QD final | 公共短答案模板 | 32；EOS/eot | 无 |
| IRCoT thought | [`active_pipeline.py`](../flashrag/pipeline/active_pipeline.py) 约 934：多跳 reasoning 指令 + 1 条完整 demonstration | 每 thought 32；`.` 或换行 stop；最多 2 轮 | thought 可含 reasoning |
| IRCoT finalization | 同文件约 1055：要求 `So the answer is:` 后最短答案 | 额外 32；双空格或换行 stop | [`pred_parse.py`](../flashrag/utils/pred_parse.py) 约 21：取最后一个 prefix 后内容并截断 |
| EFC probe | [`racp/run_racp.py`](run_racp.py) 约 180：最多两句 evidence、保留 bridge entity、不足时 unknown、固定答案前缀 | 128；无方法级 stop | 正则仅供 router，不是 final parser |
| EFC missing-hop planner | 同文件约 1993：question + tentative reasoning，输出一个 JSON query | 96；planner stop | JSON/容错；失败可 heuristic repair |
| EFC static-QD | 同文件约 1385；evidence-conditioned 版本约 1422：输出 2 条 QD query | 96；planner stop | JSON/容错 parser |
| EFC final | 与 Standard 公共 final 模板逐字等价；probe reasoning 不进入 final prompt | 32；EOS/eot | 无 |
| FLARE | [`active_pipeline.py`](../flashrag/pipeline/active_pipeline.py) 约 691：没有独立 FLARE continuation prompt，复用公共 answer-only 模板 | 每次 look-ahead **64**，最多 5 轮、总上限配置 256；特殊字符/换行 stop | 无最终清洗；拼接各轮首句 |

vLLM 还会在 [`flashrag/generator/generator.py`](../flashrag/generator/generator.py) 约第 237 行自动加入 `<|eot_id|>` stop。

表中 `planner stop` 的实际列表为：`<|eot_id|>`、`\nQuestion:`、`\n\nQuestion:`、`Expected output:`、`Solution:`、`Answer:`。Prompt 证据层级也应区分：持久化的 final prompt、IRCoT thought/new-thought 和部分 planner raw output 是 artifact-proven；未持久化的 Full-QD/EFC planner input 由相应运行版本模板与保存字段重构。HP IRCoT 的完整运行时 compatibility diff 未保存，但其 7,405 条已保存 prediction 用旧/新 parser 复算一致，因此答案结果不受该 parser 版本歧义影响。

## 5.2 公平性差异

- Standard、IterRetGen、Full-QD final、EFC final 都要求短答案；No-RAG 也明确 answer-only。IRCoT 则先允许并鼓励显式 reasoning，再用专用 parser 提取答案。
- IRCoT 独有一条完整多跳 demonstration、5 篇示例文档、raw non-chat prompt 和更强的答案截断，这可能带来额外优势，尤其影响 comparison/yes-no 与输出格式。
- 普通 final answer 的单次上限多为 32，但总生成预算不一致：Standard/No-RAG 32；IterRetGen 3×32；Full-QD planner96+final32；IRCoT 最多 2×32+finalization32；EFC probe128+可选planner96+final32；FLARE 每个 speculative/trigger generation 64。
- Full-QD 和 EFC planner 都调用同一个 Llama-3.1-8B-Instruct checkpoint，不是另一个训练 checkpoint，但必须计入 LLM 成本。
- FLARE 未复现原论文风格的专用 continuation prompt；无检索时又给公共 RAG 模板传空 reference。该实现差异必须随 FLARE 结果披露。
- 所有方法的最终 EM/F1 仍经过同一评测规范化，但 IRCoT 在此之前已做专用 reasoning extraction；因此不能把它描述为完全相同的 answer parser。

# 六、EFC 诊断信息

本节从四个 EFC FINAL 的逐样本轨迹、retrieval cache、原始 top20、扩展 query 结果和 final10 离线重构；没有重新调用模型或 retriever。候选池大小与保存的 `candidate_pool_count` 逐样本完全一致。百分数均以全 split 为分母，除非明确标为 conditional。

## 6.1 路由、route-wise 质量与 planner 健康度

| 数据集 | direct | generation-guided | static-QD | route-wise EM / F1（D / G / S） | planner 失败 | heuristic repair | 平均候选池（min–max） |
|---|---:|---:|---:|---|---:|---:|---:|
| HP | 955（12.90%） | 5,990（80.89%） | 460（6.21%） | 49.01/60.61 · 34.72/46.36 · 23.26/30.46 | 136（1.84%） | 130（1.76%） | 25.5246（20–30） |
| 2W | 1,207（9.60%） | 9,959（79.19%） | 1,410（11.21%） | 34.88/41.89 · 19.22/28.73 · 15.89/23.21 | 85（0.68%） | 72（0.57%） | 26.9002（20–30） |
| MU | 35（1.45%） | 2,244（92.84%） | 138（5.71%） | 2.86/10.70 · 10.43/18.43 · 3.62/6.73 | 29（1.20%） | 24（0.99%） | 27.6719（20–30） |
| NQ | 3,246（89.92%） | 352（9.75%） | 12（0.33%） | 34.81/46.93 · 40.34/53.01 · 25.00/22.95 | 25（0.69%） | 25（0.69%） | 20.5006（20–30） |

“planner 失败”来自轨迹中的解析/有效性失败标记；repair 是随后实际采用 heuristic repair 的子集，不应把二者混为一个字段。route-wise 分数是同一 FINAL 内按 route 切片的逐样本 EM/F1，不是独立实验。

## 6.2 支持标题在 original、expanded pool 与 final10 的转化

这里统一从数据集原始 metadata 离线取标题，沿用当前 EFC 的 lower/exact title 语义，以便忠实解释既有运行；不是第三节建议的 robust-normalized 新指标。NQ 没有 supporting-title 标注，所以写 `N/A`。

| 数据集 | Gold title 数量 | Macro support-title recall：original20 → pool → final10 | 至少两个标题同时命中：original20 → pool → final10 | 全部支持标题命中：original20 → pool → final10 |
|---|---|---|---|---|
| HP | 全部为 2 | 66.81% → 72.55% → 68.11% | 46.60% → 56.42% → 51.24% | 46.60% → 56.42% → 51.24% |
| 2W | 9,825 条为 2；2,751 条为 4 | 42.76% → 51.30% → 48.90% | 24.26% → 40.00% → 37.44% | 17.36% → 30.82% → 28.96% |
| MU | 54/1,237/745/381 条分别有 1/2/3/4 个标题 | 31.97% → 38.70% → 32.47% | 13.36% → 23.87% → 17.79% | 8.19% → 14.52% → 11.21% |
| NQ | 无该标注 | N/A | N/A | N/A |

这张表回答了三个不同问题：原始 top20 是否已含支持文档、扩展候选池是否补到、selector 是否把它们保留进 final10。对于 gold title 超过 2 的样本，“至少两个”和“全部”必须分别报告。

## 6.3 失败漏斗

| 数据集 | original20 未含全部支持 | expansion 后仍未含全部支持 | expansion 新补全 | pool 已全但 selector 丢失 | final10 已全 | final 已全但答案错误 |
|---|---:|---:|---:|---:|---:|---:|
| HP | 3,954（53.40%） | 3,227（43.58%；原始失败中 81.61%） | 727 | 384（5.19%；pool-complete 中 9.19%） | 3,794 | 1,967（26.56%；final-complete 中 51.85%） |
| 2W | 10,393（82.64%） | 8,700（69.18%；原始失败中 83.71%） | 1,693 | 234（1.86%；pool-complete 中 6.04%） | 3,642 | 2,487（19.78%；final-complete 中 68.29%） |
| MU | 2,219（91.81%） | 2,066（85.48%；原始失败中 93.10%） | 153 | 80（3.31%；pool-complete 中 22.79%） | 271 | 206（8.52%；final-complete 中 76.01%） |
| NQ | N/A | N/A | N/A | N/A | N/A | N/A |

按用户要求的五类失败解释：

1. **语料中缺少支持文档：`UNKNOWN`。** 当前没有做 corpus 全量 title/paragraph presence 扫描。
2. **retriever 未召回：** original20 未含全部支持是“语料缺失 + retriever miss”的上界，不能在未扫 corpus 时净化成纯 retriever miss。
3. **expansion 未补到：** 表中 expansion 后仍不完整的 3,227 / 8,700 / 2,066 条可直接恢复，但其中仍混有语料本身缺失。
4. **selector 未保留：** pool 已完整而 final10 丢失的 384 / 234 / 80 条，是可明确归因于 selection 的失败。
5. **generator 有证据仍答错：** final10 已包含全部标题但 EM 错误的 1,967 / 2,487 / 206 条。它是“标题级证据存在”的生成失败代理，不证明所需支持句一定完整或 prompt 未截断。

该漏斗是 route-agnostic：direct 样本的 pool 等于 original，它们没有实际执行 expansion。因此“expansion 后仍未补全”更严格的名称应是“路由后候选池仍不完整”；若要区分“尝试扩展但失败”和“router 未触发扩展”，可按已保存 route 离线再切片，不需要模型调用。

# 七、已有消融搜索

状态口径：`FINAL` 只用于唯一正式台账已登记的正式结果；`PARTIAL` 可附注 `FULL-COMPLETE/UNREGISTERED` 表示全量产物完成但未登记，也用于历史全量/子集产物不满足锁定条件或存在混杂；`SMOKE` 不作论文成绩。

| 项目 | 状态 | 目录 / 配置 / 样本数 / 指标 |
|---|---|---|
| generation-guided-only | `FINAL` | [D](output/hotpotqa_2026_07_14_13_49_hotpotqa-efc-force-generation-guided-no-static-top10-full/)；n=7,405；EM/F1/AnswerHit@10=35.61/46.95/71.60；实际7,371条generation、34条direct fallback |
| always-direct | `FINAL` | [D](output/hotpotqa_2026_07_15_10_01_hotpotqa-efc-force-direct-top10-full/)；n=7,405；33.59/44.58/65.63；仍含probe+final两次LLM |
| always-static-QD | `SMOKE` | 5 个 `force_route=static_qd`、n=1、final5 的准备产物；均无完整预测/metric |
| w/o static-QD | `FINAL` | [D](output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-auto-no-static-top10-full-v3-online-fill/)；n=7,405；35.76/47.09/71.49；auto router保持，static映射到direct |
| w/o router | `PARTIAL`（混杂） | 上述固定 generation-guided 可作一种去 router 对照，但同时关闭 static，属于混杂而非完整 router 消融 |
| w/o role weight | `FINAL` | [D](output/hotpotqa_2026_07_15_10_16_hotpotqa-efc-no-role-weight-top10-full/)；n=7,405；35.77/47.02/71.05 |
| w/o source weight | `FINAL` | [D](output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-source-weight-top10-full/)；n=7,405；35.85/47.21/71.51；六项汇总与完整EFC相同 |
| w/o title weight | `NOT FOUND` | 未发现 `title_weight=0` |
| w/o redundancy penalty | `FINAL` | [D](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-no-redundancy-weight-top10-full/)；n=7,405；36.04/47.46/72.21；未观察到penalty正收益 |
| w/o original seed reservation | `FINAL` | [D](output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-original-seed-top10-full/)；n=7,405；35.80/47.09/71.41 |
| RRF-only | `FINAL` | [D](output/hotpotqa_2026_07_15_12_46_hotpotqa-efc-rrf-only-top10-full/)；n=7,405；33.30/43.67/67.13 |
| title-dedup-only | `FINAL` | [D](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-title-dedup-only-top10-full/)；n=7,405；35.18/46.57/70.90；隔离的RRF+title-diverse selector |
| MMR | `PARTIAL` | 找到历史 50/1,000 样本实验，使用 E5 和/或 reranker，与锁定 BGE/no-reranker 条件不一致 |
| fixed original/expanded quota | `NOT FOUND` | 当前 static 分支有配额，但没有有/无配额受控对照 |
| final top5 | `FINAL` | [D](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-final-top5-controlled-full/)；n=7,405；35.53/46.63/66.31@5；只改final/评测K |
| final top6 | `PARTIAL` | 多个 Hotpot 旧全量，EM/F1 33.65/44.50 至 35.57/46.69；router/seed/static 同时变化 |
| final top10 | `FINAL` | 四个正式 EFC 主结果，完整目标 split；目录、config、n 和指标见表 1.1 的 HP/2W/MU/NQ EFC 行 |
| matched retrieval budget | `FINAL` | [D](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-ircot-matched-budget10-full/)；n=7,405；36.66/48.17/69.82@<=10；匹配最大2次query/10篇raw docs，非完整计算成本匹配 |

generation-guided-only 的实际 route 字段是 7,371 条 generation-guided、34 条 direct fallback；运行于 2026-07-14 13:48:59–15:17:04，单卡，总计 1:28:05，日志无异常。它已登记FINAL，论文必须写“强制 generation-guided、失败时 fallback”，不能声称 100% expansion 成功。

W11四项均覆盖7,405条且无缺失预测。matched-budget只锁定最大检索调用与raw-document
上限，不匹配prompt或推理路径；final top5的answer-hit为@5；`run.log`对variable-K
support-title的`@5`显示不能当作统一K口径。这些限制不影响FINAL完整性，但必须进入论文表注。

Always-static 的 5 个单样本目录为 `rs-mhr-static-smoke`（三个时间版本）、`rs-mhr-planner3b-smoke` 和 `rs-mhr-cache-only-smoke`；无 `metric_score.txt`，不得报告成绩。Router v2–v6 等 200 样本目录也只是开发 smoke，且有些目录名与实际 `force_route:auto` 不一致。

历史 MMR 代表性结果包括：HP n=1,000、E5+reranker 的 EM/F1/Recall@5=26.3/37.87/60.5；HP n=1,000、E5 无 reranker=23.1/34.11/52.9；NQ n=1,000、E5+reranker=34.9/46.99/79.3。它们只能证明做过探索，不能进入锁定论文消融表。

### 7.1 已找到产物的明细

Always-static-QD 只找到以下准备/烟测产物：

| 目录 / 配置 | n | 指标 |
|---|---:|---|
| [D](output/hotpotqa_2026_06_01_14_13_rs-mhr-static-smoke/) · [config](output/hotpotqa_2026_06_01_14_13_rs-mhr-static-smoke/config.yaml) | 1 | 无 `metric_score.txt` |
| [D](output/hotpotqa_2026_06_01_14_14_rs-mhr-static-smoke/) · [config](output/hotpotqa_2026_06_01_14_14_rs-mhr-static-smoke/config.yaml) | 1 | 无 `metric_score.txt` |
| [D](output/hotpotqa_2026_06_01_14_15_rs-mhr-static-smoke/) · [config](output/hotpotqa_2026_06_01_14_15_rs-mhr-static-smoke/config.yaml) | 1 | 无 `metric_score.txt` |
| [D](output/hotpotqa_2026_06_01_15_31_rs-mhr-planner3b-smoke/) · [config](output/hotpotqa_2026_06_01_15_31_rs-mhr-planner3b-smoke/config.yaml) | 1 | 无 `metric_score.txt` |
| [D](output/hotpotqa_2026_06_01_15_36_rs-mhr-cache-only-smoke/) · [config](output/hotpotqa_2026_06_01_15_36_rs-mhr-cache-only-smoke/config.yaml) | 1 | 无 `metric_score.txt` |

EFC 历史 final-K 产物：

| K | 目录与配置/指标 | n | EM / F1 / 当前 answer-hit 标签 | 状态理由 |
|---:|---|---:|---:|---|
| 5 | [D](output/hotpotqa_2026_06_06_17_26_efc-hotpotqa-full/) · [config](output/hotpotqa_2026_06_06_17_26_efc-hotpotqa-full/config.yaml) · [metric](output/hotpotqa_2026_06_06_17_26_efc-hotpotqa-full/metric_score.txt) | 7,405 | 26.12 / 34.12 / 55.14@5 | `PARTIAL`：旧 router/参数 |
| 6 | [D](output/hotpotqa_2026_06_06_19_55_efc-hotpotqa-full-v2/) · [config](output/hotpotqa_2026_06_06_19_55_efc-hotpotqa-full-v2/config.yaml) · [metric](output/hotpotqa_2026_06_06_19_55_efc-hotpotqa-full-v2/metric_score.txt) | 7,405 | 33.65 / 44.50 / 63.94@6 | `PARTIAL`：多参数同变 |
| 6 | [D](output/hotpotqa_2026_06_07_12_41_efc-iterretgen-hotpotqa-full/) · [config](output/hotpotqa_2026_06_07_12_41_efc-iterretgen-hotpotqa-full/config.yaml) · [metric](output/hotpotqa_2026_06_07_12_41_efc-iterretgen-hotpotqa-full/metric_score.txt) | 7,405 | 35.52 / 46.64 / 68.43@6 | `PARTIAL`：seed/逻辑同变 |
| 6 | [D](output/hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full/) · [config](output/hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full/config.yaml) · [metric](output/hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full/metric_score.txt) | 7,405 | 35.57 / 46.69 / 68.35@6 | `PARTIAL`：引入 static bridge |
| 6 | [D](output/hotpotqa_2026_06_08_16_50_full/) · [config](output/hotpotqa_2026_06_08_16_50_full/config.yaml) · [metric](output/hotpotqa_2026_06_08_16_50_full/metric_score.txt) | 7,405 | 35.54 / 46.68 / 68.39@6 | `PARTIAL`：非锁定版本 |

历史 MMR 产物：

| 目录与配置/指标 | n | 条件 | EM / F1 / answer-hit | 状态 |
|---|---:|---|---:|---|
| [D](output/hotpotqa_2026_05_27_12_50_racp-gapk+mmr+reranker/) · [config](output/hotpotqa_2026_05_27_12_50_racp-gapk+mmr+reranker/config.yaml) · [metric](output/hotpotqa_2026_05_27_12_50_racp-gapk+mmr+reranker/metric_score.txt) | 1,000 | E5 + reranker | 26.3 / 37.87 / 60.5@5 | `PARTIAL` |
| [D](output/hotpotqa_2026_05_27_13_09_racp/) · [config](output/hotpotqa_2026_05_27_13_09_racp/config.yaml) · [metric](output/hotpotqa_2026_05_27_13_09_racp/metric_score.txt) | 1,000 | E5，无 reranker | 23.1 / 34.11 / 52.9@5 | `PARTIAL` |
| [D](output/hotpotqa_2026_05_27_14_50_racp-decomp-gapk-mmr-smoke/) · [config](output/hotpotqa_2026_05_27_14_50_racp-decomp-gapk-mmr-smoke/config.yaml) · [metric](output/hotpotqa_2026_05_27_14_50_racp-decomp-gapk-mmr-smoke/metric_score.txt) | 50 | E5 + reranker | 32.0 / 42.36 / 62.0@5 | `SMOKE` |
| [D](output/hotpotqa_2026_05_27_15_12_racp-decomp-gapk-mmr-reranker-1000/) · [config](output/hotpotqa_2026_05_27_15_12_racp-decomp-gapk-mmr-reranker-1000/config.yaml) · [metric](output/hotpotqa_2026_05_27_15_12_racp-decomp-gapk-mmr-reranker-1000/metric_score.txt) | 1,000 | E5 + reranker | 26.4 / 38.65 / 61.8@5 | `PARTIAL` |
| [D](output/hotpotqa_2026_05_27_17_42_racp-decomp-gapk-mmr-no-reranker-1000/) · [config](output/hotpotqa_2026_05_27_17_42_racp-decomp-gapk-mmr-no-reranker-1000/config.yaml) · [metric](output/hotpotqa_2026_05_27_17_42_racp-decomp-gapk-mmr-no-reranker-1000/metric_score.txt) | 1,000 | E5，无 reranker | 24.6 / 36.18 / 51.6@5 | `PARTIAL` |
| [D](output/nq_2026_05_26_20_34_mmr/) · [config](output/nq_2026_05_26_20_34_mmr/config.yaml) · [metric](output/nq_2026_05_26_20_34_mmr/metric_score.txt) | 1,000 | E5 + reranker | 34.9 / 46.99 / 79.3@5 | `PARTIAL` |
| [D](output/nq_2026_05_27_13_03_racp/) · [config](output/nq_2026_05_27_13_03_racp/config.yaml) | 配置 1,000 | E5，无 reranker | 无预测/指标 | `PARTIAL`（未完成产物） |

上表各有结果的目录还保留 `intermediate_data.json`；最后一项只有 config。Full-QD 的两个 title-dedup 历史全量分别为 [top5 D](output/hotpotqa_2026_06_02_13_41_full-qd-planner3b-parser-v2-title-dedup-top5/)（[config](output/hotpotqa_2026_06_02_13_41_full-qd-planner3b-parser-v2-title-dedup-top5/config.yaml)，[metric](output/hotpotqa_2026_06_02_13_41_full-qd-planner3b-parser-v2-title-dedup-top5/metric_score.txt)，n=7,405，EM/F1/answer-hit=36.22/47.78/65.10）与 [top6 D](output/hotpotqa_2026_06_02_18_45_full-qd-parser-v2-title-dedup-top6/)（[config](output/hotpotqa_2026_06_02_18_45_full-qd-parser-v2-title-dedup-top6/config.yaml)，[metric](output/hotpotqa_2026_06_02_18_45_full-qd-parser-v2-title-dedup-top6/metric_score.txt)，n=7,405，35.03/46.64/66.28）；二者启用 reranker 且使用 3B planner，不能视为 EFC title-dedup-only 消融。

Router 开发烟测均为 n=200 且实际 `force_route:auto`，只用于证明开发搜索存在：

| 目录 / 配置 / 指标 | n | EM / F1 / answer-hit |
|---|---:|---:|
| [D](output/hotpotqa_2026_06_06_19_14_efc-router-v2-smoke200/) · [config](output/hotpotqa_2026_06_06_19_14_efc-router-v2-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_06_19_14_efc-router-v2-smoke200/metric_score.txt) | 200 | 29.0 / 37.84 / 61.0@6 |
| [D](output/hotpotqa_2026_06_06_19_24_efc-router-v3-smoke200/) · [config](output/hotpotqa_2026_06_06_19_24_efc-router-v3-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_06_19_24_efc-router-v3-smoke200/metric_score.txt) | 200 | 29.5 / 40.50 / 62.5@6 |
| [D](output/hotpotqa_2026_06_06_19_30_efc-router-v4-probe-final-smoke200/) · [config](output/hotpotqa_2026_06_06_19_30_efc-router-v4-probe-final-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_06_19_30_efc-router-v4-probe-final-smoke200/metric_score.txt) | 200 | 29.0 / 39.00 / 62.5@6 |
| [D](output/hotpotqa_2026_06_06_19_34_efc-router-v5-baseline-final-smoke200/) · [config](output/hotpotqa_2026_06_06_19_34_efc-router-v5-baseline-final-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_06_19_34_efc-router-v5-baseline-final-smoke200/metric_score.txt) | 200 | 33.5 / 45.48 / 62.5@6 |
| [D](output/hotpotqa_2026_06_06_19_40_efc-router-v6-llm-query-smoke200/) · [config](output/hotpotqa_2026_06_06_19_40_efc-router-v6-llm-query-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_06_19_40_efc-router-v6-llm-query-smoke200/metric_score.txt) | 200 | 36.0 / 47.01 / 64.0@6 |
| [D](output/hotpotqa_2026_06_08_14_53_efc-main-v2-router-gen2-smoke200/) · [config](output/hotpotqa_2026_06_08_14_53_efc-main-v2-router-gen2-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_08_14_53_efc-main-v2-router-gen2-smoke200/metric_score.txt) | 200 | 35.0 / 46.20 / 68.0@6 |
| [D](output/hotpotqa_2026_06_08_15_06_efc-main-v2-router-only-smoke200/) · [config](output/hotpotqa_2026_06_08_15_06_efc-main-v2-router-only-smoke200/config.yaml) · [metric](output/hotpotqa_2026_06_08_15_06_efc-main-v2-router-only-smoke200/metric_score.txt) | 200 | 37.0 / 49.85 / 67.0@6 |

这些结果全部保持 `SMOKE`，目录名中的 “router-only” 也不能据此解释成正式 w/o-router。

# 八、效率信息恢复

第 2.2 节已经逐 FINAL 列出总耗时、秒/题和峰值显存状态。本节专门区分冷启动、算法循环与缓存复跑。

| 方法 | HP | 2W | MU | NQ | 时间口径与资源说明 |
|---|---:|---:|---:|---:|---|
| No-RAG | 0:01:13 | 0:01:38 | 0:01:07 | 0:00:47 | FS 端到端近似；单卡 |
| Standard RAG | 0:22:59 | 0:40:54 | wall 0:09:25；active约0:08:44 | 0:12:04 | 单卡；公共检索缓存 |
| IterRetGen | 2:19:56 | 1:55:28 | 0:34:26 | 0:39:31 | 单卡；HP 无缓存，其余首轮公共缓存+动态 query |
| Full-QD | 1:35:23 | 2:33:13 | wall 0:38:52；active 0:35:21 | N/A | 单卡；original cache + QD 动态检索 |
| IRCoT | 1:19:32 | 2:07:06 | 0:31:04 | N/A | HP TP2；2W/MU TP4；wall time 不能当单卡计算量 |
| EFC-RAG | 0:59:02 | 2:30:47 | wall 1:27:39；active 0:40:26 | 0:24:08 | 单卡；HP 为 EFC cache-only 重放；MU 含47:13人工调度空档 |
| FLARE | E2E 4:51:01；loop 4:49:24 | E2E 6:14:53；loop 6:12:41 | E2E 1:25:02；loop 1:23:27 | E2E 2:19:15；loop 2:17:38 | 单卡；无原问题初始检索，仅低置信动态 query；加载公共 cache，逐 query hit/miss 未完整记录 |

恢复结论：

- **进程端到端与真正冷启动：** 可证明每个 FINAL 是新进程，FS 时间通常包含模型初始化；但不能证明 OS page cache、模型文件 cache 或索引 cache 是冷的，真正“端到端冷启动耗时”为 `UNKNOWN`。同时，多数检索 query cache 明确是热的。显式 `RACP full started/finished` marker 是否覆盖全部启动步骤取决于入口，不能与 FS 时间无条件混用。
- **算法执行时间：** FLARE 保存了循环时间；MU Standard/Full-QD/EFC 可分别恢复两阶段 active time。其他方法没有统一 per-stage profiler，只能用 E2E 或 full marker。
- **缓存复跑时间：** HP EFC 明确是 `retrieval_cache_only=true` 的热检索缓存重放。其他缓存运行大多混合公共 query hit 与动态 miss；没有同配置冷/热成对计时，不能计算缓存加速倍数。
- **首次 cache miss：** NQ Standard 记录全命中；其他动态方法没有逐 query hit/miss 汇总，准确比例为 `UNKNOWN`。
- **模型/索引加载：** 所有 E2E/FS 运行都是新进程且包含 generator 加载；No-RAG 不初始化在线检索，故不含索引检索。Full-QD、IRCoT、IterRetGen 及 FLARE 的动态 query、2W/MU/NQ EFC 的动态 expansion 都要求索引可用，其初始化位于 E2E 内但无法从算法时间中单独扣除。仅公共 cache hit 的 Standard 是否仍实际加载完整 index，日志不足，写 `UNKNOWN`；HP EFC cache-only 没有在线 FAISS query，但是否初始化了 index 也为 `UNKNOWN`。算法 loop 通常不包含模型加载。论文应并列报告口径，不应混成单一“运行时间”。
- **GPU-hours：** 2W IRCoT 的 2:07:06 使用 4 卡，约 8.47 GPU-hours；其 wall time 不应直接与单卡 EFC 的 2:30:47解释为成本更低。
- **峰值显存：** 所有 FINAL 都是 `UNKNOWN`。日志中的单卡约 14.99 GiB model-weight load、TP4 约 3.77 GiB/卡和 `gpu_memory_utilization` 配置都不是实测峰值。
- **断点/合并：** 未发现 FINAL 拼接或 `fuse_prediction_outputs` 证据。MU 的 prepare/generate 是预定两阶段，不是异常续跑；HP IterRetGen、2W Standard/IterRetGen 因缺完整日志，历史异常/重启仍为 `UNKNOWN`。

# 九、统计显著性所需文件

每个 FINAL 的 `intermediate_data.json` 都保留根字段 `id`、`golden_answers`，以及 `output.pred` 和 `output.metric_score.em/f1`。全量流式核对结果：

| 数据集 | n | ID 一一对应且顺序一致 | Gold 顺序一致 | 重复 ID | 缺失预测字段 | 缺失逐样本 EM/F1 | 顺序校验 hash（ID / ID+gold） |
|---|---:|---|---|---:|---:|---:|---|
| HP | 7,405 | 是 | 是 | 0 | 0 | 0 | `0c71e56a40bd81c6` / `b0a80739ab10a300` |
| 2W | 12,576 | 是 | 是 | 0 | 0 | 0 | `a6290259424cdfb9` / `3e31f1e2abef8273` |
| MU | 2,417 | 是 | 是 | 0 | 0 | 0 | `be981aac0853e6ad` / `fa0e3f2fe0fe440d` |
| NQ | 3,610 | 是 | 是 | 0 | 0 | 0 | `438883fd333502e6` / `5bba33477b45729c` |

IRCoT 有合法空字符串预测：HP 8、2W 13、MU 9；它们不是缺失记录，逐样本分数仍在。由此：

- EFC vs Standard RAG：四数据集均可直接 paired bootstrap。
- EFC vs IterRetGen：四数据集均可。
- EFC vs Full-QD：HP、2W、MU 可；NQ 是预定 `N/A`。
- EFC vs IRCoT：HP、2W、MU 可；NQ 是预定 `N/A`。
- 应直接 bootstrap 已保存的逐样本 EM/F1，不要用当前工作区代码重新 parse 旧 prediction。Bootstrap 用于 Δ 的置信区间；显著性 p 值另用 paired sign-flip randomization test。多数据集、多基线同时报告 p 值时再做 Holm–Bonferroni 校正。

建议命令模板如下，本次未运行。将 `EFC` 和 `BASE` 替换成表 1.1 对应目录的 `intermediate_data.json`：

```bash
EFC=racp/output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full/intermediate_data.json
BASE=racp/output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full/intermediate_data.json

python - "$EFC" "$BASE" <<'PY'
import json, sys
import numpy as np

a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert [(x["id"], x["golden_answers"]) for x in a] == [(x["id"], x["golden_answers"]) for x in b]
rng, rounds, n = np.random.default_rng(2024), 10000, len(a)
for metric in ("em", "f1"):
    d = np.asarray([x["output"]["metric_score"][metric] - y["output"]["metric_score"][metric] for x, y in zip(a, b)])
    boot = np.asarray([d[rng.integers(0, n, n)].mean() for _ in range(rounds)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed, extreme = abs(d.mean()), 0
    for _ in range(rounds):
        extreme += abs((d * rng.choice((-1.0, 1.0), size=n)).mean()) >= observed
    p = (extreme + 1) / (rounds + 1)
    print(metric, "delta_pp", 100*d.mean(), "bootstrap_95%CI_pp", (100*lo, 100*hi), "randomization_p", p)
PY
```

# 十、最终缺口分类

## A. 已从现有文件恢复，可以直接写论文

- 26 个主结果的完整 split 覆盖、最终 EM/F1/其他指标、配置和逐样本预测。来源：[`PAPER_RESULTS_REFERENCE.md`](PAPER_RESULTS_REFERENCE.md)、表 1.1 各 FINAL 的 `config.yaml`、`metric_score.txt`、`intermediate_data.json`。
- 各 FINAL 可恢复的 commit、运行时 diff 证据边界、seed、GPU/TP、起止时间、总耗时、两阶段/独立重跑状态。来源：表 1.2 各日志、文件时间戳、reflog、台账和 7 个 `run_command.txt`。
- 各方法实际 LLM/retriever/planner 调用次数、候选池、final-context 文档数和检索轮数分布。来源：逐样本中间轨迹；结果见第 2.1 节。
- No-RAG、Standard、IterRetGen、IRCoT 的实际平均 token；Full-QD output token；EFC 可见调用的 partial token。来源：保存的实际 prompt/output + 本地正式 tokenizer，见第 2.2 节。
- 现有 `Retrieval Recall` 的真实 answer-containment 定义、K、集合与去重语义；IterRetGen final5、IRCoT 5–10、FLARE 0/5。来源：evaluator、pipeline 和逐样本文件，见第三节。
- EFC 当前参数、Git 演化、四数据集同参、router/static-QD 实际规则，以及缺少系统搜索证据这一事实。来源：Git 历史、`run_racp.py`、`efc.py`、FINAL configs。
- 最终 prompts、parser、stop、max-token 差异，以及 IRCoT demonstration/专用 parser、FLARE 空 context。来源：第五节所列代码与持久化 prompt。
- EFC route 比例、route-wise EM/F1、planner failure/repair、候选池、三阶段支持标题转化与失败漏斗。来源：四个 EFC FINAL、缓存与数据集 metadata；见第六节。
- 显著性检验所需的逐样本 ID/gold/pred/EM/F1 均完整且严格对齐。来源：第九节哈希核对。
- HotpotQA已登记的direct/generation-only、w/o static/role/source/seed/redundancy、RRF-only、title-dedup-only、controlled top5和matched maximum retrieval budget全量消融；来源为第七节目录及[`PAPER_EXPERIMENT_PLAN.md`](PAPER_EXPERIMENT_PLAN.md)。

## B. 可以通过重新统计得到，不需要重新调用模型

- 按第三节方案统一重算 `FinalContextAnswerHit@5/@10`、IterRetGen/FLARE cumulative hit，并更正论文指标名称。
- 为 HP/2W/MU 统一适配 supporting-title 字段与 robust title normalization；另做 redirect/alias 映射敏感性版本。
- 扫描 `wiki18_100w` corpus，先确认 gold support title/paragraph 是否存在，再把“语料缺失”和“retriever miss”拆开。
- 对四数据集已保存逐样本 EM/F1 执行 paired bootstrap 置信区间、paired randomization p 值，并对成组 p 值做 Holm–Bonferroni 校正。
- 对新登记消融使用已保存的逐样本EM/F1做paired bootstrap，并单独检查小于0.3个百分点的差异。
- Full-QD planner input 可由保存的问题、实际 planner output 与锁定模板重构并统计；EFC 的 probe/planner input 可重构出近似/模板一致 token，但因缺实际 planner raw output，不能恢复严格完整的 EFC output-token 总量。
- 生成 route/type/question-category 的进一步分层表、证据已全但 F1 非满分的分析、selector loss 样本清单，都可离线得到。

## C. 必须重新运行实验或带 instrumentation 的补测

- 尚缺的隔离消融只包括always-static、w/o title weight、fixed-quota/MMR等扩展项；auto w/o static、always-direct、role/source/seed/redundancy、RRF-only、title-dedup-only已有FINAL。
- matched maximum retrieval budget已完成；若论文要求token-level或完全LLM成本匹配，仍需新的instrumented实验，不能把当前结果改名为完全成本匹配。
- 受控 final top5/top10 已在同一router/代码下完成；若需要曲线仍可补top6/8，但不是核心缺口。
- 参数敏感性或单变量搜索：现有历史不能证明各权重/阈值的独立贡献。
- 峰值显存：至少需对代表性方法做相同批量和同一 GPU 的 `nvidia-smi`/框架峰值监控；历史 FINAL 的峰值无法追溯。
- FLARE 的严格 input/output token、被丢弃 speculative generation、物理 cache miss 和逐阶段延迟，必须带 instrumentation 补跑；仅从最终拼接文本无法恢复。
- EFC 的严格完整 token 成本需要保存 probe/planner/final 的每次实际 prompt/output 后补测；旧 FINAL 的缺失 planner output 无法补造。
- IRCoT 去 demonstration 或统一 final parser/prompt 的公平性对照，以及更贴近原论文的 FLARE continuation prompt 对照，都需要新运行。

历史上没有实际命令或运行时 diff 的 19 个 FINAL，其 provenance 不能靠“重跑一个新实验”反向恢复；应保持 `UNKNOWN`，并在今后的运行中强制保存 command、commit、diff patch 和 artifact manifest。**这不是重跑 26 个已验证 FINAL 的理由。**

## 最小补实验与论文动作优先级

### P0：投稿前必须完成

1. **无模型调用：** 更正 `Retrieval Recall` 命名并统一离线重算 final-context @5/@10、support-title 与 paired bootstrap；这是当前结果表口径正确性的前提。
2. **无模型调用：** 对EFC/IRCoT与四项W11结果做paired bootstrap，并将matched-budget名称锁定为“matched maximum retrieval budget”。
3. **无模型调用：** 将W11的`063e29d + title-dedup diff`固化到可引用commit/补丁manifest；结果无需重跑。

### P1：强烈建议补充

1. always-static-QD或w/o title weight中至少补一项，用于完善router/selector正交表；现有always-static只是smoke。
2. 做 instrumented 效率 profile：代表性固定子集上记录每次 prompt/output token、cache hit/miss、阶段延迟与峰值显存；质量 FINAL 不必重跑。
3. IRCoT no-demo 或统一 parser 的公平性控制，至少在 HotpotQA 和 2Wiki 上做。

### P2：篇幅和算力允许时

1. w/o title weight、固定配额有/无、MMR及更细参数敏感性；w/o redundancy已有FINAL。
2. 更完整的 router threshold/规则敏感性；centroid router 若不进入论文主方法则无需补。
3. 采用论文式 continuation prompt 的 FLARE 实现对照；当前 FLARE 结果应明确标为 FlashRAG 本地实现。
4. 对 MuSiQue 做 support-sentence 而不只是 title 级诊断；对 NQ 使用适合单跳数据的 evidence provenance 指标。

## 当前现场快照与不可逆证据边界

审计时 HEAD 为 `3b4d78067cdbad6b29fb0154a3820c469b940a8c`。当前工作区有以下已存在的未提交修改，本次均未触碰：

```text
flashrag/generator/generator.py
flashrag/pipeline/active_pipeline.py
flashrag/pipeline/pipeline.py
racp/PAPER_EXPERIMENT_PLAN.md
racp/run_exp.py
racp/test_run_exp_config.py
```

审计现场快照为 2026-07-14 15:44:49 CST：HP、2W、MU TRACE 分别仍在 GPU 0、2、3 运行，不属于 26 个 FINAL；本报告没有读取运行中部分结果作为成绩，也没有干预进程。既有 untracked 文件为 `HOTPOTQA_EXPERIMENT_REPORT.md`、`NEXT_CONVERSATION_PROMPT.md`、`PAPER_RESULTS_REFERENCE.md`、`WEEKLY_REPORT_2026_07_05.md`；`PAPER_MISSING_INFORMATION_AUDIT.md` 是本次按要求新增。当前 diff 不能证明较早运行当时的 diff，因此凡缺同期 ledger/command 的字段均保持 `UNKNOWN`。

当前代码 diff 的实验相关影响如下；这是审计说明，不代表本次认可或修改这些内容：

| 文件 | 当前未提交变化 | 对既有 FINAL 的关系 |
|---|---|---|
| `flashrag/generator/generator.py` | FLARE/vLLM 返回 score 时按实际 selected token id 取 logprob，并初始化输出容器 | HP/2W/MU/NQ FLARE 的同期 compatibility diff；不属于 EFC 算法 |
| `flashrag/pipeline/active_pipeline.py` | FLARE 未触发时保存空 `retrieval_result`；动态 query 改用 batch API 解包，并保存逐轮 query/result | 使 FLARE 评测和 cache miss 形状稳定；影响 FLARE 轨迹完整性，不改变其他方法 |
| `flashrag/pipeline/pipeline.py` | Adaptive-RAG 的 multi-hop max iteration 可配置 | 当前无 Adaptive FINAL |
| `racp/run_exp.py` | 增加 sample/cache/FLARE/TRACE/Adaptive CLI 覆盖，FLARE 显式禁用 refiner，Adaptive 要求本地 checkpoint | 支持近期 FLARE/TRACE/NQ 命令；不是 EFC 主干修改 |
| `racp/test_run_exp_config.py` | 增加上述配置和 FLARE 空检索行为测试 | 测试代码，不改变运行语义 |
| `racp/PAPER_EXPERIMENT_PLAN.md` | 大量实验台账更新 | 是当前唯一正式台账，但尚未提交到 Git |

另有一个文档性不一致：`run_exp.py` 的 SuRe docstring 当前把 official repo URL 改成了 FLARE，而 FLARE docstring 仍残留 SuRe URL。它不影响任何指标，但在锁定 commit 前应人工审阅；本次没有修复。
