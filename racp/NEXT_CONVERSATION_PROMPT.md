# 下一轮 Codex 对话提示词

将下面代码块中的内容作为下一轮对话的首条消息：

```text
你将继续维护我在 /home/guanjunjie/jayj/FlashRAG 中的 RACP/EFC-RAG 论文实验。

开始任何修改或实验前，请完整阅读：
/home/guanjunjie/jayj/FlashRAG/racp/PAPER_EXPERIMENT_PLAN.md

该文档是唯一正式实验台账，包含：研究目标、EFC-RAG 三分支算法、模型与硬件环境、
代码入口、固定参数、缓存规则、正式命令、已有 FINAL 结果、待测矩阵和验收要求。
同时按需查看：
- /home/guanjunjie/jayj/FlashRAG/racp/run_racp.py
- /home/guanjunjie/jayj/FlashRAG/racp/efc.py
- /home/guanjunjie/jayj/FlashRAG/racp/run_exp.py
- /home/guanjunjie/jayj/FlashRAG/examples/methods/run_exp.py

项目目标：在 HotpotQA、2WikiMultiHopQA、MuSiQue 三个多跳数据集和 NQ 单跳数据集上，
使用相同 Llama-3.1-8B-Instruct、bge-large-en-v1.5、wiki18_100w、无 reranker 设置，比较
No-RAG、Naive RAG、IterRetGen、Full-QD、IRCoT、Adaptive-RAG 和 EFC-RAG。论文重点
不是 SOTA，而是证明 EFC 的证据反馈路由相比固定检索/固定分解有更好的质量成本平衡。

严格规则：
1. smoke、失败运行、调参中间版本不写入 PAPER_EXPERIMENT_PLAN.md 的结果表。
2. 只有完整 split、配置审计通过的运行才能标记 FINAL。
3. 论文主表关闭 reranker；EFC final_topk 必须显式为 10。
4. IRCoT 使用 examples/methods/run_exp.py 的 ircot-no-rerank，max_iter=2、topk=5。
5. 不要把 prompt cache、query retrieval cache、QD retrieval-topk bundle 混用。
6. 不要回滚当前未提交修改；先审计 git diff。正式新实验前应帮助我锁定并提交基线。
7. 最多使用 4 张 RTX 4090；首次 FAISS 全量检索不要与另一个全量检索任务并发。
8. 不要覆盖或删除任何 FINAL 输出目录。

当前已锁定 HotpotQA FINAL：
- IterRetGen：EM 34.67，F1 45.66
- Full-QD：EM 33.67，F1 44.59
- IRCoT：EM 36.26，F1 47.50
- EFC-RAG top10：EM 35.85，F1 47.21，Retrieval Recall@10 71.51

下一阶段优先级：
1. 先检查 GPU、运行进程、git status 和 racp/output 下最新正式结果，确认有没有尚未登记的完成任务。
2. 审计并锁定当前代码 commit。
3. 依次完成 2Wiki 的 Naive RAG、IterRetGen、EFC-RAG 正式全量。
4. 再完成 MuSiQue 的同三项实验。
5. 只有跨数据集结果合理后，再补 Full-QD、IRCoT、Adaptive-RAG 和 NQ。

请先向我汇报：
- 你从 PAPER_EXPERIMENT_PLAN.md 理解到的当前状态；
- 当前 Git/GPU/进程/最新输出状态；
- 下一项正式实验的准确命令、预计使用的缓存和风险。

除非我明确要求立即运行，否则先完成上述审计，不要直接启动数小时的全量任务。
```
