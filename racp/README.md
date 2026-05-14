# RACP

这个目录是 RACP 的独立实验入口，配置文件在 `racp/config.yaml`，底层复用 `flashrag.pipeline.RACPPipeline`。当前流程是：

1. E5 检索 `retrieval_topk` 篇候选文档，默认 20。
2. 可选 reranker 重排并保留 `rerank_topk` 篇，默认 20。
3. 在重排后的分数上用最大 gap 加 buffer 选择动态 `k`。
4. 对选中的文档运行 Selective-Context 压缩。
5. 把压缩后的上下文送入 Llama3-8B-Instruct 生成答案。

## 运行

在 FlashRAG 根目录执行：

```bash
conda activate flashrag
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python racp/run_racp.py --dataset_name nq --split test --gpu_id 2
```

输出默认保存在 `racp/output/` 下，目录名形如：

```text
nq_YYYY_MM_DD_HH_MM_racp/
```

## 分段运行

如果一张卡同时放不下 retriever、reranker、Selective-Context 和 Llama3，可以先只生成最终 prompt：

```bash
python racp/run_racp.py --stage prepare --dataset_name nq --split test --gpu_id 2
```

这一步会执行：

```text
检索 -> reranker -> RACP 动态 k -> Selective-Context -> final prompt
```

默认保存到当前输出目录的：

```text
prompt_cache.json
```

然后单独加载大模型生成和评测：

```bash
python racp/run_racp.py \
  --stage generate \
  --gpu_id 2 \
  --prompt_cache_path racp/output/nq_YYYY_MM_DD_HH_MM_racp/prompt_cache.json
```

`generate` 阶段不会再新建时间戳目录，`metric_score.txt` 和 `intermediate_data.json` 会写回 `prompt_cache.json` 所在的 prepare 目录。
脚本会在 `full` 和 `generate` 阶段自动把 vLLM 的 worker 启动方式切到 `fork`，避免子进程重复执行入口脚本。
如果 cache 同目录存在 `config.yaml`，`generate` 阶段会默认读取那份配置，因此 HotpotQA 等数据集第二阶段只需要传 `--gpu_id` 和 `--prompt_cache_path`。

完整一次跑完仍然使用默认的 `full`：

```bash
python racp/run_racp.py --stage full --dataset_name nq --split test --gpu_id 2
```

## 常用消融

当前较好的设置：

```bash
python racp/run_racp.py --dataset_name nq --split test --gpu_id 2 --max_k 8
```

不限制最大 k：

```bash
python racp/run_racp.py --dataset_name nq --split test --gpu_id 2 --max_k none
```

关闭 reranker：

```bash
python racp/run_racp.py --dataset_name nq --split test --gpu_id 2 --no_reranker
```

只快速跑 100 条：

```bash
python racp/run_racp.py --dataset_name nq --split test --gpu_id 2 --test_sample_num 100
```

如果显存紧张，可以降低 vLLM 的显存预留：

```bash
python racp/run_racp.py --dataset_name nq --split test --gpu_id 2 --gpu_memory_utilization 0.65
```

## 关键参数

- `--retrieval_topk`: 第一阶段召回数量，默认 20。
- `--rerank_topk`: reranker 后保留数量，默认 20。
- `--buffer`: 最大 gap 位置后额外保留的文档数，默认 5。
- `--max_k`: 最终送入 Selective-Context/LLM 的文档上限，默认 8；传 `none` 表示不限制。
- `--search_ratio`: 用前多少比例的重排分数搜索最大 gap，默认 0.9。
- `--reduce_ratio`: Selective-Context 的压缩比例，默认 0.5。
- `--stage`: 运行阶段，`full` 完整跑，`prepare` 只保存最终 prompt，`generate` 只读取 prompt 后生成和评测。
- `--prompt_cache_path`: `generate` 阶段读取的 prompt cache 路径。
