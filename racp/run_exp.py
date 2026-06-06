import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path

RACP_DIR = Path(__file__).resolve().parent
REPO_DIR = RACP_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse
import json


CONFIG_PATH = RACP_DIR / "config.yaml"
RUNTIME_CONFIG_OVERRIDES = {}


def load_config(config_dict=None):
    config_dict = {} if config_dict is None else dict(config_dict)
    config_dict.setdefault("save_dir", str(RACP_DIR / "output"))
    config_dict.update(RUNTIME_CONFIG_OVERRIDES)
    return Config(str(CONFIG_PATH), config_dict)


def load_config_from_path(config_path, config_dict=None):
    config_dict = {} if config_dict is None else dict(config_dict)
    config_dict.update(RUNTIME_CONFIG_OVERRIDES)
    return Config(str(config_path), config_dict)


def set_runtime_config_overrides(args):
    RUNTIME_CONFIG_OVERRIDES.clear()
    if args.save_note is not None:
        RUNTIME_CONFIG_OVERRIDES["save_note"] = args.save_note
    if args.retrieval_topk is not None:
        RUNTIME_CONFIG_OVERRIDES["retrieval_topk"] = args.retrieval_topk
    if args.retrieval_batch_size is not None:
        RUNTIME_CONFIG_OVERRIDES["retrieval_batch_size"] = args.retrieval_batch_size
    if args.rerank_topk is not None:
        RUNTIME_CONFIG_OVERRIDES["rerank_topk"] = args.rerank_topk
    if args.gpu_memory_utilization is not None:
        RUNTIME_CONFIG_OVERRIDES["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.use_reranker:
        RUNTIME_CONFIG_OVERRIDES["use_reranker"] = True
    if args.no_reranker:
        RUNTIME_CONFIG_OVERRIDES["use_reranker"] = False
    if args.save_retrieval_cache:
        RUNTIME_CONFIG_OVERRIDES["save_retrieval_cache"] = True
    if args.use_retrieval_cache:
        RUNTIME_CONFIG_OVERRIDES["use_retrieval_cache"] = True
    if args.retrieval_cache_path is not None:
        RUNTIME_CONFIG_OVERRIDES["retrieval_cache_path"] = str(args.retrieval_cache_path)


def normalize_args(args):
    base_config = load_config({"disable_save": True})
    if args.dataset_name is None:
        args.dataset_name = base_config["dataset_name"]
    if args.split is None:
        args.split = base_config["split"]
    if isinstance(args.split, (list, tuple)):
        if len(args.split) != 1:
            raise ValueError(f"racp/run_exp.py expects exactly one split, got {args.split}.")
        args.split = args.split[0]
    if args.gpu_id is None:
        args.gpu_id = base_config["gpu_id"]
    return args


def default_prompt_cache_path(config):
    return Path(config["save_dir"]) / "prompt_cache.json"


def save_prompt_cache(dataset, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(save_path))
    print(f"Prompt cache saved to: {save_path}")


def load_prompt_cache(config, cache_path):
    from flashrag.dataset import Dataset

    with Path(cache_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset(config=config, data=data)


def save_retrieval_cache_from_dataset(config, dataset):
    cache = {}
    for item in dataset:
        docs = []
        for doc in item.retrieval_result:
            new_doc = dict(doc)
            if "score" not in new_doc:
                raise ValueError("Cannot save retrieval cache: retrieved document has no score.")
            docs.append(new_doc)
        cache[item.question] = docs

    save_path = Path(config["save_dir"]) / "retrieval_cache.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)
    print(f"Retrieval cache saved to: {save_path}")


def native_prepare(args):
    if args.method_name != "naive":
        raise ValueError("--stage prepare currently supports --method_name naive only.")

    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_reranker, get_retriever

    config = load_config(
        {
            "save_note": args.save_note or "naive-prepare",
            "gpu_id": args.gpu_id,
            "dataset_name": args.dataset_name,
            "split": args.split,
        }
    )
    if args.source_prompt_cache is not None:
        dataset = load_prompt_cache(config, args.source_prompt_cache)
        if not dataset.output or any("retrieval_result" not in item_output for item_output in dataset.output):
            raise ValueError(f"Source prompt cache has no retrieval_result: {args.source_prompt_cache}")
        print(f"Loaded retrieval_result from source prompt cache: {args.source_prompt_cache}")

        if config["use_reranker"]:
            reranker = get_reranker(config)
            reranked_docs, rerank_scores = reranker.rerank(
                dataset.question,
                dataset.retrieval_result,
                topk=config["rerank_topk"],
            )
            for docs, scores in zip(reranked_docs, rerank_scores):
                for doc, score in zip(docs, scores):
                    doc["score"] = float(score)
            dataset.update_output("retrieval_result", reranked_docs)

        if config["save_retrieval_cache"]:
            save_retrieval_cache_from_dataset(config, dataset)
    else:
        all_split = get_dataset(config)
        dataset = all_split[args.split]
        retriever = get_retriever(config)
        retrieval_results = retriever.batch_search(dataset.question)
        dataset.update_output("retrieval_result", retrieval_results)

        if config["save_retrieval_cache"]:
            retriever._save_cache()

    prompt_template = PromptTemplate(config)
    input_prompts = [
        prompt_template.get_string(question=q, retrieval_result=r)
        for q, r in zip(dataset.question, dataset.retrieval_result)
    ]
    dataset.update_output("prompt", input_prompts)

    prompt_cache_path = args.prompt_cache_path or default_prompt_cache_path(config)
    save_prompt_cache(dataset, prompt_cache_path)
    return dataset


def native_generate(args):
    if args.method_name != "naive":
        raise ValueError("--stage generate currently supports --method_name naive only.")
    if args.prompt_cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")

    from flashrag.evaluator import Evaluator
    from flashrag.utils import get_generator

    config = load_config(
        {
            "save_note": args.save_note or "naive-generate",
            "gpu_id": args.gpu_id,
            "dataset_name": args.dataset_name,
            "split": args.split,
        }
    )
    dataset = load_prompt_cache(config, args.prompt_cache_path)
    generator = get_generator(config)
    pred_answer_list = generator.generate(dataset.prompt)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    return dataset


REFINER_METHODS = {"selective-context", "llmlingua"}


def refiner_method_config(args):
    base_config = load_config({"disable_save": True})
    model2path = base_config["model2path"]

    common = {
        "save_note": args.save_note or args.method_name,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "retrieval_topk": args.retrieval_topk or 20,
        "rerank_topk": args.rerank_topk or 5,
        "use_reranker": False if args.no_reranker else True,
        "metric_setting": {"retrieval_recall_topk": args.rerank_topk or 5},
    }

    if args.method_name == "selective-context":
        common.update(
            {
                "refiner_name": "selective-context",
                "refiner_model_path": model2path.get("gpt2", "/home/guanjunjie/my_models/gpt2"),
                "sc_config": {"reduce_ratio": 0.5},
            }
        )
        return common

    if args.method_name == "llmlingua":
        common.update(
            {
                "refiner_name": "longllmlingua",
                "refiner_model_path": model2path.get("llama2-7B", "/home/guanjunjie/my_models/Llama-2-7b-hf"),
                "llmlingua_config": {
                    "rate": 0.55,
                    "condition_in_question": "after_condition",
                    "reorder_context": "sort",
                    "dynamic_context_compression_ratio": 0.3,
                    "condition_compare": True,
                    "context_budget": "+100",
                    "rank_method": "longllmlingua",
                },
                "refiner_input_prompt_flag": False,
            }
        )
        return common

    raise ValueError(f"Unsupported refiner method: {args.method_name}")


def refiner_prepare(args):
    if args.method_name not in REFINER_METHODS:
        raise ValueError("--stage prepare supports naive, selective-context, and llmlingua.")

    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_refiner, get_retriever

    config = load_config(refiner_method_config(args))

    if args.source_prompt_cache is not None:
        dataset = load_prompt_cache(config, args.source_prompt_cache)
        if not dataset.output or any("retrieval_result" not in item_output for item_output in dataset.output):
            raise ValueError(f"Source prompt cache has no retrieval_result: {args.source_prompt_cache}")
        print(f"Loaded retrieval_result from source prompt cache: {args.source_prompt_cache}")
    else:
        all_split = get_dataset(config)
        dataset = all_split[args.split]
        retriever = get_retriever(config)
        retrieval_results = retriever.batch_search(dataset.question)
        dataset.update_output("retrieval_result", retrieval_results)
        if config["save_retrieval_cache"]:
            retriever._save_cache()

    refiner = get_refiner(config)
    refine_results = refiner.batch_run(dataset)
    dataset.update_output("refine_result", refine_results)

    prompt_template = PromptTemplate(config)
    input_prompts = [
        prompt_template.get_string(question=q, formatted_reference=r)
        for q, r in zip(dataset.question, dataset.refine_result)
    ]
    dataset.update_output("prompt", input_prompts)

    prompt_cache_path = args.prompt_cache_path or default_prompt_cache_path(config)
    save_prompt_cache(dataset, prompt_cache_path)
    return dataset


def refiner_generate(args):
    if args.method_name not in REFINER_METHODS:
        raise ValueError("--stage generate supports naive, selective-context, and llmlingua.")
    if args.prompt_cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")

    from flashrag.evaluator import Evaluator
    from flashrag.utils import get_generator

    prompt_cache_dir = Path(args.prompt_cache_path).resolve().parent
    prompt_config_path = prompt_cache_dir / "config.yaml"
    config_path = prompt_config_path if prompt_config_path.exists() else CONFIG_PATH
    config = load_config_from_path(
        config_path,
        {
            "save_note": args.save_note or f"{args.method_name}-generate",
            "save_dir": str(prompt_cache_dir),
            "gpu_id": args.gpu_id,
            "dataset_name": args.dataset_name,
            "split": args.split,
        },
    )
    dataset = load_prompt_cache(config, args.prompt_cache_path)
    generator = get_generator(config)
    pred_answer_list = generator.generate(dataset.prompt)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    return dataset


def naive(args):
    save_note = "naive"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split,
                   "retrieval_topk": 5}

    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pipeline = SequentialPipeline(config)

    result = pipeline.run(test_data)


def zero_shot(args):
    class NoOpRetriever:
        def _save_cache(self):
            pass

    save_note = args.save_note or "zero-shot"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "metrics": ["em", "f1", "acc", "precision", "recall"],
    }

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = SequentialPipeline(config, prompt_template=templete, retriever=NoOpRetriever())
    result = pipeline.naive_run(test_data)


def aar(args):
    """
    Reference:
        Zichun Yu et al. "Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In"
        in ACL 2023.
        Official repo: https://github.com/OpenMatch/Augmentation-Adapted-Retriever
    """
    # two types of checkpoint: ance / contriever
    # retrieval_method = "AAR-contriever"  # AAR-ANCE
    # index path of this retriever
    retrieval_method = args.method_name
    if "contriever" in retrieval_method:
        index_path = "aar-contriever_Flat.index"
    else:
        index_path = "aar-ance_Flat.index"

    model2path = {"AAR-contriever": "model/AAR-Contriever-KILT", "AAR-ANCE": "model/AAR-ANCE"}
    model2pooling = {"AAR-contriever": "mean", "AAR-ANCE": "cls"}
    save_note = retrieval_method
    config_dict = {
        "retrieval_method": retrieval_method,
        "model2path": model2path,
        "index_path": index_path,
        "model2pooling": model2pooling,
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)
    # result = pipeline.run(test_data, pred_process_fun=pred_process_fun)
    result = pipeline.run(test_data)


def llmlingua(args):
    """
    Reference:
        Huiqiang Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
        in EMNLP 2023
        Huiqiang Jiang et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"
        in ICLR MEFoMo 2024.
        Official repo: https://github.com/microsoft/LLMLingua
    """
    config_dict = refiner_method_config(args)
    config_dict["save_note"] = args.save_note or "longllmlingua"

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)


def recomp(args):
    """
    Reference:
        Fangyuan Xu et al. "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation"
        in ICLR 2024.
        Official repo: https://github.com/carriex/recomp
    """
    # ###### Specified parameters ######
    refiner_name = "recomp-abstractive"  # recomp-extractive
    model_dict = {
        "nq": "model/recomp_nq_abs",
        "triviaqa": "model/recomp_tqa_abs",
        "hotpotqa": "model/recomp_hotpotqa_abs",
    }

    refiner_model_path = model_dict.get(args.dataset_name, None)
    refiner_max_input_length = 1024
    refiner_max_output_length = 512
    # parameters for extractive compress
    refiner_topk = 5
    refiner_pooling_method = "mean"
    refiner_encode_max_length = 256

    config_dict = {
        "refiner_name": refiner_name,
        "refiner_model_path": refiner_model_path,
        "refiner_max_input_length": refiner_max_input_length,
        "refiner_max_output_length": refiner_max_output_length,
        "refiner_topk": 5,
        "refiner_pooling_method": refiner_pooling_method,
        "refiner_encode_max_length": refiner_encode_max_length,
        "save_note": refiner_name,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)


def sc(args):
    """
    Reference:
        Yucheng Li et al. "Compressing Context to Enhance Inference Efficiency of Large Language Models"
        in EMNLP 2023.
        Official repo: https://github.com/liyucheng09/Selective_Context

    Note:
        Need to install spacy:
            ```python -m spacy download en_core_web_sm```
        or
            ```
            wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0.tar.gz
            pip install en_core_web_sm-3.6.0.tar.gz
            ```
    """
    config_dict = refiner_method_config(args)
    config_dict["save_note"] = args.save_note or "selective-context"

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)


def retrobust(args):
    """
    Reference:
        Ori Yoran et al. "Making Retrieval-Augmented Language Models Robust to Irrelevant Context"
        in ICLR 2024.
        Official repo: https://github.com/oriyor/ret-robust
    """
    model_dict = {
        "nq": "model/llama-2-13b-peft-nq-retrobust",
        "2wiki": "model/llama-2-13b-peft-2wikihop-retrobust",
    }
    if args.dataset_name in ["nq", "triviaqa", "popqa", "web_questions"]:
        lora_path = model_dict["nq"]
    elif args.dataset_name in ["hotpotqa", "2wikimultihopqa"]:
        lora_path = model_dict["2wiki"]
    else:
        print("Not use lora")
        lora_path = model_dict.get(args.dataset_name, None)
    config_dict = {
        "save_note": "Ret-Robust",
        "generator_model": "llama2-13B",
        "generator_lora_path": lora_path,
        "generation_params": {"max_tokens": 100},
        "gpu_id": args.gpu_id,
        "generator_max_input_len": 4096,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SelfAskPipeline
    from flashrag.utils import selfask_pred_parse

    pipeline = SelfAskPipeline(config, max_iter=5, single_hop=False)
    # use specify prediction parse function
    result = pipeline.run(test_data, pred_process_fun=selfask_pred_parse)


def sure(args):
    """
    Reference:
        Jaehyung Kim et al. "SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs"
        in ICLR 2024
        Official repo: https://github.com/bbuing9/ICLR24_SuRe
    """
    config_dict = {"save_note": "SuRe", "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split}
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SuRePipeline

    pipeline = SuRePipeline(config)
    pred_process_fun = lambda x: x.split("\n")[0]
    result = pipeline.run(test_data)


def replug(args):
    """
    Reference:
        Weijia Shi et al. "REPLUG: Retrieval-Augmented Black-Box Language Models".
    """
    save_note = "replug"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split}

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    pred_process_fun = lambda x: x.split("\n")[0]

    from flashrag.pipeline import REPLUGPipeline

    pipeline = REPLUGPipeline(config)
    result = pipeline.run(test_data)


def skr(args):
    """
    Reference:
        Yile Wang et al. "Self-Knowledge Guided Retrieval Augmentation for Large Language Models"
        in EMNLP Findings 2023.
        Official repo: https://github.com/THUNLP-MT/SKR/

    Note:
        `skr-knn` need training data in inference stage to determain whether to retrieve. training data should in
        `.json` format in following format:
        format:
            [
                {
                    "question": ... ,  // question
                    "judgement": "ir_better" / "ir_worse" / "same",  // judgement result, can be obtained by comparing
                    ...
                },
                ...
            ]

    """
    judger_name = "skr"
    model_path = "model/sup-simcse-bert-base-uncased"
    training_data_path = "./sample_data/skr_training.json"

    config_dict = {
        "judger_name": judger_name,
        "judger_config": {
            "model_path": model_path,
            "training_data_path": training_data_path,
            "topk": 5,
            "batch_size": 64,
            "max_length": 128,
        },
        "save_note": "skr",
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ConditionalPipeline

    pipeline = ConditionalPipeline(config)
    result = pipeline.run(test_data)


def selfrag(args):
    """
    Reference:
        Akari Asai et al. " SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection"
        in ICLR 2024.
        Official repo: https://github.com/AkariAsai/self-rag
    """
    config_dict = {
        "generator_model": "selfrag-llama2-7B",
        "generator_model_path": "model/selfrag_llama2_7b",
        "framework": "vllm",
        "save_note": "self-rag",
        "gpu_id": args.gpu_id,
        "generation_params": {
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "skip_special_tokens": False,
        },
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    config = load_config(config_dict)

    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SelfRAGPipeline

    pipeline = SelfRAGPipeline(
        config,
        threshold=0.2,
        max_depth=2,
        beam_width=2,
        w_rel=1.0,
        w_sup=1.0,
        w_use=1.0,
        use_grounding=True,
        use_utility=True,
        use_seqscore=True,
        ignore_cont=True,
        mode="adaptive_retrieval",
    )
    result = pipeline.run(test_data, long_form=False)


def flare(args):
    """
    Reference:
        Zhengbao Jiang et al. "Active Retrieval Augmented Generation"
        in EMNLP 2023.
        Official repo: https://github.com/bbuing9/ICLR24_SuRe

    """
    config_dict = {"save_note": "flare", "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split}
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import FLAREPipeline

    pipeline = FLAREPipeline(config)
    result = pipeline.run(test_data)


def iterretgen(args):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023.
    """
    iter_num = 3
    config_dict = {
        "save_note": "iter-retgen",
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IterativePipeline

    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)


def ircot(args):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    save_note = "ircot"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split}

    from flashrag.pipeline import IRCOTPipeline

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    print(config["generator_model_path"])
    pipeline = IRCOTPipeline(config, max_iter=5)

    result = pipeline.run(test_data)


def trace(args):
    """
    Reference:
        Jinyuan Fang et al. "TRACE the Evidence: Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented Generation"
    """

    save_note = "trace"
    trace_config = {
        "num_examplars": 3,
        "max_chain_length": 4,
        "topk_triple_select": 5,  # num of candidate triples
        "num_choices": 20,
        "min_triple_prob": 1e-4,
        "num_beams": 5,  # number of selected prob at each step of constructing chain
        "num_chains": 20,  # number of generated chains
        "n_context": 5,  # number of used chains in generation
        "context_type": "triples",  # triples/triple-doc
    }
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "refiner_name": "kg-trace",
        "trace_config": trace_config,
        "framework": "hf",  # Trance only supports using Huggingface Transformers since it needs logits of outputs
        "split": args.split,
    }

    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)

    result = pipeline.run(test_data)


def spring(args):
    """
    Reference:
        Yutao Zhu et al. "One Token Can Help! Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models"
    """

    save_note = "spring"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "framework": "hf",
        "split": args.split,
    }
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    # download token embedding from: https://huggingface.co/yutaozhu94/SPRING
    token_embedding_path = "llama2.7b.chat.added_token_embeddings.pt"

    from flashrag.prompt import PromptTemplate
    from flashrag.pipeline import SequentialPipeline
    from flashrag.utils import get_generator, get_retriever

    # prepare prompt and generator for Spring method
    system_prompt = (
        "Answer the question based on the given document."
        "Only give me the answer and do not output any other words."
        "\nThe following are given documents.\n\n{reference}"
    )
    added_tokens = [f" [ref{i}]" for i in range(1, 51)]
    added_tokens = "".join(added_tokens)
    user_prompt = added_tokens + "Question: {question}\nAnswer:"
    prompt_template = PromptTemplate(config, system_prompt, user_prompt, enable_chat=False)

    generator = get_generator(config)
    generator.add_new_tokens(token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]")

    pipeline = SequentialPipeline(config=config, prompt_template=prompt_template, generator=generator)
    result = pipeline.run(test_data)


def adaptive(args):
    judger_name = "adaptive-rag"
    model_path = "illuminoplanet/adaptive-rag-classifier"

    config_dict = {
        "judger_name": judger_name,
        "judger_config": {"model_path": model_path},
        "save_note": "adaptive-rag",
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    # preparation
    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline(config)
    result = pipeline.run(test_data)

def rqrag(args):
    """
    Function to run the RQRAGPipeline.
    """
    save_note = "rqrag"
    max_depth = 3
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'generator_max_input_len': 4096,
        'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
        'generator_model_path': 'zorowin123/rq_rag_llama2_7B',
        "dataset_name": args.dataset_name,
        "split": args.split,
        "max_depth": max_depth
    }

    config = load_config(config_dict)
    
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import RQRAGPipeline
    pipeline = RQRAGPipeline(config, max_depth = max_depth)
    result = pipeline.run(test_data)


def r1searcher(args):
    """
    Function to run the R1-Searcher.
    """
    save_note = "r1-searcher"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'generator_max_input_len': 16384,
        'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
        'generator_model_path': 'XXsongLALA/Qwen-2.5-7B-base-RAG-RL',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline(config)
    result = pipeline.run(test_data)


def searchr1(args):
    """
    Function to run the search-r1.
    """
    save_note = "search-r1"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'generator_max_input_len': 16384,
        'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
        'generator_model_path':'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import SearchR1Pipeline
    pipeline = SearchR1Pipeline(config)
    result = pipeline.run(test_data)
    
def autorefine(args):
    """
    Function to run the AutoRefine.
    """
    save_note = "autorefine"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'generator_max_input_len': 16384,
        'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
        'generator_model_path': 'yrshi/AutoRefine-Qwen2.5-3B-Base',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import AutoRefinePipeline
    pipeline = AutoRefinePipeline(config)
    result = pipeline.run(test_data)


def o2searcher(args):
    """
    Function to run the O2searcher.
    """
    save_note = "O2searcher"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'retrieval_topk': 3,
        'generator_max_input_len': 16384,
        'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
        'generator_model_path': 'Jianbiao/O2-Searcher-Qwen2.5-3B-GRPO',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import O2SearcherPipeline
    pipeline = O2SearcherPipeline(config)
    result = pipeline.run(test_data)
    
def rearag(args):
    """
    Function to run the rearag.
    """
    save_note = "rearag"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'is_reasoning': True,
        'generator_max_input_len': 8192 - 2 - 1024,
        'generation_params': {'max_tokens': 1024, 'do_sample': True},
        'generator_model_path': 'THU-KEG/ReaRAG-9B',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import ReaRAGPipeline
    pipeline = ReaRAGPipeline(config)
    result = pipeline.run(test_data)
    
def corag(args):
    """
    Function to run the CoRAG.
    """
    save_note = "corag"
    task_desc = 'Given a search query, retrieve relevant documents that can help answer the query'
    if args.dataset_name in ['hotpotqa', 'musique', '2wikimultihopqa', 'bamboogle']:
        task_desc = 'Given a multi-hop question, retrieve documents that can help answer the question'
    if args.dataset_name in ['nq', 'triviaqa']:
        task_desc = 'Given a question, retrieve Wikipedia passages that answer the question'
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'task_desc': task_desc,
        'framework': 'vllm',
        'is_reasoning': True,
        'generator_max_input_len': 4096,
        'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
        'generator_model_path': 'corag/CoRAG-Llama3.1-8B-MultihopQA',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import CoRAGPipeline
    pipeline = CoRAGPipeline(config)
    result = pipeline.run(test_data)
    
def simpledeepsearcher(args):
    """
    Function to run the SimpleDeepSearcher.
    """
    save_note = "simpledeepsearcher"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        'framework': 'vllm',
        'generator_max_input_len': 20480,
        'generation_params': {'max_tokens': 2048, 'skip_special_tokens': False},
        'generator_model_path': 'RUC-AIBOX/Qwen-7B-SimpleDeepSearcher',
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    config = load_config(config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]   

    from flashrag.pipeline import SimpleDeepSearcherPipeline
    pipeline = SimpleDeepSearcherPipeline(config)
    result = pipeline.run(test_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--stage", choices=["full", "prepare", "generate"], default="full")
    parser.add_argument("--prompt_cache_path", type=Path)
    parser.add_argument("--split", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--save_note", type=str)
    parser.add_argument("--retrieval_topk", type=int)
    parser.add_argument("--retrieval_batch_size", type=int)
    parser.add_argument("--rerank_topk", type=int)
    parser.add_argument("--gpu_memory_utilization", type=float)
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument("--save_retrieval_cache", action="store_true")
    parser.add_argument("--use_retrieval_cache", action="store_true")
    parser.add_argument("--retrieval_cache_path", type=Path)
    parser.add_argument("--source_prompt_cache", type=Path)

    
    func_dict = {
        "AAR-contriever": aar,
        "AAR-ANCE": aar,
        "naive": naive,
        "zero-shot": zero_shot,
        "llmlingua": llmlingua,
        "recomp": recomp,
        "selective-context": sc,
        "ret-robust": retrobust,
        "sure": sure,
        "replug": replug,
        "skr": skr,
        "selfrag": selfrag,
        "flare": flare,
        "iterretgen": iterretgen,
        "ircot": ircot,
        "trace": trace,
        "adaptive": adaptive,
        "rqrag": rqrag,
        "r1-searcher": r1searcher,
        "search-r1": searchr1,
        "autorefine":autorefine,
        "o2-searcher": o2searcher,
        "rearag": rearag,
        "corag": corag,
        "simpledeepsearcher": simpledeepsearcher,
    }

    args = normalize_args(parser.parse_args())
    set_runtime_config_overrides(args)
    if args.stage == "prepare":
        if args.method_name == "naive":
            native_prepare(args)
        elif args.method_name in REFINER_METHODS:
            refiner_prepare(args)
        else:
            raise ValueError(f"--stage prepare is not supported for {args.method_name}.")
        raise SystemExit(0)
    if args.stage == "generate":
        if args.method_name == "naive":
            native_generate(args)
        elif args.method_name in REFINER_METHODS:
            refiner_generate(args)
        else:
            raise ValueError(f"--stage generate is not supported for {args.method_name}.")
        raise SystemExit(0)
    func = func_dict[args.method_name]
    func(args)
