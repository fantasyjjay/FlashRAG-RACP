import argparse
import json
import os
import sys
from pathlib import Path


def requested_stage():
    for arg in sys.argv:
        if arg.startswith("--stage="):
            return arg.split("=", 1)[1]
    if "--stage" in sys.argv:
        stage_idx = sys.argv.index("--stage") + 1
        if stage_idx < len(sys.argv):
            return sys.argv[stage_idx]
    return "full"


if requested_stage() == "generate":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
else:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_conda_runtime_libs():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_lib = str(Path(conda_prefix) / "lib")
    if not Path(conda_lib).is_dir():
        return

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [path for path in ld_library_path.split(":") if path]
    if paths and paths[0] == conda_lib:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([conda_lib] + [path for path in paths if path != conda_lib])
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


ensure_conda_runtime_libs()

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from flashrag.config import Config
from flashrag.dataset import Dataset
from flashrag.evaluator import Evaluator
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_dataset


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"
DEFAULT_SAVE_DIR = PROJECT_DIR / "output"
DEFAULT_MODEL_DIR = Path.home() / "my_models"


def none_or_int(value):
    if isinstance(value, int):
        return value
    value = value.strip().lower()
    if value in {"none", "null", "-1"}:
        return None
    return int(value)


def build_config_dict(args):
    config_dict = {
        "refiner_name": "selective-context",
        "refiner_model_path": str(args.refiner_model_path),
        "use_reranker": not args.no_reranker,
        "retrieval_topk": args.retrieval_topk,
        "rerank_topk": args.rerank_topk,
        "sc_config": {"reduce_ratio": args.reduce_ratio},
        "racp_config": {
            "buffer": args.buffer,
            "max_k": args.max_k,
            "search_ratio": args.search_ratio,
        },
        "save_note": args.save_note,
        "save_dir": str(args.save_dir),
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    if args.stage == "generate":
        config_dict["disable_save"] = True

    if args.test_sample_num is not None:
        config_dict["test_sample_num"] = args.test_sample_num
    if args.gpu_memory_utilization is not None:
        config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization

    return config_dict


def select_adaptive_docs(docs, scores, racp_config):
    if len(docs) <= 1:
        return docs, len(docs), None

    import numpy as np

    buffer = racp_config.get("buffer", 5)
    search_ratio = racp_config.get("search_ratio", 0.9)
    max_k = racp_config.get("max_k", None)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ranked_docs = [doc for doc, _ in ranked]
    ranked_scores = [float(score) for _, score in ranked]

    search_doc_num = int(np.ceil(len(ranked_scores) * search_ratio))
    search_doc_num = min(len(ranked_scores), max(2, search_doc_num))
    gap_search_scores = ranked_scores[:search_doc_num]
    gaps = [
        gap_search_scores[idx] - gap_search_scores[idx + 1]
        for idx in range(len(gap_search_scores) - 1)
    ]

    gap_idx = int(np.argmax(gaps))
    selected_k = min(len(ranked_docs), gap_idx + 1 + buffer)
    if max_k is not None:
        selected_k = min(selected_k, max_k)

    return ranked_docs[:selected_k], selected_k, gap_idx


def load_split(config, split):
    all_split = get_dataset(config)
    data = all_split[split]
    if data is None:
        available = sorted(path.stem for path in Path(config["dataset_path"]).glob("*.jsonl"))
        raise FileNotFoundError(
            f"Split '{split}' not found for dataset '{config['dataset_name']}'. "
            f"Available splits: {available}"
        )
    return data


def build_final_prompts(config, dataset):
    from flashrag.utils import get_refiner

    prompt_template = PromptTemplate(config)
    refiner = get_refiner(config) if config["refiner_name"] is not None else None

    if refiner:
        input_prompt_flag = refiner.input_prompt_flag
        if "llmlingua" in refiner.name and input_prompt_flag:
            input_prompts = [
                prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
            dataset.update_output("prompt", input_prompts)
            input_prompts = refiner.batch_run(dataset)
        else:
            refine_results = refiner.batch_run(dataset)
            dataset.update_output("refine_result", refine_results)
            input_prompts = [
                prompt_template.get_string(question=q, formatted_reference=r)
                for q, r in zip(dataset.question, refine_results)
            ]
    else:
        input_prompts = [
            prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]

    if config["use_fid"]:
        print("Use FiD generation")
        input_prompts = []
        for item in dataset:
            q = item.question
            docs = item.retrieval_result
            input_prompts.append([q + " " + doc["contents"] for doc in docs])

    dataset.update_output("prompt", input_prompts)
    return dataset


def save_prompt_cache(dataset, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(save_path))
    print(f"Prompt cache saved to: {save_path}")


def load_prompt_cache(config, cache_path):
    if cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")
    cache_path = Path(cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset(config=config, data=data)


def run_prepare(config, args):
    from flashrag.utils import get_retriever

    dataset = load_split(config, args.split)
    retriever = get_retriever(config)

    retrieval_results, retrieval_scores = retriever.batch_search(dataset.question, return_score=True)
    dataset.update_output("retrieval_result_full", retrieval_results)
    dataset.update_output("retrieval_score_full", retrieval_scores)

    selected_results = []
    adaptive_ks = []
    adaptive_gap_indices = []
    racp_config = config["racp_config"] or {}
    for docs, scores in zip(retrieval_results, retrieval_scores):
        selected_docs, selected_k, gap_idx = select_adaptive_docs(docs, scores, racp_config)
        selected_results.append(selected_docs)
        adaptive_ks.append(selected_k)
        adaptive_gap_indices.append(gap_idx)

    dataset.update_output("retrieval_result", selected_results)
    dataset.update_output("adaptive_k", adaptive_ks)
    dataset.update_output("adaptive_gap_index", adaptive_gap_indices)

    dataset = build_final_prompts(config, dataset)

    if config["save_retrieval_cache"]:
        retriever._save_cache()

    prompt_cache_path = args.prompt_cache_path
    if prompt_cache_path is None:
        prompt_cache_path = Path(config["save_dir"]) / "prompt_cache.json"
    save_prompt_cache(dataset, Path(prompt_cache_path))
    return dataset


def run_generate(config, args):
    from flashrag.utils import get_generator

    if args.prompt_cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")

    prompt_cache_dir = Path(args.prompt_cache_path).resolve().parent
    config["save_dir"] = str(prompt_cache_dir)
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_prompt_cache(config, args.prompt_cache_path)
    input_prompts = dataset.prompt

    generator = get_generator(config)
    pred_answer_list = generator.generate(input_prompts)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    return dataset


def run(args):
    from flashrag.pipeline import RACPPipeline

    config = Config(str(args.config_path), build_config_dict(args))
    if args.stage == "prepare":
        return run_prepare(config, args)
    if args.stage == "generate":
        return run_generate(config, args)

    test_data = load_split(config, args.split)
    pipeline = RACPPipeline(config)
    return pipeline.run(test_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the standalone RACP experiment.")
    parser.add_argument("--stage", choices=["full", "prepare", "generate"], default="full")
    parser.add_argument("--prompt_cache_path", type=Path, default=None)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_name", type=str, default="nq")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_note", type=str, default="racp")

    parser.add_argument("--retrieval_topk", type=int, default=20)
    parser.add_argument("--rerank_topk", type=int, default=20)
    parser.add_argument("--no_reranker", action="store_true")

    parser.add_argument("--buffer", type=int, default=5)
    parser.add_argument("--max_k", type=none_or_int, default=8)
    parser.add_argument("--search_ratio", type=float, default=0.9)
    parser.add_argument("--reduce_ratio", type=float, default=0.5)

    parser.add_argument("--refiner_model_path", type=Path, default=DEFAULT_MODEL_DIR / "gpt2")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--test_sample_num", type=none_or_int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
