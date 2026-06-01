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


if requested_stage() == "prepare":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
else:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
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

import numpy as np

from flashrag.config import Config
from flashrag.dataset import Dataset
from flashrag.evaluator import Evaluator
from flashrag.pipeline import BasicPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_dataset, get_generator, get_retriever


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"
DEFAULT_SAVE_DIR = PROJECT_DIR / "output"
PIPELINE_REGISTRY = {}


def register_pipeline(name):
    def decorator(pipeline_cls):
        PIPELINE_REGISTRY[name] = pipeline_cls
        return pipeline_cls

    return decorator


def none_or_int(value):
    if isinstance(value, int):
        return value
    value = value.strip().lower()
    if value in {"none", "null", "-1"}:
        return None
    return int(value)


def doc_to_text(doc):
    if isinstance(doc, str):
        return doc
    if not isinstance(doc, dict):
        return str(doc)
    if doc.get("contents"):
        return doc["contents"]
    parts = [doc.get("title"), doc.get("text")]
    return "\n".join(part for part in parts if part)


def l2_normalize(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def save_result_outputs(config, dataset, file_name="mmr_result.json"):
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / file_name
    dataset.save(str(result_path))
    print(f"MMR result saved to: {result_path}")
    if config["save_metric_score"]:
        print(f"Metric score saved to: {save_dir / 'metric_score.txt'}")
    if config["save_intermediate_data"]:
        print(f"Intermediate data saved to: {save_dir / 'intermediate_data.json'}")


def build_config_dict(args):
    if args.stage == "generate":
        config_dict = {"disable_save": True}
        if args.gpu_id is not None:
            config_dict["gpu_id"] = args.gpu_id
        if args.gpu_memory_utilization is not None:
            config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization
        return config_dict

    mmr_topk = args.mmr_topk if args.mmr_topk is not None else args.max_k
    config_dict = {
        "refiner_name": None,
        "refiner_model_path": None,
        #"use_reranker": not args.no_reranker,
        "retrieval_topk": args.retrieval_topk,
        "rerank_topk": args.rerank_topk,
        "mmr_config": {
            "lambda": args.mmr_lambda,
            "topk": mmr_topk,
        },
        "racp_config": {
            "buffer": args.buffer,
            "max_k": args.max_k,
            "search_ratio": args.search_ratio,
        },
        "save_note": args.save_note,
        "save_dir": str(args.save_dir),
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name or "nq",
        "split": args.split or "test",
    }

    if args.test_sample_num is not None:
        config_dict["test_sample_num"] = args.test_sample_num
    if args.gpu_memory_utilization is not None:
        config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization

    return config_dict


@register_pipeline("mmr")
class MMRPipeline(BasicPipeline):
    """Sequential RAG pipeline with MMR selection before prompt construction."""

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, load_generator=True):
        super().__init__(config, prompt_template)
        self.retriever = retriever if retriever is not None else get_retriever(config)
        self.generator = generator if generator is not None else (get_generator(config) if load_generator else None)
        self.use_fid = config["use_fid"]

        mmr_config = config["mmr_config"] or {}
        self.mmr_lambda = float(mmr_config.get("lambda", 0.5))
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"MMR lambda must be in [0, 1], got {self.mmr_lambda}")

        self.mmr_topk = mmr_config.get("topk", None)
        if self.mmr_topk is not None:
            self.mmr_topk = int(self.mmr_topk)
            if self.mmr_topk < 1:
                raise ValueError(f"MMR topk must be positive or None, got {self.mmr_topk}")

    def _encode_with_retriever(self, texts, is_query):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoder = getattr(self.retriever, "encoder", None)
        if encoder is None:
            raise ValueError(
                "MMR needs retriever embeddings for document-document similarity, "
                "but the current retriever has no encoder."
            )
        batch_size = self.config["retrieval_batch_size"] or 64
        return l2_normalize(encoder.encode(texts, batch_size=batch_size, is_query=is_query))

    def _scores_are_usable(self, scores, expected_len):
        if scores is None or len(scores) != expected_len:
            return False
        try:
            score_array = np.asarray(scores, dtype=np.float32)
        except (TypeError, ValueError):
            return False
        return np.all(np.isfinite(score_array)) and not np.allclose(score_array, score_array[0])

    def _query_doc_similarity(self, query, docs, scores, doc_embeddings):
        if self._scores_are_usable(scores, len(docs)):
            score_array = np.asarray(scores, dtype=np.float32)
            score_min = float(score_array.min())
            score_max = float(score_array.max())
            if score_min < -1.0 or score_max > 1.0:
                score_range = score_max - score_min
                if score_range <= 1e-12:
                    return None
                score_array = (score_array - score_min) / score_range
            return score_array

        query_embedding = self._encode_with_retriever([query], is_query=True)
        return (query_embedding @ doc_embeddings.T).reshape(-1)

    def _select_mmr_docs(self, query, docs, scores):
        if len(docs) <= 1:
            return docs, list(range(len(docs))), [float(score) for score in scores] if scores is not None else []

        topk = len(docs) if self.mmr_topk is None else min(self.mmr_topk, len(docs))
        doc_texts = [doc_to_text(doc) for doc in docs]
        doc_embeddings = self._encode_with_retriever(doc_texts, is_query=False)
        query_doc_sim = self._query_doc_similarity(query, docs, scores, doc_embeddings)
        if query_doc_sim is None:
            query_embedding = self._encode_with_retriever([query], is_query=True)
            query_doc_sim = (query_embedding @ doc_embeddings.T).reshape(-1)

        doc_doc_sim = doc_embeddings @ doc_embeddings.T
        selected = [int(np.argmax(query_doc_sim))]
        candidates = set(range(len(docs))) - set(selected)

        while candidates and len(selected) < topk:
            best_idx = None
            best_score = None
            for candidate_idx in candidates:
                max_selected_sim = float(np.max(doc_doc_sim[candidate_idx, selected]))
                mmr_score = (
                    self.mmr_lambda * float(query_doc_sim[candidate_idx])
                    - (1.0 - self.mmr_lambda) * max_selected_sim
                )
                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = candidate_idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        selected_docs = [docs[idx] for idx in selected]
        selected_query_scores = [float(query_doc_sim[idx]) for idx in selected]
        return selected_docs, selected, selected_query_scores

    def build_final_prompts(self, dataset):
        if not self.use_fid:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc["contents"] for doc in docs])

        dataset.update_output("prompt", input_prompts)
        return dataset

    def prepare_dataset(self, dataset):
        retrieval_results, retrieval_scores = self.retriever.batch_search(dataset.question, return_score=True)
        dataset.update_output("retrieval_result_full", retrieval_results)
        dataset.update_output("retrieval_score_full", retrieval_scores)

        selected_results = []
        selected_indices = []
        selected_query_scores = []
        for query, docs, scores in zip(dataset.question, retrieval_results, retrieval_scores):
            selected_docs, indices, query_scores = self._select_mmr_docs(query, docs, scores)
            selected_results.append(selected_docs)
            selected_indices.append(indices)
            selected_query_scores.append(query_scores)

        dataset.update_output("retrieval_result", selected_results)
        dataset.update_output("mmr_selected_indices", selected_indices)
        dataset.update_output("mmr_selected_query_scores", selected_query_scores)
        dataset.update_output("mmr_topk", [len(docs) for docs in selected_results])
        dataset.update_output("mmr_lambda", [self.mmr_lambda for _ in selected_results])

        dataset = self.build_final_prompts(dataset)

        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        if self.generator is None:
            raise ValueError("MMRPipeline.run requires a generator. Use prepare_dataset for prepare-only runs.")

        dataset = self.prepare_dataset(dataset)
        pred_answer_list = self.generator.generate(dataset.prompt)
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        save_result_outputs(self.config, dataset)
        return dataset


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


def build_pipeline(config, **kwargs):
    return PIPELINE_REGISTRY["mmr"](config, **kwargs)


def run_prepare(config, args):
    dataset = load_split(config, args.split)
    pipeline = build_pipeline(config, load_generator=False)
    dataset = pipeline.prepare_dataset(dataset)

    prompt_cache_path = args.prompt_cache_path
    if prompt_cache_path is None:
        prompt_cache_path = Path(config["save_dir"]) / "prompt_cache.json"
    save_prompt_cache(dataset, Path(prompt_cache_path))
    return dataset


def run_generate(config, args):
    if args.prompt_cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")

    prompt_cache_dir = Path(args.prompt_cache_path).resolve().parent
    config["save_dir"] = str(prompt_cache_dir)
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_prompt_cache(config, args.prompt_cache_path)
    generator = get_generator(config)
    pred_answer_list = generator.generate(dataset.prompt)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    save_result_outputs(config, dataset)
    return dataset


def run_full(config, args):
    generator = get_generator(config)
    dataset = load_split(config, args.split)
    pipeline = build_pipeline(config, generator=generator)
    return pipeline.run(dataset)


def run(args):
    if args.stage == "generate" and args.prompt_cache_path is not None:
        prompt_config_path = Path(args.prompt_cache_path).resolve().parent / "config.yaml"
        if Path(args.config_path).resolve() == DEFAULT_CONFIG_PATH.resolve() and prompt_config_path.exists():
            args.config_path = prompt_config_path

    config = Config(str(args.config_path), build_config_dict(args))
    if args.stage == "prepare":
        return run_prepare(config, args)
    if args.stage == "generate":
        return run_generate(config, args)

    return run_full(config, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the standalone MMR experiment.")
    parser.add_argument("--stage", choices=["full", "prepare", "generate"], default="full")
    parser.add_argument("--prompt_cache_path", type=Path, default=None)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_note", type=str, default="mmr")

    parser.add_argument("--retrieval_topk", type=int, default=20)
    parser.add_argument("--rerank_topk", type=int, default=20)
    parser.add_argument("--no_reranker", action="store_true")

    parser.add_argument("--mmr_lambda", type=float, default=0.5)
    parser.add_argument("--mmr_topk", type=none_or_int, default=None)

    parser.add_argument("--buffer", type=int, default=5)
    parser.add_argument("--max_k", type=none_or_int, default=8)
    parser.add_argument("--search_ratio", type=float, default=0.9)

    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--test_sample_num", type=none_or_int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
