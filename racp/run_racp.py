import argparse
import os
import sys
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from flashrag.config import Config
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

    if args.test_sample_num is not None:
        config_dict["test_sample_num"] = args.test_sample_num
    if args.gpu_memory_utilization is not None:
        config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization

    return config_dict


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


def run(args):
    from flashrag.pipeline import RACPPipeline

    config = Config(str(args.config_path), build_config_dict(args))
    test_data = load_split(config, args.split)
    pipeline = RACPPipeline(config)
    return pipeline.run(test_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the standalone RACP experiment.")
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
