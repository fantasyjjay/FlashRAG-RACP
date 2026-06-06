import argparse
import json
import os
import random
import sys
from pathlib import Path


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
from flashrag.utils import get_dataset, get_generator
from run_racp import (
    DEFAULT_CONFIG_PATH,
    build_query_decomposition_planner_config,
    generate_subqueries,
    release_generator,
)


DEFAULT_OUTPUT_PATH = PROJECT_DIR / "output" / "planner_decomposition_sample.jsonl"


def build_config(args):
    config_dict = {
        "disable_save": True,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "test_sample_num": None,
        "decomposition_config": {
            "enabled": True,
            "subquery_num": 2,
            "subquery_topk": 5,
            "planner_model": args.planner_model,
            "planner_model_path": str(args.planner_model_path)
            if args.planner_model_path is not None
            else None,
            "planner_gpu_memory_utilization": args.planner_gpu_memory_utilization,
            "planner_batch_size": args.planner_batch_size,
            "planner_max_tokens": args.planner_max_tokens,
        },
    }
    if args.gpu_memory_utilization is not None:
        config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization
    return Config(str(args.config_path), config_dict)


def load_random_items(config, split, sample_num, seed):
    dataset = get_dataset(config)[split]
    if dataset is None:
        available = sorted(path.stem for path in Path(config["dataset_path"]).glob("*.jsonl"))
        raise FileNotFoundError(
            f"Split '{split}' not found for dataset '{config['dataset_name']}'. "
            f"Available splits: {available}"
        )

    rng = random.Random(seed)
    sample_num = min(sample_num, len(dataset))
    indices = rng.sample(range(len(dataset)), sample_num)
    return [dataset[idx] for idx in indices], indices


def run(args):
    config = build_config(args)
    items, indices = load_random_items(config, args.split, args.sample_num, args.seed)
    questions = [item.question for item in items]

    planner_config = build_query_decomposition_planner_config(config)
    generator = get_generator(planner_config)
    decomposition_config = dict(config["decomposition_config"])
    decomposition_config["planner_model_path"] = planner_config["generator_model_path"]
    planner_records = generate_subqueries(
        questions,
        generator,
        decomposition_config,
    )
    release_generator(generator)

    records = []
    for idx, item, planner_record in zip(indices, items, planner_records):
        raw_output = planner_record["raw_planner_output"]
        parsed_subqueries = planner_record["parsed_subqueries"]
        record = {
            "dataset_index": idx,
            "id": item.id,
            "question": item.question,
            "raw_planner_output": raw_output,
            "parsed_subqueries": parsed_subqueries,
            "valid": planner_record["valid"],
            "invalid": not planner_record["valid"],
            "partial": planner_record["partial"],
            "fallback": planner_record["fallback"],
        }
        records.append(record)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    valid_num = sum(record["valid"] for record in records)
    print(f"Saved planner decomposition sample to: {args.output_path}")
    print(f"Valid: {valid_num}/{len(records)}")
    for idx, record in enumerate(records, 1):
        status = "valid" if record["valid"] else "partial" if record["partial"] else "invalid"
        print(f"\n=== Sample {idx} | {status} | fallback={record['fallback']} ===")
        print(f"question: {record['question']}")
        print(f"raw planner output: {record['raw_planner_output']}")
        print(f"parsed subqueries: {record['parsed_subqueries']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample HotpotQA questions and test planner decomposition.")
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--planner_model", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--planner_model_path", type=Path, default=None)
    parser.add_argument("--planner_gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--planner_batch_size", type=int, default=32)
    parser.add_argument("--planner_max_tokens", type=int, default=96)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
