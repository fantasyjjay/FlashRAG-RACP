#!/usr/bin/env python3
"""Download local models required by the RACP experiments.

Default target directory matches racp/config.yaml:
    ~/my_models/

Examples:
    python scripts/download_racp_models.py
    python scripts/download_racp_models.py --only generator
    python scripts/download_racp_models.py --model-dir /data/guanjunjie/my_models
"""

import argparse
import os
from pathlib import Path


RACP_MODELS = {
    "retriever": {
        "repo_id": "intfloat/e5-base-v2",
        "local_name": "e5-base-v2",
        "note": "E5 dense retriever used by racp/config.yaml",
    },
    "reranker": {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "local_name": "bge-reranker-v2-m3",
        "note": "Default cross-encoder reranker used by RACP",
    },
    "refiner": {
        "repo_id": "openai-community/gpt2",
        "local_name": "gpt2",
        "note": "GPT2 model used by Selective-Context compression",
    },
    "generator": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "local_name": "Meta-Llama-3-8B-Instruct",
        "note": "Final QA generator. This gated model may require HF login/token.",
    },
}

OPTIONAL_MODELS = {
    "qwen-reranker": {
        "repo_id": "Qwen/Qwen3-Reranker-0.6B",
        "local_name": "Qwen3-Reranker-0.6B",
        "note": "Optional reranker for ablation.",
    },
    "bge-reranker-base": {
        "repo_id": "BAAI/bge-reranker-base",
        "local_name": "bge-reranker-base",
        "note": "Optional reranker for ablation.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download models for RACP experiments.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "my_models",
        help="Directory to store downloaded models. Default: ~/my_models",
    )
    parser.add_argument(
        "--only",
        choices=list(RACP_MODELS.keys()) + ["all"],
        default="all",
        help="Download only one required model group, or all required models.",
    )
    parser.add_argument(
        "--with-optional",
        action="store_true",
        help="Also download optional reranker ablation models.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Can also be set with HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume incomplete downloads. Default: true.",
    )
    return parser.parse_args()


def selected_models(args):
    if args.only == "all":
        models = dict(RACP_MODELS)
    else:
        models = {args.only: RACP_MODELS[args.only]}

    if args.with_optional:
        models.update(OPTIONAL_MODELS)
    return models


def download_one(model_dir, name, spec, token, resume):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `huggingface_hub`. Install it with:\n"
            "  pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc

    local_dir = model_dir / spec["local_name"]
    print(f"\n===== {name}: {spec['repo_id']} =====")
    print(f"note: {spec['note']}")
    print(f"to:   {local_dir}")

    snapshot_download(
        repo_id=spec["repo_id"],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=resume,
        token=token,
    )
    print(f"done: {local_dir}")


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    models = selected_models(args)
    print(f"Model directory: {args.model_dir}")
    print("Models to download:")
    for name, spec in models.items():
        print(f"  - {name}: {spec['repo_id']} -> {spec['local_name']}")

    for name, spec in models.items():
        try:
            download_one(args.model_dir, name, spec, args.token, args.resume)
        except Exception as exc:
            print(f"\nERROR while downloading {name} ({spec['repo_id']}): {exc}")
            if "Meta-Llama-3-8B-Instruct" in spec["repo_id"]:
                print(
                    "Llama3-8B-Instruct is a gated Hugging Face model. "
                    "Make sure your account has access and run either:\n"
                    "  huggingface-cli login\n"
                    "or:\n"
                    "  export HF_TOKEN=your_token"
                )
            raise

    print("\nAll requested models downloaded.")
    print("Expected racp/config.yaml paths:")
    for spec in models.values():
        print(f"  {args.model_dir / spec['local_name']}")


if __name__ == "__main__":
    main()
