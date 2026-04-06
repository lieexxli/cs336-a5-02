#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from cs336_alignment.repro import resolve_repo_file


SFT_TRAIN_URL = (
    "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/"
    "safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
)
SFT_TEST_URL = (
    "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/"
    "safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"
)

HF_DATASETS = {
    "mmlu": "cais/mmlu",
    "gsm8k": "openai/gsm8k",
    "alpaca_eval": "tatsu-lab/alpaca_eval",
    "simple_safety_tests": "Bertievidgen/SimpleSafetyTests",
    "hh_rlhf": "Anthropic/hh-rlhf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all public datasets needed by the supplemental "
            "instruction-tuning / RLHF / safety portion of this repository."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/supplemental"),
        help="Root directory for supplemental datasets.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "benchmarks", "sft", "preferences"],
        default="all",
        help="Subset of supplemental data to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if local copies already exist.",
    )
    return parser.parse_args()


def ensure_download(url: str, out_path: Path, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"Reuse existing file: {out_path}")
        return
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def ensure_snapshot(
    repo_id: str,
    local_dir: Path,
    force: bool,
    allow_patterns: list[str] | None = None,
) -> None:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if force and local_dir.exists():
        shutil.rmtree(local_dir)
    print(f"Downloading dataset repo {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )


def download_benchmarks(root: Path, force: bool) -> None:
    benchmarks_root = root / "benchmarks"
    ensure_snapshot(HF_DATASETS["mmlu"], benchmarks_root / "mmlu", force)
    ensure_snapshot(HF_DATASETS["gsm8k"], benchmarks_root / "gsm8k", force)
    ensure_snapshot(HF_DATASETS["alpaca_eval"], benchmarks_root / "alpaca_eval", force)
    ensure_snapshot(
        HF_DATASETS["simple_safety_tests"],
        benchmarks_root / "simple_safety_tests",
        force,
    )


def download_sft_data(root: Path, force: bool) -> None:
    sft_root = root / "sft" / "safety_augmented_ultrachat_200k_single_turn"
    ensure_download(SFT_TRAIN_URL, sft_root / "train.jsonl.gz", force)
    ensure_download(SFT_TEST_URL, sft_root / "test.jsonl.gz", force)


def download_preference_data(root: Path, force: bool) -> None:
    pref_root = root / "preferences" / "hh-rlhf"
    ensure_snapshot(
        HF_DATASETS["hh_rlhf"],
        pref_root,
        force,
        allow_patterns=["*.jsonl.gz", "*.json", "README.md"],
    )


def main() -> None:
    args = parse_args()
    root = resolve_repo_file(args.output_dir)

    if args.stage in {"all", "benchmarks"}:
        download_benchmarks(root, args.force)
    if args.stage in {"all", "sft"}:
        download_sft_data(root, args.force)
    if args.stage in {"all", "preferences"}:
        download_preference_data(root, args.force)

    print(f"Supplemental data prepared under {root}")


if __name__ == "__main__":
    main()
