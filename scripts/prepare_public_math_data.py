#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from cs336_alignment.public_data import make_eval_record, make_sft_record, write_jsonl
from cs336_alignment.repro import default_math_dir, resolve_repo_file


HF_BASE_URL = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main"
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public Hendrycks MATH dataset and convert it to this repo's JSONL format."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_math_dir(),
        help="Directory where train.jsonl, validation.jsonl, and sft.jsonl will be written.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("cs336_alignment/prompts/r1_zero.prompt"),
        help="Prompt template used to construct SFT prompts.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/_raw_math"),
        help="Directory used to store downloaded parquet files and extracted raw files.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing extracted dataset in --cache-dir without downloading again.",
    )
    return parser.parse_args()


def download_math_from_hf(math_root: Path, cache_dir: Path) -> None:
    """Download MATH dataset parquet files from HuggingFace and build JSON directory structure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        for split_hf, split_out in [("train", "train"), ("test", "test")]:
            url = f"{HF_BASE_URL}/{subject}/{split_hf}-00000-of-00001.parquet"
            local_pq = cache_dir / f"{subject}_{split_hf}.parquet"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, local_pq)
            df = pd.read_parquet(local_pq)
            out_dir = math_root / split_out / subject
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, row in df.iterrows():
                record = {
                    "problem": row["problem"],
                    "level": row["level"],
                    "type": row["type"],
                    "solution": row["solution"],
                }
                with open(out_dir / f"{i:04d}.json", "w") as f:
                    json.dump(record, f)
            print(f"    -> {len(df)} examples to {out_dir}")


def iter_math_examples(split_root: Path, split_name: str):
    for json_path in sorted(split_root.glob("*/*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            example = json.load(f)
        source_file = json_path.relative_to(split_root.parent).as_posix()
        yield example, split_name, source_file


def main() -> None:
    args = parse_args()

    output_dir = resolve_repo_file(args.output_dir)
    prompt_template_path = resolve_repo_file(args.prompt_template)
    cache_dir = resolve_repo_file(args.cache_dir)
    extracted_root = cache_dir / "MATH"

    if not args.skip_download or not extracted_root.exists():
        download_math_from_hf(extracted_root, cache_dir)

    with open(prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    train_records = []
    validation_records = []
    sft_records = []
    skipped_train = 0
    skipped_validation = 0

    for example, split_name, source_file in iter_math_examples(
        extracted_root / "train", "train"
    ):
        try:
            train_records.append(make_eval_record(example, split_name, source_file))
            sft_records.append(
                make_sft_record(example, prompt_template, split_name, source_file)
            )
        except ValueError:
            skipped_train += 1

    for example, split_name, source_file in iter_math_examples(
        extracted_root / "test", "validation"
    ):
        try:
            validation_records.append(make_eval_record(example, split_name, source_file))
        except ValueError:
            skipped_validation += 1

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "validation.jsonl", validation_records)
    write_jsonl(output_dir / "sft.jsonl", sft_records)

    print(f"Wrote {len(train_records):,} train examples to {output_dir / 'train.jsonl'}")
    print(
        f"Wrote {len(validation_records):,} validation examples to {output_dir / 'validation.jsonl'}"
    )
    print(f"Wrote {len(sft_records):,} SFT examples to {output_dir / 'sft.jsonl'}")
    if skipped_train or skipped_validation:
        print(
            f"Skipped {skipped_train:,} train examples and {skipped_validation:,} validation examples without a boxed final answer."
        )


if __name__ == "__main__":
    main()
