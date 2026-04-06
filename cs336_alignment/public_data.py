from __future__ import annotations

import json
from pathlib import Path

from cs336_alignment.drgrpo_grader import extract_answer


def extract_ground_truth(solution: str) -> str:
    answer = extract_answer(solution)
    if answer is None:
        raise ValueError("Could not extract a boxed answer from the solution.")
    return answer.strip()


def render_r1_zero_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.replace("{question}", question.strip())


def render_r1_zero_response(solution: str, answer: str | None = None) -> str:
    ground_truth = answer or extract_ground_truth(solution)
    reasoning = solution.strip()
    if reasoning.endswith("</think>"):
        reasoning = reasoning[: -len("</think>")].rstrip()
    return f"{reasoning}\n</think> <answer>{ground_truth}</answer>"


def make_eval_record(example: dict, source_split: str, source_file: str) -> dict:
    return {
        "problem": example["problem"].strip(),
        "answer": extract_ground_truth(example["solution"]),
        "level": example.get("level"),
        "type": example.get("type"),
        "source_split": source_split,
        "source_file": source_file,
    }


def make_sft_record(
    example: dict, prompt_template: str, source_split: str, source_file: str
) -> dict:
    question = example["problem"].strip()
    answer = extract_ground_truth(example["solution"])
    return {
        "prompt": render_r1_zero_prompt(question, prompt_template),
        "response": render_r1_zero_response(example["solution"], answer),
        "ground_truth": answer,
        "question": question,
        "level": example.get("level"),
        "type": example.get("type"),
        "source_split": source_split,
        "source_file": source_file,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
