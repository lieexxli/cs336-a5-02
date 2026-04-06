import json
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from pydantic import BaseModel
from cs336_alignment.repro import resolve_data_path

TRAIN_EXAMPLES_PATH: Path = resolve_data_path("data/MATH/sft.jsonl")


class EvalResult(BaseModel):
    prompt: str
    completion: str
    ground_truth: str
    rewards: dict[str, float]


def main():
    with open(TRAIN_EXAMPLES_PATH, "r") as f:
        examples = [json.loads(line) for line in f]

    prompts = [ex["prompt"] for ex in examples]
    completions = [ex["response"] for ex in examples]
    answers = [ex["ground_truth"] for ex in examples]

    results = [
        EvalResult(
            prompt=prompt,
            completion=completion,
            ground_truth=answer,
            rewards=r1_zero_reward_fn(completion, answer),
        )
        for prompt, completion, answer in zip(prompts, completions, answers)
    ]

    correct_results = [result for result in results if result.rewards["reward"]]

    print(
        f"Correct examples: {len(correct_results)} / {len(results)} ({len(correct_results) / len(results) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
