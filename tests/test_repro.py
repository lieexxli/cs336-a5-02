from pathlib import Path

from cs336_alignment.public_data import (
    extract_ground_truth,
    make_eval_record,
    make_sft_record,
    render_r1_zero_response,
)
from cs336_alignment.repro import (
    PUBLIC_MODEL_ID,
    resolve_data_path,
    resolve_model_path,
    resolve_output_path,
)


def test_resolve_data_path_maps_cluster_math_path_to_repo_data():
    resolved = resolve_data_path("/data/a5-alignment/MATH/train.jsonl")
    assert resolved == Path("data/MATH/train.jsonl") or resolved.as_posix().endswith(
        "/data/MATH/train.jsonl"
    )


def test_resolve_output_path_maps_cluster_output_to_repo_runs():
    resolved = resolve_output_path(
        "/data/c-sniderb/a5-alignment/grpo-experiments/lr_sweep/lr1e-5"
    )
    assert resolved == Path(
        "runs/grpo-experiments/lr_sweep/lr1e-5"
    ) or resolved.as_posix().endswith("/runs/grpo-experiments/lr_sweep/lr1e-5")


def test_resolve_model_path_maps_cluster_model_to_public_id():
    resolved = resolve_model_path("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    assert resolved == PUBLIC_MODEL_ID


def test_extract_ground_truth_reads_boxed_answer():
    assert extract_ground_truth("We compute the result. Final answer is \\boxed{42}.") == "42"


def test_render_r1_zero_response_appends_answer_block():
    response = render_r1_zero_response(
        "We compute the result. Final answer is \\boxed{42}.", "42"
    )
    assert response.endswith("</think> <answer>42</answer>")


def test_make_records_from_public_math_example():
    example = {
        "problem": "What is 6 * 7?",
        "solution": "Multiply to get \\boxed{42}.",
        "level": "Level 1",
        "type": "prealgebra",
    }
    prompt_template = "User: {question}\nAssistant: <think>"

    eval_record = make_eval_record(example, "train", "train/prealgebra/0.json")
    sft_record = make_sft_record(
        example, prompt_template, "train", "train/prealgebra/0.json"
    )

    assert eval_record["answer"] == "42"
    assert sft_record["question"] == "What is 6 * 7?"
    assert sft_record["prompt"].endswith("What is 6 * 7?\nAssistant: <think>")
    assert sft_record["response"].endswith("</think> <answer>42</answer>")
