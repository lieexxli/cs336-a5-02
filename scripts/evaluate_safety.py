"""
Run safety judging over model responses.

Running:

```
python scripts/evaluate_safety.py \
    --input-path <path_to_predictions.jsonl> \
    --output-path <path_to_write_output.jsonl>
```

This script defaults to DeepSeek's official API (`deepseek-chat` on
`https://api.deepseek.com`). To use a local judge model instead, pass
`--backend local-vllm --model-name-or-path /path/to/model`.
"""
import argparse
import json
import logging
import sys
from statistics import mean

from tqdm import tqdm
from xopen import xopen

from cs336_alignment.deepseek_api import (
    default_deepseek_base_url,
    default_deepseek_model,
    make_deepseek_client,
)
from cs336_alignment.safety_eval import (
    build_safety_judge_messages,
    safety_judge_marks_unsafe,
)

logger = logging.getLogger(__name__)


def _load_input_examples(input_path):
    input_examples = []
    with xopen(input_path) as f:
        for line in f:
            input_examples.append(json.loads(line))
    return input_examples


def _run_local_vllm_judge(input_examples, model_name_or_path, num_gpus):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=6144,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    prompts = []
    for example in input_examples:
        messages = build_safety_judge_messages(
            example["prompts_final"], example["output"]
        )
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)
    raw_responses = model.generate(prompts, sampling_params)
    responses = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()
        responses.append(response)
    assert len(responses) == len(prompts)
    return prompts, responses


def _run_deepseek_judge(input_examples, model_name_or_path, api_base_url):
    client = make_deepseek_client(base_url=api_base_url)

    prompts = []
    responses = []

    for example in tqdm(input_examples, desc="Safety judge"):
        messages = build_safety_judge_messages(
            example["prompts_final"], example["output"]
        )
        completion = client.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )
        prompts.append(messages)
        responses.append((completion.choices[0].message.content or "").strip())

    return prompts, responses


def main(
    input_path,
    model_name_or_path,
    num_gpus,
    output_path,
    backend,
    api_base_url,
):
    input_examples = _load_input_examples(input_path)
    logger.info(f"Read {len(input_examples)} model responses from {input_path}")

    if backend == "local-vllm":
        prompts, responses = _run_local_vllm_judge(
            input_examples, model_name_or_path, num_gpus
        )
    else:
        prompts, responses = _run_deepseek_judge(
            input_examples, model_name_or_path, api_base_url
        )

    logger.info(f"Processed {len(prompts)} prompts")

    all_metrics = []
    with xopen(output_path, "w") as fout:
        for input_example, prompt, response in tqdm(
            zip(input_examples, prompts, responses)
        ):
            metrics = {
                "safe": 0.0 if safety_judge_marks_unsafe(response) else 1.0
            }
            all_metrics.append(metrics)

            fout.write(
                json.dumps(
                    {
                        **input_example,
                        "model_name_or_path": model_name_or_path,
                        "model_prompt": prompt,
                        "model_response": response,
                        "metrics": metrics,
                    }
                )
                + "\n"
            )

    for key in sorted(list(all_metrics[0].keys())):
        metric_value = mean([metrics[key] for metrics in all_metrics])
        logger.info(f"{key}: {metric_value}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to file with model predictions (JSONL format with key 'output')",
    )
    parser.add_argument(
        "--model-name-or-path",
        help="Judge model name. In deepseek-api mode this should be a DeepSeek model id; in local-vllm mode this can be a local path or HF id.",
        default=default_deepseek_model(),
    )
    parser.add_argument(
        "--backend",
        choices=("deepseek-api", "local-vllm"),
        default="deepseek-api",
        help="Judge backend to use. Default is DeepSeek's official API.",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=default_deepseek_base_url(),
        help="Base URL for DeepSeek/OpenAI-compatible API backends.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model_name_or_path,
        args.num_gpus,
        args.output_path,
        args.backend,
        args.api_base_url,
    )
    logger.info("finished running %s", sys.argv[0])
