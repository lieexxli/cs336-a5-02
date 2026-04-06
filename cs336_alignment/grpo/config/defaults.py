from dataclasses import dataclass, field

from cs336_alignment.repro import (
    default_math_dir,
    default_model_id,
    default_output_dir,
)


@dataclass
class PathsConfig:
    train_examples_path: str = field(
        default_factory=lambda: str(default_math_dir() / "train.jsonl")
    )
    val_examples_path: str = field(
        default_factory=lambda: str(default_math_dir() / "validation.jsonl")
    )
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    model_path: str = field(default_factory=default_model_id)
    model_output: str = field(
        default_factory=lambda: str(default_output_dir() / "grpo-experiments")
    )


@dataclass
class TrainingConfig:
    seed: int = 42
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    vllm_device: str = "cuda:0"

    n_grpo_steps: int = 200
    learning_rate: float = 3e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy
    gradient_accumulation_steps: int = 128  # microbatch size is 2, will fit on H100

    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    sampling_top_p: float = 1.0

    gpu_memory_utilization: float = 0.2

    # Have to retry
    loss_type: str = "reinforce_with_baseline"  # "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"

    use_std_normalization: bool = True

    # - "mean" (div. each by seq's adv. by it's seq. len)
    # - "constant" (divide all seq adv. by max seq. len)
    # - "microbatch" (divide all seq adv. by longest seq. in microbatch)
    normalize_mode: str = "constant"

    max_grad_norm: float = 1.0
    cliprange: float = 0.2
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    wandb_entity: str | None = None
    wandb_project: str | None = None
    torch_compile: bool = True
    log_step_interval: int = 1
    wandb_tags: list[str] = field(default_factory=lambda: ["grpo"])

    eval_before_training: bool = False
    eval_step_interval: int = 20
    eval_n_examples: int = 1024

    old_log_probs_batch_size: int = 8

    question_only: bool = False


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
