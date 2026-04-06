# 复现指南

本文档面向需要在自己服务器上复现本项目的人员或自动化执行器。

## 项目范围

本仓库实现了 CS 336 Assignment 5 的 MATH 主线：

| 模块 | 脚本 | 状态 |
|------|------|------|
| MATH 数据集准备 | `scripts/prepare_public_math_data.py` | **可直接运行** |
| Baseline 评测 | `cs336_alignment/math_baseline.py` | **可直接运行** |
| SFT 训练 | `cs336_alignment/sft_exp/train.py` | **可直接运行**（需 2 GPU） |
| Expert Iteration | `cs336_alignment/expert_iteration_exp/train.py` | **可直接运行**（需 2 GPU） |
| GRPO 训练 | `cs336_alignment/grpo/train.py` | **可直接运行**（需 1 GPU） |
| Supplemental 作业部分 | `tests/adapters.py` 等 | **未实现**，忽略 |

## 1. 硬件要求

| 阶段 | 最低要求 |
|------|---------|
| 数据准备 | 无 GPU；~15 GB 磁盘；可访问公网 |
| Baseline 评测 | 1 块 GPU，建议 ≥ 16 GB 显存 |
| SFT 训练 | 默认 2 块 GPU；单卡可行，需覆盖 `vllm_device=cuda:0` + 调低 `gpu_memory_utilization` |
| Expert Iteration | 同 SFT |
| GRPO 训练 | **1 块 GPU**，建议 ≥ 40 GB 显存（训练和 vLLM 均在 `cuda:0`） |

GRPO 默认参数（`rollout_batch_size=256`，`n_grpo_steps=200`）显存需求大，H100/A100 40GB+ 更稳妥。

## 2. 环境安装

需要 Python 3.11 或 3.12，以及支持 CUDA 的 `uv`。

`flash-attn` 需要在有 CUDA 工具链的机器上编译安装：

```bash
# 跳过 flash-attn 先同步其余依赖（数据准备阶段不需要 flash-attn）
uv sync --no-install-package flash-attn

# 进入训练阶段前，安装完整依赖（需要 CUDA 编译环境）
uv sync
```

## 3. 环境变量配置

复制示例文件并填写：

```bash
cp .env.example .env
```

`.env` 内容说明：

```bash
# 数据根目录，留空则使用仓库内 data/ 目录
CS336_ALIGNMENT_DATA_DIR=

# 训练输出根目录，留空则使用仓库内 runs/ 目录
CS336_ALIGNMENT_OUTPUT_DIR=

# 从 HuggingFace 下载模型，或填写服务器本地路径
CS336_ALIGNMENT_MODEL=Qwen/Qwen2.5-Math-1.5B

# 以下仅用于 safety judge（supplemental 部分），MATH 主线不需要
DEEPSEEK_API_KEY=
OPENAI_API_KEY=${DEEPSEEK_API_KEY}
OPENAI_BASE_URL=https://api.deepseek.com
CS336_ALIGNMENT_JUDGE_MODEL=deepseek-chat
```

如果留空 `CS336_ALIGNMENT_DATA_DIR`，数据将写到仓库内 `data/MATH/`；训练输出写到 `runs/`；baseline 输出写到 `out/`（硬编码，不受 `CS336_ALIGNMENT_OUTPUT_DIR` 影响）。

## 4. 数据准备

仅需一条命令，从 Berkeley 公开 URL 下载 MATH 数据集并转换为 JSONL 格式：

```bash
uv run python scripts/prepare_public_math_data.py
```

生成文件：

```
data/MATH/train.jsonl       # 约 7,500 条训练样本（包含 problem / answer / level / type）
data/MATH/validation.jsonl  # 约 5,000 条验证样本
data/MATH/sft.jsonl         # SFT 格式，包含 prompt（完整格式化） / response / ground_truth
```

下载耗时取决于网络，MATH.tar 约 120 MB。

## 5. 运行主线

### 5.1 Baseline 评测

```bash
uv run python cs336_alignment/math_baseline.py
```

默认评测全部 validation 样本。用 `--max-prompts` 快速验证：

```bash
uv run python cs336_alignment/math_baseline.py --max-prompts 128
```

输出：`out/math_baseline.jsonl`（第一行为 metrics，后续每行为单条结果）

### 5.2 SFT 训练

默认需要 2 块 GPU（`cuda:0` 训练，`cuda:1` 跑 vLLM 评测）。**单卡时脚本会自动检测并回退**：若 `cuda:1` 不存在，自动将 vLLM 切到 `cuda:0`，并将 `gpu_memory_utilization` 降至 `0.4`，无需手动配置。

```bash
uv run python cs336_alignment/sft_exp/train.py --config-name 128-examples.yaml
```

若单卡 OOM，可手动调低显存占用（新建 yaml 覆盖）：

```yaml
# cs336_alignment/sft_exp/config/my-single-gpu.yaml
training:
  gpu_memory_utilization: 0.3
  max_unique_examples: 128
```

其他可用配置（在 `cs336_alignment/sft_exp/config/` 下）：

| 配置文件 | 说明 |
|----------|------|
| `128-examples.yaml` | 仅用 128 条样本，快速实验 |
| `256-examples.yaml` | 256 条 |
| `512-examples.yaml` | 512 条 |
| `1024-examples.yaml` | 1024 条 |
| `all-examples.yaml` | 全量 |
| `all-correct-examples.yaml` | 仅正确样本 |

输出目录：`runs/sft-experiment/<config-name>/`（集群路径会自动重映射）

### 5.3 Expert Iteration 训练

默认需要 2 块 GPU，同 SFT。**单卡时同样自动回退**，无需额外配置。

```bash
uv run python cs336_alignment/expert_iteration_exp/train.py --config-name exp-iter-r5e3.yaml
```

可用配置（`cs336_alignment/expert_iteration_exp/config/`），命名规则 `exp-iter-r{rollouts}e{sft_epochs}.yaml`：

```
exp-iter-r5e3.yaml    # 5 rollouts/问题，3 SFT epochs/批次
exp-iter-r5e5.yaml
exp-iter-r10e3.yaml
exp-iter-r10e5.yaml
exp-iter-r20e3.yaml
...
```

输出目录：`runs/expert-iteration-exp/<config-name>/`

### 5.4 GRPO 训练（需 1 GPU，显存要求高）

默认配置（200 steps，完整训练）：

```bash
uv run python cs336_alignment/grpo/train.py
```

轻量验证链路（更快，输出在 `runs/a5-alignment/grpo-experiments/grpo-test/`）：

```bash
uv run python cs336_alignment/grpo/train.py --config-name test.yaml
```

其他配置在 `cs336_alignment/grpo/config/` 下（学习率 sweep、baseline 对比等），可按需传入。

输出目录：`runs/grpo-experiments/`，包含每个 step 的 rollout jsonl 和最终模型权重。

## 6. 路径解析逻辑

代码通过 `cs336_alignment/repro.py` 统一解析路径，了解这一点有助于排查问题：

- 数据路径：优先用文件实际路径；若以原集群前缀 `/data/a5-alignment/MATH/` 开头，自动替换为 `{CS336_ALIGNMENT_DATA_DIR}/MATH/`
- 输出路径：若以集群前缀 `/data/c-sniderb/a5-alignment/` 或 `/data/a5-alignment/` 开头，自动剥离前缀并替换为 `{CS336_ALIGNMENT_OUTPUT_DIR}/`
- 模型路径：若以 `/data/a5-alignment/models/` 开头，自动替换为 `{CS336_ALIGNMENT_MODEL}` 的值（即从 HuggingFace 下载）

这意味着 yaml 配置文件里的集群硬编码路径不需要手动修改，脚本启动时会自动重映射。

## 7. WandB（可选）

默认关闭。若需要 WandB 追踪，在 `.env` 中设置（或直接 `export`）：

```bash
WANDB_API_KEY=...
```

然后在代码或 yaml 中设置：

```yaml
training:
  wandb_entity: your-wandb-entity
  wandb_project: your-project-name
```

## 8. Slurm（可选，仅集群）

脚本支持通过 `--submit` 提交 Slurm 作业（需要有 `submitit` 和 Slurm 环境）：

```bash
uv run python cs336_alignment/grpo/train.py --config-name test.yaml --submit --wait
```

不传 `--submit` 时在本地前台运行，无需 Slurm。

Slurm 日志目录：`{model_output}/slurm/`

## 9. 未实现部分（Supplemental 作业）

以下文件属于 Supplemental Assignment 的作业骨架，接口有 `NotImplementedError`，**无法直接运行**：

- `tests/adapters.py` — SFT / DPO / RLHF 的作业接口
- `tests/test_data.py` — 数据加载测试
- `tests/test_dpo.py` — DPO loss 测试
- `tests/test_metrics.py` — MMLU/GSM8K 指标测试
- `tests/test_sft.py` — SFT 测试

相关的数据下载脚本（`scripts/prepare_supplemental_data.py`）和安全评测脚本（`scripts/evaluate_safety.py`）代码已完整，但因 Supplemental 训练未实现，实际上不会用到。

**结论：跑完 MATH 主线（第 5 节）即为完整复现。**
