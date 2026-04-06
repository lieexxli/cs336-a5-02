# ============================================================
#  CS 336 A5 — MATH Alignment Reproduction
#  Usage: make <target>  (run from repo root)
# ============================================================

# ---------- configurable knobs ----------
SFT_CONFIG   ?= 128-examples.yaml
EI_CONFIG    ?= exp-iter-r5e3.yaml
GRPO_CONFIG  ?= test.yaml          # swap to nothing for full 200-step run
MAX_PROMPTS  ?= 128                # baseline quick-check; remove flag for full eval

PYTHON       := uv run python
# ----------------------------------------

.PHONY: help install env data baseline sft ei grpo grpo-full check-all clean

# ---------- default ----------
help:
	@echo ""
	@echo "Targets:"
	@echo "  make install      Install dependencies (uv sync)"
	@echo "  make env          Copy .env.example -> .env  (edit before training)"
	@echo "  make data         Download & convert MATH dataset"
	@echo "  make baseline     Run MATH baseline eval  (MAX_PROMPTS=$(MAX_PROMPTS))"
	@echo "  make sft          SFT training            (SFT_CONFIG=$(SFT_CONFIG))"
	@echo "  make ei           Expert Iteration        (EI_CONFIG=$(EI_CONFIG))"
	@echo "  make grpo         GRPO quick run          (GRPO_CONFIG=$(GRPO_CONFIG))"
	@echo "  make grpo-full    GRPO full 200-step run"
	@echo "  make all          data -> baseline -> sft -> ei -> grpo-full"
	@echo "  make check-all    Verify key output files exist"
	@echo "  make clean        Remove generated data & run outputs"
	@echo ""
	@echo "Override examples:"
	@echo "  make baseline MAX_PROMPTS=512"
	@echo "  make sft SFT_CONFIG=all-examples.yaml"
	@echo "  make ei  EI_CONFIG=exp-iter-r10e5.yaml"

# ---------- install ----------
install:
	uv sync --no-install-package flash-attn
	uv sync

# ---------- env ----------
env:
	@if [ -f .env ]; then \
		echo ".env already exists, skipping copy."; \
	else \
		cp .env.example .env; \
		echo "Created .env — fill in CS336_ALIGNMENT_MODEL / DEEPSEEK_API_KEY as needed."; \
	fi

# ---------- data ----------
data: env
	$(PYTHON) scripts/prepare_public_math_data.py
	@echo "✓ MATH data ready"

# ---------- baseline ----------
baseline: data
	$(PYTHON) cs336_alignment/math_baseline.py --max-prompts $(MAX_PROMPTS)
	@echo "✓ Baseline done → out/math_baseline.jsonl"

baseline-full: data
	$(PYTHON) cs336_alignment/math_baseline.py
	@echo "✓ Full baseline done → out/math_baseline.jsonl"

# ---------- sft ----------
sft: data
	$(PYTHON) cs336_alignment/sft_exp/train.py --config-name $(SFT_CONFIG)
	@echo "✓ SFT done (config: $(SFT_CONFIG))"

# ---------- expert iteration ----------
ei: data
	$(PYTHON) cs336_alignment/expert_iteration_exp/train.py --config-name $(EI_CONFIG)
	@echo "✓ Expert Iteration done (config: $(EI_CONFIG))"

# ---------- grpo ----------
grpo: data
	$(PYTHON) cs336_alignment/grpo/train.py --config-name $(GRPO_CONFIG)
	@echo "✓ GRPO done (config: $(GRPO_CONFIG))"

grpo-full: data
	$(PYTHON) cs336_alignment/grpo/train.py
	@echo "✓ GRPO full run done → runs/grpo-experiments/"

# ---------- full pipeline ----------
all: data baseline sft ei grpo-full

# ---------- check outputs ----------
check-all:
	@echo "Checking key output files..."
	@test -f data/MATH/train.jsonl       && echo "  ✓ data/MATH/train.jsonl"       || echo "  ✗ data/MATH/train.jsonl MISSING"
	@test -f data/MATH/validation.jsonl  && echo "  ✓ data/MATH/validation.jsonl"  || echo "  ✗ data/MATH/validation.jsonl MISSING"
	@test -f data/MATH/sft.jsonl         && echo "  ✓ data/MATH/sft.jsonl"         || echo "  ✗ data/MATH/sft.jsonl MISSING"
	@test -f out/math_baseline.jsonl     && echo "  ✓ out/math_baseline.jsonl"     || echo "  ✗ out/math_baseline.jsonl MISSING (run: make baseline)"
	@test -d runs/                       && echo "  ✓ runs/ exists"                || echo "  ✗ runs/ MISSING (no training done yet)"

# ---------- clean ----------
clean:
	@echo "This will remove data/ and runs/ — are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	rm -rf data/ runs/ out/
	@echo "Cleaned."
