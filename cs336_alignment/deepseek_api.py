from __future__ import annotations

import os

from openai import OpenAI

from cs336_alignment.env import load_repo_env


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def default_deepseek_base_url() -> str:
    load_repo_env()
    return (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_DEEPSEEK_BASE_URL
    )


def default_deepseek_model() -> str:
    load_repo_env()
    return (
        os.environ.get("CS336_ALIGNMENT_JUDGE_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or DEFAULT_DEEPSEEK_MODEL
    )


def get_deepseek_api_key() -> str | None:
    load_repo_env()
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")


def make_deepseek_client(
    api_key: str | None = None, base_url: str | None = None
) -> OpenAI:
    resolved_api_key = api_key or get_deepseek_api_key()
    if not resolved_api_key:
        raise RuntimeError(
            "Missing DeepSeek API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
        )

    return OpenAI(
        api_key=resolved_api_key,
        base_url=base_url or default_deepseek_base_url(),
    )
