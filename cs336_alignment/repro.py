from __future__ import annotations

import os
from pathlib import Path

from cs336_alignment.env import load_repo_env


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"

_CLUSTER_DATA_PREFIX = "/data/a5-alignment/MATH/"
_CLUSTER_MODEL_PREFIX = "/data/a5-alignment/models/"
_CLUSTER_OUTPUT_PREFIXES = (
    "/data/c-sniderb/a5-alignment/",
    "/data/a5-alignment/",
)


def project_root() -> Path:
    return PROJECT_ROOT


def default_data_dir() -> Path:
    load_repo_env()
    override = os.environ.get("CS336_ALIGNMENT_DATA_DIR")
    return Path(override) if override else PROJECT_ROOT / "data"


def default_math_dir() -> Path:
    return default_data_dir() / "MATH"


def default_output_dir() -> Path:
    load_repo_env()
    override = os.environ.get("CS336_ALIGNMENT_OUTPUT_DIR")
    return Path(override) if override else PROJECT_ROOT / "runs"


def default_model_id() -> str:
    load_repo_env()
    return os.environ.get("CS336_ALIGNMENT_MODEL") or PUBLIC_MODEL_ID


def resolve_repo_file(path_like: str | os.PathLike) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_data_path(path_like: str | os.PathLike) -> Path:
    raw_path = str(path_like)
    path = Path(raw_path)
    if path.exists():
        return path
    if raw_path.startswith(_CLUSTER_DATA_PREFIX):
        return default_math_dir() / path.name
    return resolve_repo_file(raw_path)


def resolve_output_path(path_like: str | os.PathLike) -> Path:
    raw_path = str(path_like)
    path = Path(raw_path)
    if not raw_path:
        return default_output_dir()
    for prefix in _CLUSTER_OUTPUT_PREFIXES:
        if raw_path.startswith(prefix):
            relative = raw_path[len(prefix) :].strip("/\\")
            if not relative:
                return default_output_dir()
            return default_output_dir() / Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_model_path(model_path: str | os.PathLike) -> str:
    raw_path = str(model_path)
    path = Path(raw_path)
    if path.exists():
        return str(path)
    if raw_path.startswith(_CLUSTER_MODEL_PREFIX):
        return default_model_id()
    return raw_path


def default_submitit_dir(model_output: str | os.PathLike) -> Path:
    return resolve_output_path(model_output) / "slurm"
