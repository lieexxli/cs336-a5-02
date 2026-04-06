from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_env_path() -> Path:
    return PROJECT_ROOT / ".env"


def load_repo_env(env_path: str | Path | None = None) -> bool:
    path = Path(env_path) if env_path is not None else default_env_path()
    if not path.exists():
        return False

    load_dotenv(path, override=False)
    return True
