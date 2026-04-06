#!/usr/bin/env python
from __future__ import annotations

import os
import subprocess
import sys

from cs336_alignment.env import load_repo_env


def main() -> int:
    load_repo_env()
    command = ["alpaca_eval", *sys.argv[1:]]
    completed = subprocess.run(command, env=os.environ.copy(), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
