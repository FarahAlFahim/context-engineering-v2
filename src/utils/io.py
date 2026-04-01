"""I/O utilities: file operations, JSON helpers, subprocess wrapper."""

import json
import logging
import os
import shutil
import subprocess
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("context_engineering.utils.io")


def mkdirp(p: str):
    """Create directory (and parents) if it doesn't exist."""
    if p:
        os.makedirs(p, exist_ok=True)


def save_json_atomic(obj: Any, path: str):
    """Write JSON atomically (write to .tmp then rename)."""
    mkdirp(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json_safe(path: str) -> list:
    """Load a JSON file; return [] on missing/corrupt file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            if os.path.getsize(path) == 0:
                return []
            return json.load(f)
    except Exception:
        try:
            shutil.move(path, path + ".corrupt")
            logger.warning(f"Moved corrupt file to {path}.corrupt")
        except Exception:
            pass
        return []


def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 300) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return 1, "", str(e)
