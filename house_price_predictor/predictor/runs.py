"""Storage of analysis runs: one folder per upload under MEDIA_ROOT/runs/<id>/."""
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from django.conf import settings

_ID = re.compile(r"[0-9a-f]{32}")
MAX_AGE_SECONDS = 24 * 3600


def runs_root() -> Path:
    root = Path(settings.MEDIA_ROOT) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_run():
    run_id = uuid.uuid4().hex
    path = runs_root() / run_id
    path.mkdir(parents=True)
    return run_id, path


def run_path(run_id):
    """Return the folder of a run, or None when the id is invalid / unknown."""
    if not run_id or not _ID.fullmatch(str(run_id)):
        return None
    path = runs_root() / run_id
    return path if (path / "results.json").exists() else None


def load_results(path: Path):
    with open(path / "results.json", encoding="utf-8") as fh:
        return json.load(fh)


def cleanup():
    """Delete analyses older than 24 hours."""
    now = time.time()
    for child in runs_root().iterdir():
        try:
            if child.is_dir() and now - child.stat().st_mtime > MAX_AGE_SECONDS:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass
