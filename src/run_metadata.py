"""Collect GitHub push and Git metadata for MLflow run tags."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MAX_TAG_LENGTH = 5000


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _has_commit(commit: str, cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def collect_run_metadata(cwd: Path | None = None) -> dict[str, str]:
    """Return MLflow tags describing the GitHub push that started this run."""
    if not os.getenv("GITHUB_ACTIONS"):
        return {}

    repository = cwd or Path.cwd()
    commit_sha = os.getenv("GITHUB_SHA") or _git("rev-parse", "HEAD", cwd=repository)
    branch = os.getenv("GITHUB_REF_NAME", "")
    actor = os.getenv("GITHUB_ACTOR", "")
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    commit_message = _git("log", "-1", "--format=%B", commit_sha, cwd=repository)
    commit_author = _git("show", "-s", "--format=%an", commit_sha, cwd=repository)
    commit_time = _git("show", "-s", "--format=%cI", commit_sha, cwd=repository)

    tags = {
        "git.commit": commit_sha,
        "git.branch": branch,
        "git.commit_author": commit_author,
        "git.commit_message": commit_message,
        "git.committed_at": commit_time,
        "github.actor": actor,
        "github.event": event_name,
        "ci.started_at": datetime.now(timezone.utc).isoformat(),
    }

    event_path = os.getenv("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).is_file():
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        before = event.get("before")
        if before and set(before) != {"0"} and _has_commit(before, repository):
            tags["git.changed_files"] = _git(
                "diff", "--name-only", before, commit_sha, cwd=repository
            )
            messages = _git(
                "log", "--format=%s", f"{before}..{commit_sha}", cwd=repository
            )
            if messages:
                tags["git.push_commit_messages"] = messages

    return {
        key: value[:MAX_TAG_LENGTH]
        for key, value in tags.items()
        if value
    }
