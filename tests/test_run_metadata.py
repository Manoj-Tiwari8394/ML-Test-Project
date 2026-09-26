import json
import subprocess
from pathlib import Path

from src.run_metadata import collect_run_metadata


def test_collect_run_metadata_includes_actor_commit_reason_and_changed_files(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "config", "user.name", "Test Author"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True
    )
    tracked_file = repository / "src" / "predict.py"
    tracked_file.parent.mkdir()
    tracked_file.write_text("initial code\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "Initial project"], cwd=repository, check=True)
    before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_file.write_text("improved code\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "Improve prediction\n\nReason: validate deployment input"],
        cwd=repository,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"before": before}), encoding="utf-8")

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_SHA", head)
    monkeypatch.setenv("GITHUB_REF_NAME", "master")
    monkeypatch.setenv("GITHUB_ACTOR", "learning-user")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    tags = collect_run_metadata(repository)

    assert tags["git.commit"] == head
    assert tags["git.branch"] == "master"
    assert tags["github.actor"] == "learning-user"
    assert tags["github.event"] == "push"
    assert tags["git.commit_message"]
    assert tags["git.committed_at"]
    assert tags["ci.started_at"]
    assert tags["git.changed_files"] == "src/predict.py"
    assert tags["git.push_commit_messages"] == "Improve prediction"


def test_collect_run_metadata_is_empty_outside_github_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    assert collect_run_metadata() == {}
