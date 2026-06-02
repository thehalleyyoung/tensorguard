"""Step 162 — GitHub Action verifies only the models *changed* in a PR.

Builds a throwaway git repository with a clean base commit, then a feature
branch that (a) adds a buggy model, (b) modifies a previously-clean model into a
buggy one, (c) leaves an untouched buggy model, (d) deletes a file, and (e) adds
a non-Python file. ``changed_only`` must verify exactly the changed *.py files
and ignore the rest, while a full scan still sees everything — proving the diff
restriction is real, not cosmetic.
"""

from __future__ import annotations

import os
import subprocess

import pytest

torch = pytest.importorskip("torch")  # noqa: F401

from src.github_action import (
    _git_changed_python_files,
    resolve_changed_paths,
    run_action,
)

_BUGGY = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.a = nn.Linear(10, 20)\n"
    "        self.b = nn.Linear(30, 5)\n"  # 20 != 30 -> shape bug
    "    def forward(self, x):\n"
    "        return self.b(self.a(x))\n"
)
_CLEAN = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.a = nn.Linear(10, 20)\n"
    "        self.b = nn.Linear(20, 5)\n"
    "    def forward(self, x):\n"
    "        return self.b(self.a(x))\n"
)
_SHAPES = {"x": ("bb", 10)}


def _git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True,
                   capture_output=True, text=True)


def _make_repo(tmp_path):
    repo = str(tmp_path / "proj")
    os.makedirs(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    # Base commit: one clean model, one untouched buggy model, one to-be-deleted.
    (tmp_path / "proj" / "clean_then_buggy.py").write_text(_CLEAN)
    (tmp_path / "proj" / "untouched_buggy.py").write_text(_BUGGY)
    (tmp_path / "proj" / "to_delete.py").write_text(_CLEAN)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # Feature branch: add a new buggy model, turn the clean one buggy, delete a
    # file, add a non-Python file. `untouched_buggy.py` is left as-is.
    (tmp_path / "proj" / "new_buggy.py").write_text(_BUGGY)
    (tmp_path / "proj" / "clean_then_buggy.py").write_text(_BUGGY)
    (tmp_path / "proj" / "notes.txt").write_text("not python\n")
    os.remove(tmp_path / "proj" / "to_delete.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature")
    return repo, base_sha


def test_diff_lists_only_changed_python_files(tmp_path):
    repo, base = _make_repo(tmp_path)
    changed = _git_changed_python_files(base, "HEAD", repo_root=repo)
    names = sorted(os.path.basename(p) for p in changed)
    assert names == ["clean_then_buggy.py", "new_buggy.py"], names
    # Deletions and non-Python files are excluded; untouched files are excluded.
    assert "to_delete.py" not in names
    assert "untouched_buggy.py" not in names
    assert "notes.txt" not in names


def test_run_action_changed_only_verifies_just_changed_models(tmp_path):
    repo, base = _make_repo(tmp_path)
    res = run_action(
        [repo], input_shapes=_SHAPES, changed_only=True,
        base_ref=base, head_ref="HEAD", repo_root=repo,
    )
    # Exactly the two changed buggy files are checked (not the untouched buggy one).
    assert res.files_checked == 2, res.files_checked
    assert res.files_with_issues == 2
    assert res.failed
    checked_files = {os.path.basename(f) for f, _ in res.results_by_file}
    assert checked_files == {"new_buggy.py", "clean_then_buggy.py"}, checked_files

    # A full scan, by contrast, also sees the untouched buggy model.
    full = run_action([repo], input_shapes=_SHAPES, changed_only=False)
    assert full.files_checked >= 3
    full_files = {os.path.basename(f) for f, _ in full.results_by_file}
    assert "untouched_buggy.py" in full_files


def test_missing_base_ref_falls_back_to_full_scan(tmp_path):
    repo, _ = _make_repo(tmp_path)
    # An unresolvable base ref must degrade to a full scan, not error out.
    paths, used = resolve_changed_paths(
        [repo], changed_only=True, base_ref="origin/does-not-exist",
        head_ref="HEAD", repo_root=repo,
    )
    assert used is False
    assert paths == [repo]


def test_no_git_repo_falls_back(tmp_path):
    plain = str(tmp_path / "plain")
    os.makedirs(plain)
    (tmp_path / "plain" / "m.py").write_text(_BUGGY)
    paths, used = resolve_changed_paths(
        [plain], changed_only=True, repo_root=plain
    )
    assert used is False
    assert paths == [plain]
