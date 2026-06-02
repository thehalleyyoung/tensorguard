"""Step 66 — production GitHub Action: run TensorGuard on a PR and annotate the diff.

GitHub renders *workflow commands* of the form
``::error file=PATH,line=L,col=C,title=T::MESSAGE`` as inline annotations on the
exact line of a pull-request diff.  This module turns a verification result into
those commands and provides a small env-driven entry point that the composite
``action.yml`` invokes.

Everything except :func:`main` is pure and unit-tested: the escaping, the
file→annotation mapping (preferring the human-readable diagnostics, falling back
to located bugs), de-duplication, and the pass/fail summary.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LEVELS = {"error", "warning", "notice"}


def escape_data(message: str) -> str:
    """Escape the *data* portion of a workflow command (after ``::``)."""
    return (
        (message or "")
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def escape_property(value: str) -> str:
    """Escape a property *value* (e.g. ``title=``) of a workflow command."""
    return (
        escape_data(value)
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def format_annotation(
    file: str,
    line: int,
    message: str,
    col: Optional[int] = None,
    level: str = "error",
    title: Optional[str] = None,
) -> str:
    """Render a single GitHub annotation workflow command."""
    if level not in _LEVELS:
        level = "error"
    props = [f"file={escape_property(file)}"]
    if line and line > 0:
        props.append(f"line={int(line)}")
    if col and col > 0:
        props.append(f"col={int(col)}")
    if title:
        props.append(f"title={escape_property(title)}")
    return f"::{level}::" if not props else (
        f"::{level} {','.join(props)}::{escape_data(message)}"
    )


@dataclass
class Annotation:
    file: str
    line: int
    col: Optional[int]
    message: str
    level: str = "error"
    title: str = "TensorGuard"

    def render(self) -> str:
        return format_annotation(
            self.file, self.line, self.message, self.col, self.level, self.title
        )


def _diag_annotations(file: str, result: Any) -> List[Annotation]:
    out: List[Annotation] = []
    for d in getattr(result, "diagnostics", None) or []:
        line = getattr(d, "source_line", None)
        if not line or line <= 0:
            continue
        msg = getattr(d, "message", "") or ""
        sev = getattr(d, "severity", "error") or "error"
        level = sev if sev in _LEVELS else "error"
        out.append(
            Annotation(file, int(line), getattr(d, "source_col", None), msg, level)
        )
    return out


def _bug_annotations(file: str, result: Any) -> List[Annotation]:
    out: List[Annotation] = []
    for b in getattr(result, "bugs", None) or []:
        loc = getattr(b, "location", None)
        line = getattr(loc, "line", 0) if loc else 0
        if not line or line <= 0:
            continue
        col = getattr(loc, "column", None) if loc else None
        msg = (getattr(b, "message", "") or "").splitlines()[0]
        sev = getattr(b, "severity", "error") or "error"
        level = sev if sev in _LEVELS else "error"
        out.append(Annotation(file, int(line), col, msg, level))
    return out


def annotations_for_result(file: str, result: Any) -> List[Annotation]:
    """Annotations for one file, preferring diagnostics, de-duplicated.

    Diagnostics are the curated, human-readable one-liners; only if a result has
    none (older path) do we fall back to located bugs.  Identical
    ``(line, col, message)`` triples are collapsed.
    """
    anns = _diag_annotations(file, result) or _bug_annotations(file, result)
    seen: set = set()
    unique: List[Annotation] = []
    for a in anns:
        key = (a.line, a.col, a.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    return unique


@dataclass
class ActionResult:
    annotations: List[Annotation]
    files_checked: int
    files_with_issues: int
    total_issues: int
    failed: bool
    results_by_file: List[Tuple[str, Any]] = field(default_factory=list)

    def render_annotations(self) -> str:
        return "\n".join(a.render() for a in self.annotations)

    def summary_markdown(self) -> str:
        if self.total_issues == 0:
            return (
                f"### TensorGuard\n\n"
                f"Verified {self.files_checked} file(s); no issues found.\n"
            )
        return (
            f"### TensorGuard\n\n"
            f"Found {self.total_issues} issue(s) across "
            f"{self.files_with_issues} of {self.files_checked} file(s).\n"
        )


def _iter_python_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            files.extend(str(f) for f in sorted(path.rglob("*.py")))
        elif path.is_file():
            files.append(str(path))
    # stable de-dup
    seen: set = set()
    out: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _git_changed_python_files(
    base_ref: str = "",
    head_ref: str = "",
    *,
    repo_root: Optional[str] = None,
    runner=None,
) -> Optional[List[str]]:
    """Return existing changed ``*.py`` files between *base_ref* and *head_ref*.

    Uses ``git diff --name-only --diff-filter=ACMRT base...head`` (added / copied /
    modified / renamed / type-changed — deletions are intentionally excluded, a
    deleted file has nothing left to verify). Paths are resolved against the repo
    root and filtered to files that still exist and end in ``.py``.

    Returns ``None`` (signalling "fall back to a full scan") when git is
    unavailable, the base ref cannot be resolved (shallow clone, fork, default
    branch is not ``main``), or the diff command fails — so the Action degrades
    gracefully rather than verifying the wrong set on common PR configurations.
    ``runner`` is injectable for testing (defaults to ``subprocess.run``).
    """
    import subprocess

    if runner is None:
        runner = subprocess.run
    root = repo_root or os.getcwd()

    def _git(args: List[str]):
        try:
            return runner(
                ["git", "-C", root, *args],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            return None

    base = base_ref or os.environ.get("INPUT_BASE_REF") or ""
    head = head_ref or os.environ.get("INPUT_HEAD_REF") or "HEAD"
    if not base:
        # Try the usual GitHub PR / push event refs, then origin's default branch.
        for cand in (
            os.environ.get("GITHUB_BASE_REF"),
            "origin/main",
            "origin/master",
            "main",
            "master",
        ):
            if not cand:
                continue
            probe = _git(["rev-parse", "--verify", "--quiet", cand])
            if probe is not None and probe.returncode == 0:
                base = cand.strip()
                break
    if not base:
        return None
    # Verify both endpoints resolve before diffing (avoids a misleading error).
    for ref in (base, head):
        probe = _git(["rev-parse", "--verify", "--quiet", ref])
        if probe is None or probe.returncode != 0:
            return None

    proc = _git(
        ["diff", "--name-only", "--diff-filter=ACMRT", f"{base}...{head}"]
    )
    if proc is None or proc.returncode != 0:
        return None
    out: List[str] = []
    seen: set = set()
    for rel in proc.stdout.splitlines():
        rel = rel.strip()
        if not rel.endswith(".py"):
            continue
        abs_path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if abs_path in seen or not os.path.isfile(abs_path):
            continue
        seen.add(abs_path)
        out.append(abs_path)
    return out


def resolve_changed_paths(
    paths: List[str],
    *,
    changed_only: bool,
    base_ref: str = "",
    head_ref: str = "",
    repo_root: Optional[str] = None,
    runner=None,
) -> Tuple[List[str], bool]:
    """Resolve the file set to verify, honouring *changed_only*.

    Returns ``(resolved_paths, used_changed)``. When ``changed_only`` is set and a
    git diff is obtainable, ``resolved_paths`` is the changed ``*.py`` files that
    also lie under the requested *paths*; otherwise it falls back to *paths*
    unchanged (``used_changed`` is ``False``).
    """
    if not changed_only:
        return paths, False
    changed = _git_changed_python_files(
        base_ref, head_ref, repo_root=repo_root, runner=runner
    )
    if changed is None:
        return paths, False
    # Keep only changed files inside the requested scope (so `paths: src` does not
    # suddenly verify changed files elsewhere in the repo).
    roots = [os.path.abspath(p) for p in paths]

    def _in_scope(f: str) -> bool:
        af = os.path.abspath(f)
        for r in roots:
            if af == r or af.startswith(os.path.join(r, "")):
                return True
            if os.path.isfile(r) and af == r:
                return True
        return False

    scoped = [f for f in changed if _in_scope(f)] if paths not in ([], ["."]) else changed
    return scoped, True


def run_action(
    paths: List[str],
    *,
    soundness_mode: str = "balanced",
    input_shapes: Optional[Dict[str, Tuple]] = None,
    fail_on: str = "any",
    verify_fn=None,
    config_fn=None,
    baseline: Optional[set] = None,
    inline_suppression: bool = True,
    fingerprint_root: Optional[str] = None,
    changed_only: bool = False,
    base_ref: str = "",
    head_ref: str = "",
    repo_root: Optional[str] = None,
    git_runner=None,
) -> ActionResult:
    """Verify every ``*.py`` under *paths* and collect PR annotations.

    ``fail_on`` is ``"any"`` (fail if any issue) or ``"never"`` (annotate only).
    ``verify_fn``/``config_fn`` are injectable for testing; by default they are
    the real :func:`src.api.verify_architecture` and the Step-64 config loader.

    ``inline_suppression`` honours ``# tensorguard: ignore`` comments (Step 72);
    ``baseline`` (a set of fingerprints, or a path to a ``.tensorguard-baseline``
    file) suppresses findings already recorded so only *new* findings can fail.

    ``changed_only`` (Step 162) restricts verification to the ``*.py`` files
    changed between ``base_ref`` and ``head_ref`` (``git diff``), so a PR only
    pays for the models it touched; it degrades to a full scan when the diff is
    unobtainable (shallow clone, missing base ref, no git).
    """
    if verify_fn is None:
        from src.api import verify_architecture as verify_fn  # noqa: N806
    if config_fn is None:
        from src.tg_config import load_tg_config, filter_result, is_ignored_file

        def config_fn(file_path):  # noqa: ANN001
            cfg = load_tg_config(file_path)
            return cfg, filter_result, is_ignored_file

    all_anns: List[Annotation] = []
    files_with_issues = 0
    checked = 0
    results_by_file: List[Tuple[str, Any]] = []
    scan_paths, _used_changed = resolve_changed_paths(
        paths,
        changed_only=changed_only,
        base_ref=base_ref,
        head_ref=head_ref,
        repo_root=repo_root,
        runner=git_runner,
    )
    for file in _iter_python_files(scan_paths):
        try:
            source = pathlib.Path(file).read_text(encoding="utf-8")
        except Exception:
            continue
        cfg, filter_result, is_ignored_file = config_fn(file)
        if is_ignored_file(cfg, file):
            continue
        checked += 1
        result = verify_fn(
            source,
            input_shapes=input_shapes,
            filename=file,
            soundness_mode=getattr(cfg, "soundness_mode", None) or soundness_mode,
        )
        result = filter_result(cfg, result)
        results_by_file.append((file, result))
        anns = annotations_for_result(file, result)
        if inline_suppression and anns:
            from src.baseline import filter_inline

            anns, _suppressed = filter_inline(anns, source)
        if anns:
            files_with_issues += 1
            all_anns.extend(anns)

    total = len(all_anns)
    failed = bool(total) and fail_on != "never"
    action = ActionResult(
        all_anns, checked, files_with_issues, total, failed, results_by_file
    )

    if baseline is not None:
        from src.baseline import apply_baseline, load_baseline_fingerprints

        fps = (
            load_baseline_fingerprints(baseline)
            if isinstance(baseline, str)
            else set(baseline)
        )
        action = apply_baseline(action, fps, root=fingerprint_root)
    return action


def _parse_shapes(spec: str) -> Dict[str, Tuple]:
    shapes: Dict[str, Tuple] = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, dims = part.split("=", 1)
        parsed: List[Any] = []
        for d in dims.split(","):
            d = d.strip()
            try:
                parsed.append(int(d))
            except ValueError:
                parsed.append(d)
        shapes[name.strip()] = tuple(parsed)
    return shapes


def main(argv: Optional[List[str]] = None) -> int:
    """Env-driven entry point invoked by the composite ``action.yml``.

    Reads ``INPUT_PATHS``, ``INPUT_SOUNDNESS_MODE``, ``INPUT_INPUT_SHAPES`` and
    ``INPUT_FAIL_ON`` (GitHub's ``with:`` inputs), prints annotations to stdout,
    writes ``GITHUB_OUTPUT`` / ``GITHUB_STEP_SUMMARY`` when present, and returns
    a non-zero exit code when the gate fails.
    """
    paths = (os.environ.get("INPUT_PATHS") or ".").split()
    soundness = os.environ.get("INPUT_SOUNDNESS_MODE") or "balanced"
    shapes = _parse_shapes(os.environ.get("INPUT_INPUT_SHAPES", ""))
    fail_on = os.environ.get("INPUT_FAIL_ON") or "any"

    # Step 72: discover a baseline file (explicit input or nearest ancestor) so
    # only new findings can fail the gate on a legacy repo.
    baseline_path = os.environ.get("INPUT_BASELINE")
    if not baseline_path:
        from src.baseline import find_baseline_file

        baseline_path = find_baseline_file(paths[0] if paths else ".")
    root = os.getcwd()

    changed_only = (os.environ.get("INPUT_CHANGED_ONLY") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    base_ref = os.environ.get("INPUT_BASE_REF") or ""
    head_ref = os.environ.get("INPUT_HEAD_REF") or ""

    result = run_action(
        paths,
        soundness_mode=soundness,
        input_shapes=shapes or None,
        fail_on=fail_on,
        baseline=baseline_path,
        fingerprint_root=root,
        changed_only=changed_only,
        base_ref=base_ref,
        head_ref=head_ref,
        repo_root=root,
    )

    rendered = result.render_annotations()
    if rendered:
        print(rendered)

    sarif_path = os.environ.get("INPUT_SARIF_OUTPUT") or os.environ.get("INPUT_SARIF")
    if sarif_path:
        from src.sarif_codescan import write_sarif

        write_sarif(sarif_path, result.results_by_file)

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"issues={result.total_issues}\n")
            fh.write(f"files-with-issues={result.files_with_issues}\n")
            fh.write(f"files-checked={result.files_checked}\n")

    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write(result.summary_markdown())

    return 1 if result.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
