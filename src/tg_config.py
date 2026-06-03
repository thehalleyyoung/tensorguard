"""Per-repository configuration for the architecture verifier (`tensorguard.toml`).

Step 64.  A repository can drop a ``tensorguard.toml`` (or a ``[tool.tensorguard]``
table in ``pyproject.toml``) to set project-wide defaults for the ``verify``
command: which checks run, the soundness mode, files to ignore, and specific
rule kinds to suppress.  Configuration is discovered by walking up from the file
under analysis, so a single repo-root config governs the whole tree.

Everything here is pure and dependency-light (stdlib ``tomllib``), so the
loader, the precedence rules, and the ignore filters are fully unit-testable.

Schema (all keys optional)::

    [tensorguard]
    soundness_mode   = "sound"            # sound | balanced | heuristic
    infer_inputs     = true
    high_confidence  = false
    cegar_iterations = 12
    max_loop_unrolls = 3
    ignore           = ["experiments/**", "legacy/old.py"]
    ignore_rules     = ["cegar-real-bug"]  # bug [KIND] tags, case-insensitive

    [tensorguard.checks]
    devices   = true
    phases    = false
    gradients = true
"""

from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

CONFIG_FILENAME = "tensorguard.toml"
_VALID_MODES = {"sound", "balanced", "heuristic"}
_TAG_RE = re.compile(r"^\s*\[([A-Z0-9_\-]+)\]")


@dataclass
class TGConfig:
    """Resolved verifier configuration (None = "not specified")."""

    soundness_mode: Optional[str] = None
    check_devices: bool = True
    check_phases: bool = True
    check_gradients: bool = True
    infer_inputs: bool = True
    high_confidence: bool = False
    cegar_iterations: Optional[int] = None
    max_loop_unrolls: Optional[int] = None
    ignore_files: List[str] = field(default_factory=list)
    ignore_rules: List[str] = field(default_factory=list)
    source_path: Optional[str] = None  # where it was loaded from
    project_root: Optional[str] = None  # dir containing the config


def find_config_file(start: Path) -> Optional[Path]:
    """Walk up from *start* looking for tensorguard.toml or pyproject.toml.

    A ``tensorguard.toml`` wins over a ``pyproject.toml`` in the same directory.
    A ``pyproject.toml`` is only considered a hit if it has a ``[tool.tensorguard]``
    table.
    """
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for directory in [start, *start.parents]:
        cand = directory / CONFIG_FILENAME
        if cand.is_file():
            return cand
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            try:
                with open(pyproject, "rb") as fh:
                    data = tomllib.load(fh)
                if isinstance(data.get("tool"), dict) and isinstance(
                    data["tool"].get("tensorguard"), dict
                ):
                    return pyproject
            except Exception:
                pass
    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    return bool(value) if isinstance(value, bool) else default


def parse_config(data: dict, source_path: Optional[str] = None) -> TGConfig:
    """Turn a parsed TOML mapping into a :class:`TGConfig`.

    Accepts either a top-level ``[tensorguard]`` table (standalone file) or a
    ``[tool.tensorguard]`` table (pyproject); the caller passes the already
    selected sub-table here.
    """
    cfg = TGConfig(source_path=source_path)
    if source_path:
        cfg.project_root = str(Path(source_path).resolve().parent)

    mode = data.get("soundness_mode")
    if isinstance(mode, str) and mode in _VALID_MODES:
        cfg.soundness_mode = mode

    cfg.infer_inputs = _coerce_bool(data.get("infer_inputs"), True)
    cfg.high_confidence = _coerce_bool(data.get("high_confidence"), False)

    ci = data.get("cegar_iterations")
    if isinstance(ci, int) and ci > 0:
        cfg.cegar_iterations = ci

    loop_unrolls = data.get("max_loop_unrolls")
    if isinstance(loop_unrolls, int) and loop_unrolls >= 0:
        cfg.max_loop_unrolls = loop_unrolls

    checks = data.get("checks")
    if isinstance(checks, dict):
        cfg.check_devices = _coerce_bool(checks.get("devices"), True)
        cfg.check_phases = _coerce_bool(checks.get("phases"), True)
        cfg.check_gradients = _coerce_bool(checks.get("gradients"), True)

    ignore = data.get("ignore")
    if isinstance(ignore, list):
        cfg.ignore_files = [str(p) for p in ignore if isinstance(p, str)]

    rules = data.get("ignore_rules")
    if isinstance(rules, list):
        cfg.ignore_rules = [
            str(r).strip().lower() for r in rules if isinstance(r, str)
        ]

    return cfg


def load_tg_config(
    start_dir: Any, explicit_path: Optional[str] = None
) -> TGConfig:
    """Load configuration for a file/dir, or an explicit path.

    Returns an empty (all-default) :class:`TGConfig` when nothing is found, so
    callers can always apply it unconditionally.
    """
    path: Optional[Path] = None
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            path = p
    else:
        path = find_config_file(Path(start_dir))
    if path is None:
        return TGConfig()

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return TGConfig()

    if path.name == "pyproject.toml":
        table = data.get("tool", {}).get("tensorguard", {})
    else:
        # Standalone file: accept either a [tensorguard] table or top level.
        table = data.get("tensorguard", data)
    if not isinstance(table, dict):
        return TGConfig(source_path=str(path))
    return parse_config(table, source_path=str(path))


def is_ignored_file(cfg: TGConfig, file_path: str) -> bool:
    """True if *file_path* matches an ``ignore`` glob pattern.

    Patterns are matched against the path relative to the config's project root
    (and also against the basename), using shell-style globbing where ``**``
    matches across directories.
    """
    if not cfg.ignore_files:
        return False
    p = Path(file_path).resolve()
    candidates = [str(p), p.name]
    if cfg.project_root:
        try:
            rel = os.path.relpath(str(p), cfg.project_root)
            candidates.append(rel)
            candidates.append(rel.replace(os.sep, "/"))
        except ValueError:
            pass
    for pattern in cfg.ignore_files:
        for cand in candidates:
            if fnmatch.fnmatch(cand, pattern):
                return True
            # Make "**" behave intuitively for fnmatch (which treats * greedily
            # but stops at nothing): also try the pattern with "**" -> "*".
            if "**" in pattern and fnmatch.fnmatch(
                cand, pattern.replace("**", "*")
            ):
                return True
    return False


def rule_tag(message: str) -> str:
    """Extract the lowercased ``[KIND]`` tag from a bug message, or ''."""
    m = _TAG_RE.match(message or "")
    return m.group(1).lower() if m else ""


def is_ignored_rule(cfg: TGConfig, message: str) -> bool:
    """True if a bug message's ``[KIND]`` tag is in ``ignore_rules``."""
    if not cfg.ignore_rules:
        return False
    return rule_tag(message) in cfg.ignore_rules


def filter_result(cfg: TGConfig, result: Any) -> Any:
    """Drop bugs (and their aligned diagnostics) whose rule is ignored.

    Mutates and returns *result*.  Diagnostics are kept only if a surviving bug
    still occupies their source line, mirroring the way diagnostics are aligned
    to bugs upstream.
    """
    if not cfg.ignore_rules:
        return result
    bugs = list(getattr(result, "bugs", None) or [])
    kept = [b for b in bugs if not is_ignored_rule(cfg, getattr(b, "message", ""))]
    result.bugs = kept
    surviving_lines = {
        getattr(getattr(b, "location", None), "line", None) for b in kept
    }
    diagnostics = list(getattr(result, "diagnostics", None) or [])
    if diagnostics:
        result.diagnostics = [
            d for d in diagnostics
            if getattr(d, "source_line", None) in surviving_lines
        ]
    return result
