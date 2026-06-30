from __future__ import annotations

"""
reftype.cli.main — Main CLI entry point and application logic.

Provides the ``ReftypeCliApp`` class with argparse-based subcommands for
analysing Python / TypeScript code via CEGAR-based refinement type inference.
"""

import argparse
import ast
import contextlib
import enum
import fnmatch
import glob as _glob
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    Union,
)

# ---------------------------------------------------------------------------
# Locally-defined domain types (no imports from other project modules)
# ---------------------------------------------------------------------------

logger = logging.getLogger("reftype.cli")

_VERSION = "0.1.0"


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Language(enum.Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    AUTO = "auto"


class OutputFormat(enum.Enum):
    PYI = "pyi"
    DTS = "dts"
    SARIF = "sarif"
    HTML = "html"
    JSON = "json"


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}:{self.column}"
        if self.end_line is not None:
            loc += f"-{self.end_line}:{self.end_column}"
        return loc


@dataclass
class RefinementType:
    base: str
    predicate: str
    constraints: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.predicate:
            return f"{{{self.base} | {self.predicate}}}"
        return self.base


@dataclass
class Bug:
    id: str
    message: str
    severity: Severity
    location: SourceLocation
    category: str
    refinement_type: Optional[RefinementType] = None
    fix_suggestion: Optional[str] = None
    cegar_trace: Optional[List[str]] = None

    def fingerprint(self) -> str:
        raw = f"{self.category}:{self.location.file}:{self.message}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class FunctionContract:
    name: str
    file: str
    line: int
    params: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: Optional[RefinementType] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    file: str
    language: Language
    bugs: List[Bug] = field(default_factory=list)
    contracts: List[FunctionContract] = field(default_factory=list)
    duration_ms: float = 0.0
    functions_analyzed: int = 0
    cegar_iterations: int = 0
    timed_out: bool = False


@dataclass
class AnalysisSummary:
    total_files: int = 0
    total_functions: int = 0
    total_bugs: int = 0
    bugs_by_severity: Dict[str, int] = field(default_factory=dict)
    bugs_by_category: Dict[str, int] = field(default_factory=dict)
    total_contracts: int = 0
    total_cegar_iterations: int = 0
    duration_ms: float = 0.0
    files_timed_out: int = 0

    def merge(self, result: AnalysisResult) -> None:
        self.total_files += 1
        self.total_functions += result.functions_analyzed
        self.total_bugs += len(result.bugs)
        for b in result.bugs:
            sev = b.severity.value
            self.bugs_by_severity[sev] = self.bugs_by_severity.get(sev, 0) + 1
            self.bugs_by_category[b.category] = (
                self.bugs_by_category.get(b.category, 0) + 1
            )
        self.total_contracts += len(result.contracts)
        self.total_cegar_iterations += result.cegar_iterations
        self.duration_ms += result.duration_ms
        if result.timed_out:
            self.files_timed_out += 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DomainSettings:
    interval_precision: int = 64
    octagon_enabled: bool = False
    polyhedra_enabled: bool = False
    widening_delay: int = 3
    narrowing_iterations: int = 2


@dataclass
class CegarSettings:
    max_iterations: int = 50
    refinement_strategy: str = "counterexample-guided"
    interpolation_enabled: bool = True
    predicate_abstraction: bool = True
    lazy_abstraction: bool = False


@dataclass
class IncrementalSettings:
    enabled: bool = True
    cache_dir: str = ".reftype-cache"
    hash_algorithm: str = "sha256"
    max_cache_age_hours: int = 168


@dataclass
class ParallelSettings:
    workers: int = 0  # 0 = auto (cpu_count)
    chunk_size: int = 4
    timeout_per_file: float = 60.0


@dataclass
class Configuration:
    """Complete configuration for a reftype analysis run."""

    paths: List[str] = field(default_factory=lambda: ["."])
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py", "**/*.ts"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "node_modules/**",
            ".venv/**",
            "__pycache__/**",
            "*.egg-info/**",
            "dist/**",
            "build/**",
        ]
    )
    language: Language = Language.AUTO
    output_format: OutputFormat = OutputFormat.JSON
    output_file: Optional[str] = None
    verbosity: int = 0
    config_file: Optional[str] = None

    bug_classes: List[str] = field(
        default_factory=lambda: [
            "null-deref",
            "index-out-of-bounds",
            "division-by-zero",
            "type-mismatch",
            "unreachable-code",
            "unused-refinement",
        ]
    )
    min_severity: Severity = Severity.WARNING
    max_functions: int = 0  # 0 = unlimited
    timeout: float = 300.0

    domain: DomainSettings = field(default_factory=DomainSettings)
    cegar: CegarSettings = field(default_factory=CegarSettings)
    incremental: IncrementalSettings = field(default_factory=IncrementalSettings)
    parallel: ParallelSettings = field(default_factory=ParallelSettings)

    baseline_file: Optional[str] = None
    fail_on_new_bugs: bool = False
    telemetry_enabled: bool = False

    def effective_workers(self) -> int:
        if self.parallel.workers > 0:
            return self.parallel.workers
        return max(1, (os.cpu_count() or 1) - 1)


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Loads configuration from files, merges with CLI args, validates."""

    SEARCH_FILES = (".reftype.toml", "pyproject.toml", "package.json")

    def __init__(self) -> None:
        self._raw: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def find_config_file(self, start_dir: str = ".") -> Optional[str]:
        current = pathlib.Path(start_dir).resolve()
        while True:
            for name in self.SEARCH_FILES:
                candidate = current / name
                if candidate.is_file():
                    return str(candidate)
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        if path is None:
            path = self.find_config_file()
        if path is None:
            return {}
        p = pathlib.Path(path)
        if not p.exists():
            logger.warning("Config file %s does not exist", path)
            return {}
        if p.name == ".reftype.toml" or p.suffix == ".toml":
            return self._load_toml(p)
        if p.name == "pyproject.toml":
            data = self._load_toml(p)
            return data.get("tool", {}).get("reftype", {})
        if p.name == "package.json":
            return self._load_package_json(p)
        return {}

    def _load_toml(self, p: pathlib.Path) -> Dict[str, Any]:
        try:
            import tomllib  # 3.11+
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                return self._load_toml_fallback(p)
        with open(p, "rb") as fh:
            return tomllib.load(fh)

    @staticmethod
    def _load_toml_fallback(p: pathlib.Path) -> Dict[str, Any]:
        """Minimal TOML key=value parser when no library is available."""
        data: Dict[str, Any] = {}
        section = data
        with open(p) as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                hdr = re.match(r"^\[(.+)]$", line)
                if hdr:
                    parts = hdr.group(1).split(".")
                    section = data
                    for part in parts:
                        section = section.setdefault(part.strip(), {})
                    continue
                m = re.match(r'^(\w+)\s*=\s*"(.*)"\s*$', line)
                if m:
                    section[m.group(1)] = m.group(2)
                    continue
                m = re.match(r"^(\w+)\s*=\s*(\d+)\s*$", line)
                if m:
                    section[m.group(1)] = int(m.group(2))
                    continue
                m = re.match(r"^(\w+)\s*=\s*(true|false)\s*$", line, re.I)
                if m:
                    section[m.group(1)] = m.group(2).lower() == "true"
        return data

    @staticmethod
    def _load_package_json(p: pathlib.Path) -> Dict[str, Any]:
        with open(p) as fh:
            data = json.load(fh)
        return data.get("reftype", {})

    # ------------------------------------------------------------------

    def merge(self, file_cfg: Dict[str, Any], cli_args: argparse.Namespace) -> Configuration:
        cfg = Configuration()
        mapping: Dict[str, str] = {
            "paths": "paths",
            "include": "include_patterns",
            "exclude": "exclude_patterns",
            "language": "language",
            "format": "output_format",
            "output": "output_file",
            "verbose": "verbosity",
            "config": "config_file",
            "bug_classes": "bug_classes",
            "min_severity": "min_severity",
            "max_functions": "max_functions",
            "timeout": "timeout",
            "baseline": "baseline_file",
            "fail_on_new_bugs": "fail_on_new_bugs",
            "workers": "parallel.workers",
            "incremental": "incremental.enabled",
        }
        for src_key, dst_key in mapping.items():
            val = file_cfg.get(src_key)
            if val is not None:
                self._set_nested(cfg, dst_key, val)

        for src_key, dst_key in mapping.items():
            val = getattr(cli_args, src_key, None)
            if val is not None:
                self._set_nested(cfg, dst_key, val)

        domain_raw = file_cfg.get("domain", {})
        if isinstance(domain_raw, dict):
            for k, v in domain_raw.items():
                if hasattr(cfg.domain, k):
                    setattr(cfg.domain, k, v)

        cegar_raw = file_cfg.get("cegar", {})
        if isinstance(cegar_raw, dict):
            for k, v in cegar_raw.items():
                if hasattr(cfg.cegar, k):
                    setattr(cfg.cegar, k, v)

        return cfg

    @staticmethod
    def _set_nested(obj: Any, key: str, value: Any) -> None:
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        attr = parts[-1]
        field_val = getattr(obj, attr, None)
        if isinstance(field_val, enum.Enum):
            if isinstance(value, str):
                value = type(field_val)(value)
        setattr(obj, attr, value)

    # ------------------------------------------------------------------

    def validate(self, cfg: Configuration) -> List[str]:
        errors: List[str] = []
        if cfg.timeout <= 0:
            errors.append("timeout must be positive")
        if cfg.max_functions < 0:
            errors.append("max_functions must be >= 0")
        if cfg.parallel.workers < 0:
            errors.append("workers must be >= 0")
        if cfg.cegar.max_iterations < 1:
            errors.append("cegar.max_iterations must be >= 1")
        return errors


# ---------------------------------------------------------------------------
# LoggingSetup
# ---------------------------------------------------------------------------


class LoggingSetup:
    """Configure logging with verbosity levels."""

    LEVELS = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }

    @classmethod
    def configure(cls, verbosity: int = 0, log_file: Optional[str] = None) -> None:
        level = cls.LEVELS.get(min(verbosity, 2), logging.DEBUG)
        handlers: List[logging.Handler] = []

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        handlers.append(stream_handler)

        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            handlers.append(fh)

        root = logging.getLogger("reftype")
        root.setLevel(level)
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in handlers:
            root.addHandler(h)


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Progress bars / status output with non-TTY fallback."""

    def __init__(self, total: int = 0, stream: TextIO = sys.stderr) -> None:
        self.total = total
        self.current = 0
        self.stream = stream
        self._is_tty = hasattr(stream, "isatty") and stream.isatty()
        self._start = time.monotonic()
        self._lock = threading.Lock()

    def update(self, n: int = 1, message: str = "") -> None:
        with self._lock:
            self.current += n
            if self._is_tty:
                self._render_bar(message)
            elif message:
                self.stream.write(f"[{self.current}/{self.total}] {message}\n")
                self.stream.flush()

    def _render_bar(self, message: str) -> None:
        width = shutil.get_terminal_size((80, 24)).columns - 30
        width = max(10, width)
        if self.total > 0:
            frac = self.current / self.total
            filled = int(width * frac)
            bar = "█" * filled + "░" * (width - filled)
            pct = f"{frac * 100:5.1f}%"
        else:
            bar = "░" * width
            pct = "  ?%"
        elapsed = time.monotonic() - self._start
        eta = ""
        if self.total > 0 and self.current > 0:
            remaining = elapsed / self.current * (self.total - self.current)
            eta = f" ETA {remaining:.0f}s"
        line = f"\r{bar} {pct} ({self.current}/{self.total}){eta}  {message[:30]}"
        self.stream.write(line)
        self.stream.flush()
        if self.current >= self.total > 0:
            self.stream.write("\n")

    def finish(self, message: str = "Done") -> None:
        with self._lock:
            elapsed = time.monotonic() - self._start
            if self._is_tty:
                self.stream.write(f"\r{'':80}\r")
            self.stream.write(f"{message} in {elapsed:.1f}s\n")
            self.stream.flush()


# ---------------------------------------------------------------------------
# ResultPrinter
# ---------------------------------------------------------------------------


_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "magenta": "\033[35m",
}


def _c(text: str, color: str, *, use_color: bool = True) -> str:
    if not use_color:
        return text
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


class ResultPrinter:
    """Formats and prints analysis results to terminal (colorised)."""

    def __init__(self, stream: TextIO = sys.stdout, color: bool = True) -> None:
        self.stream = stream
        self.color = color and hasattr(stream, "isatty") and stream.isatty()

    def print_bug(self, bug: Bug) -> None:
        sev_colors = {
            Severity.ERROR: "red",
            Severity.WARNING: "yellow",
            Severity.INFO: "cyan",
            Severity.HINT: "gray",
        }
        col = sev_colors.get(bug.severity, "reset")
        header = _c(f"[{bug.severity.value.upper()}]", col, use_color=self.color)
        loc = _c(str(bug.location), "bold", use_color=self.color)
        self.stream.write(f"{header} {loc}: {bug.message}\n")
        if bug.category:
            self.stream.write(
                f"  category: {_c(bug.category, 'magenta', use_color=self.color)}\n"
            )
        if bug.refinement_type:
            self.stream.write(f"  type: {bug.refinement_type}\n")
        if bug.fix_suggestion:
            self.stream.write(
                f"  fix: {_c(bug.fix_suggestion, 'green', use_color=self.color)}\n"
            )

    def print_contract(self, contract: FunctionContract) -> None:
        name = _c(contract.name, "bold", use_color=self.color)
        self.stream.write(f"  {name}(")
        parts = []
        for pname, ptype in contract.params.items():
            parts.append(f"{pname}: {ptype}")
        self.stream.write(", ".join(parts))
        ret = f" -> {contract.return_type}" if contract.return_type else ""
        self.stream.write(f"){ret}\n")
        for pre in contract.preconditions:
            self.stream.write(f"    requires {pre}\n")
        for post in contract.postconditions:
            self.stream.write(f"    ensures  {post}\n")

    def print_result(self, result: AnalysisResult) -> None:
        header = _c(f"── {result.file} ", "bold", use_color=self.color)
        lang = _c(f"[{result.language.value}]", "cyan", use_color=self.color)
        self.stream.write(f"\n{header}{lang}\n")
        if result.bugs:
            self.stream.write(
                _c(f"  {len(result.bugs)} bug(s) found:\n", "red", use_color=self.color)
            )
            for bug in result.bugs:
                self.print_bug(bug)
        if result.contracts:
            self.stream.write(
                _c(
                    f"  {len(result.contracts)} contract(s) inferred:\n",
                    "green",
                    use_color=self.color,
                )
            )
            for c in result.contracts:
                self.print_contract(c)
        meta = (
            f"  analyzed {result.functions_analyzed} functions, "
            f"{result.cegar_iterations} CEGAR iterations, "
            f"{result.duration_ms:.0f}ms"
        )
        if result.timed_out:
            meta += _c(" (TIMEOUT)", "yellow", use_color=self.color)
        self.stream.write(_c(meta, "gray", use_color=self.color) + "\n")

    def print_summary(self, summary: AnalysisSummary) -> None:
        self.stream.write("\n")
        self.stream.write(
            _c("═══ Analysis Summary ═══\n", "bold", use_color=self.color)
        )
        self.stream.write(f"  Files analysed : {summary.total_files}\n")
        self.stream.write(f"  Functions      : {summary.total_functions}\n")
        self.stream.write(f"  Bugs found     : {summary.total_bugs}\n")
        if summary.bugs_by_severity:
            for sev, cnt in sorted(summary.bugs_by_severity.items()):
                self.stream.write(f"    {sev:10s}: {cnt}\n")
        if summary.bugs_by_category:
            self.stream.write("  By category:\n")
            for cat, cnt in sorted(
                summary.bugs_by_category.items(), key=lambda x: -x[1]
            ):
                self.stream.write(f"    {cat:25s}: {cnt}\n")
        self.stream.write(f"  Contracts      : {summary.total_contracts}\n")
        self.stream.write(f"  CEGAR iters    : {summary.total_cegar_iterations}\n")
        self.stream.write(f"  Duration       : {summary.duration_ms:.0f}ms\n")
        if summary.files_timed_out:
            self.stream.write(
                _c(
                    f"  Timed-out files: {summary.files_timed_out}\n",
                    "yellow",
                    use_color=self.color,
                )
            )
        self.stream.write("\n")


# ---------------------------------------------------------------------------
# ErrorHandler
# ---------------------------------------------------------------------------


class ErrorHandler:
    """Graceful error handling with optional crash reports."""

    CRASH_DIR = ".reftype-crashes"

    def handle(self, exc: BaseException, context: str = "") -> int:
        if isinstance(exc, KeyboardInterrupt):
            sys.stderr.write("\nInterrupted.\n")
            return 130
        if isinstance(exc, SystemExit):
            return exc.code if isinstance(exc.code, int) else 1
        msg = f"reftype: internal error"
        if context:
            msg += f" ({context})"
        msg += f": {exc}"
        sys.stderr.write(f"{msg}\n")
        report_path = self._write_crash_report(exc, context)
        if report_path:
            sys.stderr.write(f"Crash report written to {report_path}\n")
        return 2

    def _write_crash_report(self, exc: BaseException, context: str) -> Optional[str]:
        try:
            d = pathlib.Path(self.CRASH_DIR)
            d.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = d / f"crash-{ts}-{uuid.uuid4().hex[:8]}.txt"
            with open(path, "w") as fh:
                fh.write(f"tensorguard {_VERSION} crash report\n")
                fh.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
                fh.write(f"Python: {sys.version}\n")
                fh.write(f"Platform: {platform.platform()}\n")
                fh.write(f"Context: {context}\n\n")
                fh.write("Traceback:\n")
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=fh)
            return str(path)
        except OSError:
            return None


# ---------------------------------------------------------------------------
# SignalHandler
# ---------------------------------------------------------------------------


class SignalHandler:
    """Handles SIGINT / SIGTERM gracefully."""

    def __init__(self) -> None:
        self._interrupted = threading.Event()
        self._original_sigint: Any = None
        self._original_sigterm: Any = None

    def install(self) -> None:
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def uninstall(self) -> None:
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handler(self, signum: int, frame: Any) -> None:
        self._interrupted.set()
        logger.info("Received signal %d, shutting down…", signum)

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------


class LanguageDetector:
    """Auto-detect language from file extensions, shebangs, config."""

    EXT_MAP: Dict[str, Language] = {
        ".py": Language.PYTHON,
        ".pyi": Language.PYTHON,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".mts": Language.TYPESCRIPT,
        ".cts": Language.TYPESCRIPT,
    }

    SHEBANG_PATTERNS: List[Tuple[str, Language]] = [
        ("python", Language.PYTHON),
        ("node", Language.TYPESCRIPT),
        ("ts-node", Language.TYPESCRIPT),
        ("deno", Language.TYPESCRIPT),
    ]

    def detect_file(self, path: str) -> Optional[Language]:
        ext = pathlib.Path(path).suffix.lower()
        lang = self.EXT_MAP.get(ext)
        if lang is not None:
            return lang
        return self._detect_shebang(path)

    def _detect_shebang(self, path: str) -> Optional[Language]:
        try:
            with open(path, "r", errors="ignore") as fh:
                first = fh.readline(256)
        except OSError:
            return None
        if not first.startswith("#!"):
            return None
        for pattern, lang in self.SHEBANG_PATTERNS:
            if pattern in first:
                return lang
        return None

    def detect_project(self, directory: str) -> Language:
        d = pathlib.Path(directory)
        py_count = len(list(d.rglob("*.py")))
        ts_count = len(list(d.rglob("*.ts"))) + len(list(d.rglob("*.tsx")))
        if py_count > ts_count:
            return Language.PYTHON
        if ts_count > py_count:
            return Language.TYPESCRIPT
        if (d / "pyproject.toml").exists() or (d / "setup.py").exists():
            return Language.PYTHON
        if (d / "tsconfig.json").exists() or (d / "package.json").exists():
            return Language.TYPESCRIPT
        return Language.PYTHON


# ---------------------------------------------------------------------------
# FileDiscovery
# ---------------------------------------------------------------------------


class FileDiscovery:
    """Discover source files, respecting .gitignore and config exclusions."""

    def __init__(
        self,
        include: Sequence[str] = ("**/*.py", "**/*.ts"),
        exclude: Sequence[str] = (),
        respect_gitignore: bool = True,
    ) -> None:
        self.include = list(include)
        self.exclude = list(exclude)
        self.respect_gitignore = respect_gitignore
        self._gitignore_patterns: Optional[List[str]] = None

    def discover(self, roots: Sequence[str]) -> List[str]:
        found: List[str] = []
        for root in roots:
            p = pathlib.Path(root)
            if p.is_file():
                if self._matches_include(str(p)):
                    found.append(str(p.resolve()))
            elif p.is_dir():
                found.extend(self._walk(p))
            else:
                logger.warning("Path does not exist: %s", root)
        return sorted(set(found))

    def _walk(self, directory: pathlib.Path) -> Generator[str, None, None]:
        gitignore = self._load_gitignore(directory) if self.respect_gitignore else []
        for pat in self.include:
            for match in directory.glob(pat):
                if not match.is_file():
                    continue
                rel = str(match.relative_to(directory))
                if self._is_excluded(rel, gitignore):
                    continue
                yield str(match.resolve())

    def _matches_include(self, path: str) -> bool:
        for pat in self.include:
            if fnmatch.fnmatch(path, pat) or pathlib.Path(path).match(pat):
                return True
        return True  # single file explicitly given

    def _is_excluded(self, rel: str, gitignore: List[str]) -> bool:
        for pat in self.exclude:
            if fnmatch.fnmatch(rel, pat):
                return True
        for pat in gitignore:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _load_gitignore(self, directory: pathlib.Path) -> List[str]:
        if self._gitignore_patterns is not None:
            return self._gitignore_patterns
        patterns: List[str] = []
        gi = directory / ".gitignore"
        if gi.is_file():
            with open(gi) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        self._gitignore_patterns = patterns
        return patterns


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class CacheManager:
    """Manages the analysis cache directory for incremental analysis."""

    def __init__(self, cache_dir: str = ".reftype-cache") -> None:
        self.cache_dir = pathlib.Path(cache_dir)

    def ensure(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def get(self, file_path: str) -> Optional[AnalysisResult]:
        key = self._key(file_path)
        meta_path = self.cache_dir / f"{key}.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("source_hash") != self._hash_file(file_path):
            return None
        max_age = data.get("max_age_hours", 168) * 3600
        if time.time() - data.get("timestamp", 0) > max_age:
            return None
        return self._deserialise_result(data.get("result", {}), file_path)

    def put(self, file_path: str, result: AnalysisResult) -> None:
        self.ensure()
        key = self._key(file_path)
        meta_path = self.cache_dir / f"{key}.json"
        data = {
            "source_hash": self._hash_file(file_path),
            "timestamp": time.time(),
            "max_age_hours": 168,
            "result": self._serialise_result(result),
        }
        try:
            with open(meta_path, "w") as fh:
                json.dump(data, fh)
        except OSError as exc:
            logger.warning("Failed to write cache for %s: %s", file_path, exc)

    def invalidate(self, file_path: str) -> None:
        key = self._key(file_path)
        meta_path = self.cache_dir / f"{key}.json"
        meta_path.unlink(missing_ok=True)

    def clear(self) -> int:
        count = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    count += 1
        return count

    @staticmethod
    def _serialise_result(result: AnalysisResult) -> Dict[str, Any]:
        bugs = []
        for b in result.bugs:
            bugs.append(
                {
                    "id": b.id,
                    "message": b.message,
                    "severity": b.severity.value,
                    "location": {
                        "file": b.location.file,
                        "line": b.location.line,
                        "column": b.location.column,
                    },
                    "category": b.category,
                    "fix_suggestion": b.fix_suggestion,
                }
            )
        contracts = []
        for c in result.contracts:
            contracts.append(
                {
                    "name": c.name,
                    "file": c.file,
                    "line": c.line,
                    "params": {k: str(v) for k, v in c.params.items()},
                    "return_type": str(c.return_type) if c.return_type else None,
                    "preconditions": c.preconditions,
                    "postconditions": c.postconditions,
                }
            )
        return {
            "file": result.file,
            "language": result.language.value,
            "bugs": bugs,
            "contracts": contracts,
            "duration_ms": result.duration_ms,
            "functions_analyzed": result.functions_analyzed,
            "cegar_iterations": result.cegar_iterations,
            "timed_out": result.timed_out,
        }

    @staticmethod
    def _deserialise_result(data: Dict[str, Any], file_path: str) -> AnalysisResult:
        bugs = []
        for bd in data.get("bugs", []):
            loc_d = bd.get("location", {})
            bugs.append(
                Bug(
                    id=bd.get("id", ""),
                    message=bd.get("message", ""),
                    severity=Severity(bd.get("severity", "warning")),
                    location=SourceLocation(
                        file=loc_d.get("file", file_path),
                        line=loc_d.get("line", 0),
                        column=loc_d.get("column", 0),
                    ),
                    category=bd.get("category", ""),
                    fix_suggestion=bd.get("fix_suggestion"),
                )
            )
        contracts = []
        for cd in data.get("contracts", []):
            contracts.append(
                FunctionContract(
                    name=cd.get("name", ""),
                    file=cd.get("file", file_path),
                    line=cd.get("line", 0),
                    preconditions=cd.get("preconditions", []),
                    postconditions=cd.get("postconditions", []),
                )
            )
        return AnalysisResult(
            file=file_path,
            language=Language(data.get("language", "python")),
            bugs=bugs,
            contracts=contracts,
            duration_ms=data.get("duration_ms", 0),
            functions_analyzed=data.get("functions_analyzed", 0),
            cegar_iterations=data.get("cegar_iterations", 0),
            timed_out=data.get("timed_out", False),
        )


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------


class PluginLoader:
    """Loads analysis plugins from entry points."""

    ENTRY_POINT_GROUP = "reftype.plugins"

    def __init__(self) -> None:
        self._plugins: Dict[str, Any] = {}

    def discover(self) -> List[str]:
        names: List[str] = []
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points

                eps = entry_points(group=self.ENTRY_POINT_GROUP)
            else:
                from importlib.metadata import entry_points

                all_eps = entry_points()
                eps = all_eps.get(self.ENTRY_POINT_GROUP, [])
            for ep in eps:
                names.append(ep.name)
        except Exception as exc:
            logger.debug("Plugin discovery failed: %s", exc)
        return names

    def load(self, name: str) -> Any:
        if name in self._plugins:
            return self._plugins[name]
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points

                eps = entry_points(group=self.ENTRY_POINT_GROUP)
            else:
                from importlib.metadata import entry_points

                all_eps = entry_points()
                eps = all_eps.get(self.ENTRY_POINT_GROUP, [])
            for ep in eps:
                if ep.name == name:
                    plugin = ep.load()
                    self._plugins[name] = plugin
                    return plugin
        except Exception as exc:
            logger.warning("Failed to load plugin %s: %s", name, exc)
        return None

    def load_all(self) -> Dict[str, Any]:
        for name in self.discover():
            self.load(name)
        return dict(self._plugins)


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------


def _analyze_file_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Worker function executed in a child process.

    Uses the real liquid type / shape analysis engine instead of regex
    pattern matching.  Falls back to basic AST parsing when the engine
    is unavailable or raises.
    """
    file_path, language_str = args
    start = time.monotonic()
    lang = Language(language_str) if language_str != "auto" else Language.PYTHON
    bugs: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    functions_analyzed = 0
    cegar_iterations = 0
    timed_out = False

    # Map from liquid / shape engine kinds to CLI category strings
    _KIND_TO_CLI_CAT = {
        "NULL_DEREF": "null-deref",
        "DIV_BY_ZERO": "division-by-zero",
        "INDEX_OOB": "index-out-of-bounds",
        "TYPE_ERROR": "type-error",
        "ATTRIBUTE_ERROR": "attribute-error",
        "PRECONDITION_VIOLATION": "precondition-violation",
        "UNSAT_CONSTRAINT": "type-error",
        # Shape error kinds
        "DIM_MISMATCH": "shape-error",
        "NDIM_MISMATCH": "shape-error",
        "RESHAPE_INVALID": "shape-error",
        "BROADCAST_FAIL": "shape-error",
        "MATMUL_INCOMPAT": "shape-error",
        "CAT_INCOMPAT": "shape-error",
        "CONV_INCOMPAT": "shape-error",
    }

    try:
        with open(file_path, "r", errors="replace") as fh:
            source = fh.read()

        analysis_ok = False

        # ── Try the real analysis engine ──────────────────────────────
        if lang == Language.PYTHON:
            try:
                from ..api import analyze, liquid_analyze, _HAS_LIQUID
                from ..tensor_shapes import analyze_shapes

                # Run liquid type analysis (Z3-backed if available)
                if _HAS_LIQUID:
                    liq_result = liquid_analyze(source, filename=file_path)
                else:
                    liq_result = analyze(source, filename=file_path)

                for b in liq_result.bugs:
                    cat = b.category.value if hasattr(b.category, "value") else str(b.category)
                    bugs.append(
                        {
                            "id": f"{cat}-{b.location.line}",
                            "message": b.message,
                            "severity": b.severity,
                            "location": {
                                "file": file_path,
                                "line": b.location.line,
                                "column": b.location.column,
                            },
                            "category": cat.replace("_", "-"),
                        }
                    )
                functions_analyzed += liq_result.functions_analyzed
                cegar_iterations += getattr(liq_result, "guards_harvested", 0)

                # Extract contracts if available
                liq_contracts = getattr(liq_result, "_liquid_contracts", None)
                if liq_contracts:
                    for name, contract in liq_contracts.items():
                        contracts.append(
                            {
                                "name": name,
                                "file": file_path,
                                "line": getattr(contract, "line", 0),
                                "params": {
                                    p: str(t) for p, t in contract.params.items()
                                } if hasattr(contract, "params") else {},
                                "return_type": str(contract.return_type) if hasattr(contract, "return_type") else None,
                                "preconditions": [str(p) for p in contract.preconditions] if hasattr(contract, "preconditions") else [],
                                "postconditions": [str(p) for p in contract.postconditions] if hasattr(contract, "postconditions") else [],
                            }
                        )

                # Run tensor shape analysis
                try:
                    shape_result = analyze_shapes(source)
                    for err in shape_result.errors:
                        kind_name = err.kind.name if hasattr(err.kind, "name") else str(err.kind)
                        bugs.append(
                            {
                                "id": f"shape-{err.line}",
                                "message": err.message,
                                "severity": err.severity,
                                "location": {
                                    "file": file_path,
                                    "line": err.line,
                                    "column": err.col,
                                },
                                "category": _KIND_TO_CLI_CAT.get(kind_name, "shape-error"),
                            }
                        )
                    functions_analyzed += shape_result.functions_analyzed
                except Exception as exc:
                    logger.warning("Shape analysis failed for %s: %s", file_path, exc)

                analysis_ok = True

            except Exception as exc:
                logger.warning("Analysis engine unavailable for %s: %s", file_path, exc)

        # ── Fallback: basic AST-level parsing ─────────────────────────
        if not analysis_ok:
            import ast as _ast
            try:
                tree = _ast.parse(source)
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        functions_analyzed += 1
                        contracts.append(
                            {
                                "name": node.name,
                                "file": file_path,
                                "line": node.lineno,
                                "params": {},
                                "return_type": None,
                                "preconditions": [],
                                "postconditions": [],
                            }
                        )
                        cegar_iterations += 1
            except _ast.error:
                pass

        elapsed = (time.monotonic() - start) * 1000

    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        bugs.append(
            {
                "id": "parse-error",
                "message": str(exc),
                "severity": "error",
                "location": {"file": file_path, "line": 1, "column": 0},
                "category": "parse-error",
            }
        )

    return {
        "file": file_path,
        "language": lang.value,
        "bugs": bugs,
        "contracts": contracts,
        "duration_ms": elapsed,
        "functions_analyzed": functions_analyzed,
        "cegar_iterations": cegar_iterations,
        "timed_out": timed_out,
    }


class ParallelExecutor:
    """Parallel analysis of multiple files using multiprocessing."""

    def __init__(
        self,
        workers: int = 0,
        chunk_size: int = 4,
        timeout_per_file: float = 60.0,
        language: Language = Language.AUTO,
    ) -> None:
        self.workers = workers if workers > 0 else max(1, (os.cpu_count() or 1) - 1)
        self.chunk_size = chunk_size
        self.timeout_per_file = timeout_per_file
        self.language = language
        self._detector = LanguageDetector()

    def execute(
        self,
        files: Sequence[str],
        progress: Optional[ProgressReporter] = None,
        signal_handler: Optional[SignalHandler] = None,
    ) -> List[AnalysisResult]:
        if not files:
            return []

        tasks: List[Tuple[str, str]] = []
        for f in files:
            if self.language == Language.AUTO:
                detected = self._detector.detect_file(f)
                lang_str = detected.value if detected else "python"
            else:
                lang_str = self.language.value
            tasks.append((f, lang_str))

        if len(files) == 1 or self.workers == 1:
            return self._execute_sequential(tasks, progress, signal_handler)

        return self._execute_parallel(tasks, progress, signal_handler)

    def _execute_sequential(
        self,
        tasks: List[Tuple[str, str]],
        progress: Optional[ProgressReporter],
        signal_handler: Optional[SignalHandler],
    ) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        for task in tasks:
            if signal_handler and signal_handler.interrupted:
                break
            raw = _analyze_file_worker(task)
            results.append(self._raw_to_result(raw))
            if progress:
                progress.update(1, task[0].rsplit("/", 1)[-1])
        return results

    def _execute_parallel(
        self,
        tasks: List[Tuple[str, str]],
        progress: Optional[ProgressReporter],
        signal_handler: Optional[SignalHandler],
    ) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        try:
            with multiprocessing.Pool(processes=self.workers) as pool:
                for raw in pool.imap_unordered(
                    _analyze_file_worker, tasks, chunksize=self.chunk_size
                ):
                    if signal_handler and signal_handler.interrupted:
                        pool.terminate()
                        break
                    results.append(self._raw_to_result(raw))
                    if progress:
                        progress.update(1, raw["file"].rsplit("/", 1)[-1])
        except Exception as exc:
            logger.error("Parallel execution failed: %s", exc)
            return self._execute_sequential(tasks, progress, signal_handler)
        return results

    @staticmethod
    def _raw_to_result(raw: Dict[str, Any]) -> AnalysisResult:
        bugs = []
        for bd in raw.get("bugs", []):
            loc_d = bd.get("location", {})
            bugs.append(
                Bug(
                    id=bd.get("id", ""),
                    message=bd.get("message", ""),
                    severity=Severity(bd.get("severity", "warning")),
                    location=SourceLocation(
                        file=loc_d.get("file", ""),
                        line=loc_d.get("line", 0),
                        column=loc_d.get("column", 0),
                    ),
                    category=bd.get("category", ""),
                )
            )
        contracts = []
        for cd in raw.get("contracts", []):
            contracts.append(
                FunctionContract(
                    name=cd.get("name", ""),
                    file=cd.get("file", ""),
                    line=cd.get("line", 0),
                    preconditions=cd.get("preconditions", []),
                    postconditions=cd.get("postconditions", []),
                )
            )
        return AnalysisResult(
            file=raw["file"],
            language=Language(raw.get("language", "python")),
            bugs=bugs,
            contracts=contracts,
            duration_ms=raw.get("duration_ms", 0),
            functions_analyzed=raw.get("functions_analyzed", 0),
            cegar_iterations=raw.get("cegar_iterations", 0),
            timed_out=raw.get("timed_out", False),
        )


# ---------------------------------------------------------------------------
# ExitCodeManager
# ---------------------------------------------------------------------------


class ExitCodeManager:
    """Determines exit code from analysis results."""

    EXIT_SUCCESS = 0
    EXIT_BUGS_FOUND = 1
    EXIT_ERROR = 2

    def __init__(self, fail_on_new_bugs: bool = False) -> None:
        self.fail_on_new_bugs = fail_on_new_bugs

    def compute(
        self,
        summary: AnalysisSummary,
        new_bugs: Optional[int] = None,
    ) -> int:
        if summary.files_timed_out > 0 and summary.total_files == summary.files_timed_out:
            return self.EXIT_ERROR
        if self.fail_on_new_bugs and new_bugs is not None:
            return self.EXIT_BUGS_FOUND if new_bugs > 0 else self.EXIT_SUCCESS
        if summary.total_bugs > 0:
            has_errors = summary.bugs_by_severity.get("error", 0) > 0
            if has_errors:
                return self.EXIT_BUGS_FOUND
        return self.EXIT_SUCCESS


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TelemetryCollector:
    """Optional anonymous usage telemetry (opt-in only)."""

    TELEMETRY_FILE = ".reftype-telemetry.json"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._events: List[Dict[str, Any]] = []
        self._session_id = uuid.uuid4().hex[:12]

    def record(self, event: str, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self._events.append(
            {
                "event": event,
                "session": self._session_id,
                "timestamp": time.time(),
                **kwargs,
            }
        )

    def flush(self) -> None:
        if not self.enabled or not self._events:
            return
        try:
            p = pathlib.Path(self.TELEMETRY_FILE)
            existing: List[Dict[str, Any]] = []
            if p.exists():
                with open(p) as fh:
                    existing = json.load(fh)
            existing.extend(self._events)
            # keep only last 1000 events
            existing = existing[-1000:]
            with open(p, "w") as fh:
                json.dump(existing, fh)
            self._events.clear()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# UpdateChecker
# ---------------------------------------------------------------------------


class UpdateChecker:
    """Checks for newer versions (non-blocking, best-effort)."""

    PYPI_URL = "https://pypi.org/pypi/reftype/json"
    CACHE_FILE = ".reftype-update-check"
    CHECK_INTERVAL = 86400  # 1 day

    def __init__(self) -> None:
        self._latest: Optional[str] = None

    def should_check(self) -> bool:
        p = pathlib.Path(self.CACHE_FILE)
        if not p.exists():
            return True
        try:
            ts = float(p.read_text().strip())
            return (time.time() - ts) > self.CHECK_INTERVAL
        except (ValueError, OSError):
            return True

    def check_async(self) -> None:
        if not self.should_check():
            return
        t = threading.Thread(target=self._do_check, daemon=True)
        t.start()

    def _do_check(self) -> None:
        try:
            import urllib.request

            with urllib.request.urlopen(self.PYPI_URL, timeout=5) as resp:
                data = json.load(resp)
                self._latest = data.get("info", {}).get("version")
            p = pathlib.Path(self.CACHE_FILE)
            p.write_text(str(time.time()))
        except Exception as exc:
            logger.warning("Update check failed: %s", exc)

    def notify_if_outdated(self, current: str = _VERSION) -> Optional[str]:
        if self._latest and self._latest != current:
            return (
                f"A newer version of reftype is available: {self._latest} "
                f"(current: {current}). Run `pip install --upgrade reftype` to update."
            )
        return None


# ---------------------------------------------------------------------------
# SARIF / HTML / JSON output helpers
# ---------------------------------------------------------------------------


class SarifGenerator:
    """Generates SARIF 2.1.0 output for GitHub Advanced Security."""

    SARIF_VERSION = "2.1.0"
    SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"

    def generate(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        rules: Dict[str, Dict[str, Any]] = {}
        sarif_results: List[Dict[str, Any]] = []

        for result in results:
            for bug in result.bugs:
                rule_id = bug.category
                if rule_id not in rules:
                    rules[rule_id] = {
                        "id": rule_id,
                        "name": rule_id.replace("-", " ").title(),
                        "shortDescription": {"text": f"Refinement type: {rule_id}"},
                        "defaultConfiguration": {
                            "level": self._severity_to_level(bug.severity)
                        },
                    }
                sarif_results.append(self._bug_to_result(bug, rule_id))

        return {
            "$schema": self.SCHEMA,
            "version": self.SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "reftype",
                            "version": _VERSION,
                            "informationUri": "https://github.com/reftype/reftype",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": sarif_results,
                }
            ],
        }

    @staticmethod
    def _severity_to_level(sev: Severity) -> str:
        return {
            Severity.ERROR: "error",
            Severity.WARNING: "warning",
            Severity.INFO: "note",
            Severity.HINT: "note",
        }.get(sev, "warning")

    @staticmethod
    def _bug_to_result(bug: Bug, rule_id: str) -> Dict[str, Any]:
        return {
            "ruleId": rule_id,
            "level": SarifGenerator._severity_to_level(bug.severity),
            "message": {"text": bug.message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": bug.location.file,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {
                            "startLine": bug.location.line,
                            "startColumn": bug.location.column + 1,
                        },
                    }
                }
            ],
            "fingerprints": {"reftype/v1": bug.fingerprint()},
            "fixes": (
                [
                    {
                        "description": {"text": bug.fix_suggestion},
                        "artifactChanges": [],
                    }
                ]
                if bug.fix_suggestion
                else []
            ),
        }


class HtmlReportGenerator:
    """Generates a self-contained HTML analysis report."""

    def generate(
        self, results: List[AnalysisResult], summary: AnalysisSummary
    ) -> str:
        bugs_html = []
        for result in results:
            for bug in result.bugs:
                sev_class = bug.severity.value
                bugs_html.append(
                    f'<tr class="{sev_class}">'
                    f"<td>{bug.severity.value}</td>"
                    f"<td>{bug.location}</td>"
                    f"<td>{bug.category}</td>"
                    f"<td>{bug.message}</td>"
                    f"<td>{bug.fix_suggestion or ''}</td>"
                    f"</tr>"
                )

        contracts_html = []
        for result in results:
            for c in result.contracts:
                params = ", ".join(f"{k}: {v}" for k, v in c.params.items())
                ret = str(c.return_type) if c.return_type else "—"
                contracts_html.append(
                    f"<tr>"
                    f"<td>{c.file}:{c.line}</td>"
                    f"<td>{c.name}</td>"
                    f"<td>{params}</td>"
                    f"<td>{ret}</td>"
                    f"</tr>"
                )

        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title>reftype Analysis Report</title>
        <style>
          body {{ font-family: system-ui, sans-serif; margin: 2em; }}
          h1 {{ color: #333; }}
          table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background: #f4f4f4; }}
          .error {{ background: #ffe0e0; }}
          .warning {{ background: #fff3cd; }}
          .info {{ background: #d1ecf1; }}
          .hint {{ background: #f0f0f0; }}
          .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; }}
          .card {{ background: #f8f9fa; border-radius: 8px; padding: 1em; }}
          .card h3 {{ margin: 0 0 0.5em; }}
          .card .value {{ font-size: 2em; font-weight: bold; }}
        </style>
        </head>
        <body>
        <h1>reftype Analysis Report</h1>
        <p>Generated {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <div class="summary">
          <div class="card"><h3>Files</h3><div class="value">{summary.total_files}</div></div>
          <div class="card"><h3>Functions</h3><div class="value">{summary.total_functions}</div></div>
          <div class="card"><h3>Bugs</h3><div class="value">{summary.total_bugs}</div></div>
          <div class="card"><h3>Contracts</h3><div class="value">{summary.total_contracts}</div></div>
          <div class="card"><h3>Duration</h3><div class="value">{summary.duration_ms:.0f}ms</div></div>
        </div>
        <h2>Bugs ({len(bugs_html)})</h2>
        <table>
        <tr><th>Severity</th><th>Location</th><th>Category</th><th>Message</th><th>Fix</th></tr>
        {"".join(bugs_html)}
        </table>
        <h2>Inferred Contracts ({len(contracts_html)})</h2>
        <table>
        <tr><th>Location</th><th>Function</th><th>Parameters</th><th>Return</th></tr>
        {"".join(contracts_html)}
        </table>
        </body>
        </html>
        """)


class JsonOutputGenerator:
    """Generates JSON output."""

    def generate(
        self, results: List[AnalysisResult], summary: AnalysisSummary
    ) -> Dict[str, Any]:
        return {
            "version": _VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total_files": summary.total_files,
                "total_functions": summary.total_functions,
                "total_bugs": summary.total_bugs,
                "bugs_by_severity": summary.bugs_by_severity,
                "bugs_by_category": summary.bugs_by_category,
                "total_contracts": summary.total_contracts,
                "total_cegar_iterations": summary.total_cegar_iterations,
                "duration_ms": summary.duration_ms,
            },
            "results": [CacheManager._serialise_result(r) for r in results],
        }


class PyiGenerator:
    """Generates .pyi stub files from inferred contracts."""

    def generate(self, contracts: List[FunctionContract]) -> str:
        lines: List[str] = ["# Auto-generated by reftype", ""]
        for c in contracts:
            params = ", ".join(f"{k}: {v}" for k, v in c.params.items())
            ret = f" -> {c.return_type}" if c.return_type else ""
            lines.append(f"def {c.name}({params}){ret}: ...")
            for pre in c.preconditions:
                lines.append(f"    # requires: {pre}")
            for post in c.postconditions:
                lines.append(f"    # ensures: {post}")
            lines.append("")
        return "\n".join(lines)


class DtsGenerator:
    """Generates .d.ts declaration files from inferred contracts."""

    TYPE_MAP: Dict[str, str] = {
        "int": "number",
        "float": "number",
        "str": "string",
        "bool": "boolean",
        "None": "void",
        "NoneType": "void",
        "list": "Array<any>",
        "dict": "Record<string, any>",
    }

    def generate(self, contracts: List[FunctionContract]) -> str:
        lines: List[str] = ["// Auto-generated by reftype", ""]
        for c in contracts:
            params = ", ".join(
                f"{k}: {self._map_type(str(v))}" for k, v in c.params.items()
            )
            ret = self._map_type(str(c.return_type)) if c.return_type else "void"
            lines.append(f"declare function {c.name}({params}): {ret};")
            for pre in c.preconditions:
                lines.append(f"  // requires: {pre}")
            for post in c.postconditions:
                lines.append(f"  // ensures: {post}")
            lines.append("")
        return "\n".join(lines)

    def _map_type(self, t: str) -> str:
        base = t.split("|")[0].strip().lstrip("{").split()[0]
        return self.TYPE_MAP.get(base, "any")


# ---------------------------------------------------------------------------
# Subcommand Protocol & Implementations
# ---------------------------------------------------------------------------


class Command(Protocol):
    """Protocol for CLI subcommands."""

    def register(self, parser: argparse.ArgumentParser) -> None: ...
    def execute(self, args: argparse.Namespace) -> int: ...


# ── AnalyzeCommand ────────────────────────────────────────────────────────


class AnalyzeCommand:
    """Analyze files and directories for refinement type bugs."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", default=["."], help="Files or directories")
        parser.add_argument(
            "-l", "--language", choices=["python", "typescript", "auto"], default="auto"
        )
        parser.add_argument(
            "-f",
            "--format",
            choices=["pyi", "dts", "sarif", "html", "json"],
            default="json",
        )
        parser.add_argument("-o", "--output", help="Output file (default: stdout)")
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("-c", "--config", help="Config file path")
        parser.add_argument("--include", nargs="*", help="Include patterns")
        parser.add_argument("--exclude", nargs="*", help="Exclude patterns")
        parser.add_argument("--max-functions", type=int, default=0)
        parser.add_argument("--timeout", type=float, default=300.0)
        parser.add_argument("-w", "--workers", type=int, default=0)
        parser.add_argument("--incremental", action="store_true", default=False)
        parser.add_argument("--baseline", help="Baseline file for comparison")
        parser.add_argument("--no-color", action="store_true")
        parser.add_argument("--fail-on-new-bugs", action="store_true")
        parser.add_argument(
            "--stubs-dir", metavar="PATH",
            help="Directory containing .pyi stub files to use as known types",
        )
        parser.add_argument(
            "--mypy-baseline", metavar="FILE",
            help="mypy output file (text or JSON) for baseline comparison",
        )
        parser.add_argument(
            "--pyright-baseline", metavar="FILE",
            help="pyright JSON output file for baseline comparison",
        )

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)
        errors = loader.validate(cfg)
        if errors:
            for e in errors:
                sys.stderr.write(f"Config error: {e}\n")
            return 2

        signal_handler = SignalHandler()
        signal_handler.install()

        telemetry = TelemetryCollector(enabled=cfg.telemetry_enabled)
        telemetry.record("analyze_start", files=len(cfg.paths))

        # Load .pyi stubs if provided
        known_stubs: List[StubType] = []
        stubs_dir = getattr(args, "stubs_dir", None)
        if stubs_dir:
            known_stubs = _scan_pyi_stubs(stubs_dir)
            sys.stderr.write(f"Loaded {len(known_stubs)} type annotations from stubs\n")
        typeshed = _detect_typeshed()
        if typeshed and not stubs_dir:
            logger.debug("Auto-detected typeshed at %s", typeshed)

        try:
            include = getattr(args, "include", None) or cfg.include_patterns
            exclude = getattr(args, "exclude", None) or cfg.exclude_patterns
            discovery = FileDiscovery(include=include, exclude=exclude)
            files = discovery.discover(getattr(args, "paths", cfg.paths))

            if not files:
                sys.stderr.write("No files found to analyse.\n")
                return 0

            if cfg.max_functions > 0:
                files = files[: cfg.max_functions]

            logger.info("Analysing %d files with %d workers", len(files), cfg.effective_workers())

            cache = CacheManager(cfg.incremental.cache_dir) if cfg.incremental.enabled else None

            cached_results: List[AnalysisResult] = []
            files_to_analyze = list(files)
            if cache:
                remaining = []
                for f in files_to_analyze:
                    cached = cache.get(f)
                    if cached:
                        cached_results.append(cached)
                    else:
                        remaining.append(f)
                files_to_analyze = remaining
                if cached_results:
                    logger.info("Using %d cached results", len(cached_results))

            progress = ProgressReporter(total=len(files_to_analyze))
            executor = ParallelExecutor(
                workers=cfg.effective_workers(),
                chunk_size=cfg.parallel.chunk_size,
                timeout_per_file=cfg.parallel.timeout_per_file,
                language=cfg.language,
            )

            new_results = executor.execute(files_to_analyze, progress, signal_handler)
            progress.finish()

            if cache:
                for r in new_results:
                    cache.put(r.file, r)

            all_results = cached_results + new_results
            summary = AnalysisSummary()
            for r in all_results:
                summary.merge(r)

            # Baseline comparison
            new_bug_count: Optional[int] = None
            if cfg.baseline_file:
                new_bug_count = self._compare_baseline(cfg.baseline_file, all_results)

            # Checker baseline comparisons (mypy / pyright)
            mypy_bl = getattr(args, "mypy_baseline", None)
            pyright_bl = getattr(args, "pyright_baseline", None)
            if mypy_bl:
                mypy_issues = _parse_mypy_baseline(mypy_bl)
                cmp = _compare_checker_baseline(mypy_issues, all_results)
                sys.stderr.write(f"\n── mypy baseline comparison ({len(mypy_issues)} issues) ──\n")
                sys.stderr.write(cmp.summary_text() + "\n")
            if pyright_bl:
                pyright_issues = _parse_pyright_baseline(pyright_bl)
                cmp = _compare_checker_baseline(pyright_issues, all_results)
                sys.stderr.write(f"\n── pyright baseline comparison ({len(pyright_issues)} issues) ──\n")
                sys.stderr.write(cmp.summary_text() + "\n")

            # Output
            self._write_output(args, cfg, all_results, summary)

            # Terminal summary (only when stdout is a TTY, not piped)
            use_color = not getattr(args, "no_color", False)
            printer = ResultPrinter(stream=sys.stderr, color=use_color)
            if getattr(args, "verbose", 0) >= 1:
                for r in all_results:
                    printer.print_result(r)
            printer.print_summary(summary)

            telemetry.record(
                "analyze_end",
                bugs=summary.total_bugs,
                files=summary.total_files,
                duration=summary.duration_ms,
            )
            telemetry.flush()

            exit_mgr = ExitCodeManager(fail_on_new_bugs=cfg.fail_on_new_bugs)
            return exit_mgr.compute(summary, new_bug_count)

        except Exception as exc:
            return ErrorHandler().handle(exc, "analyze")
        finally:
            signal_handler.uninstall()

    def _compare_baseline(
        self, baseline_path: str, results: List[AnalysisResult]
    ) -> int:
        try:
            with open(baseline_path) as fh:
                baseline_data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load baseline: %s", exc)
            return 0

        baseline_fps: Set[str] = set()
        for r in baseline_data.get("results", []):
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                baseline_fps.add(fp)

        new_bugs = 0
        for result in results:
            for bug in result.bugs:
                if bug.fingerprint() not in baseline_fps:
                    new_bugs += 1
        return new_bugs

    def _write_output(
        self,
        args: argparse.Namespace,
        cfg: Configuration,
        results: List[AnalysisResult],
        summary: AnalysisSummary,
    ) -> None:
        fmt = getattr(args, "format", cfg.output_format.value)
        if isinstance(fmt, OutputFormat):
            fmt = fmt.value
        output_path = getattr(args, "output", cfg.output_file)

        content: str
        if fmt == "sarif":
            data = SarifGenerator().generate(results)
            content = json.dumps(data, indent=2)
        elif fmt == "html":
            content = HtmlReportGenerator().generate(results, summary)
        elif fmt == "json":
            data = JsonOutputGenerator().generate(results, summary)
            content = json.dumps(data, indent=2)
        elif fmt == "pyi":
            all_contracts = [c for r in results for c in r.contracts]
            content = PyiGenerator().generate(all_contracts)
        elif fmt == "dts":
            all_contracts = [c for r in results for c in r.contracts]
            content = DtsGenerator().generate(all_contracts)
        else:
            data = JsonOutputGenerator().generate(results, summary)
            content = json.dumps(data, indent=2)

        if output_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(content)
            logger.info("Output written to %s", output_path)
        else:
            sys.stdout.write(content + "\n")


# ── WatchCommand ──────────────────────────────────────────────────────────


class WatchCommand:
    """File watcher for incremental development."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", default=["."])
        parser.add_argument("-l", "--language", default="auto")
        parser.add_argument("--debounce", type=float, default=0.5)
        parser.add_argument(
            "--editor", choices=["vim", "emacs", "vscode"], default=None
        )
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("-c", "--config", help="Config file path")

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)

        signal_handler = SignalHandler()
        signal_handler.install()

        debounce = getattr(args, "debounce", 0.5)
        printer = ResultPrinter()

        discovery = FileDiscovery(
            include=cfg.include_patterns, exclude=cfg.exclude_patterns
        )
        files = discovery.discover(getattr(args, "paths", cfg.paths))
        if not files:
            sys.stderr.write("No files found to watch.\n")
            return 0

        sys.stderr.write(f"Watching {len(files)} files (debounce={debounce}s)…\n")

        mtimes: Dict[str, float] = {}
        for f in files:
            try:
                mtimes[f] = os.path.getmtime(f)
            except OSError:
                pass

        executor = ParallelExecutor(workers=1, language=cfg.language)

        try:
            while not signal_handler.interrupted:
                changed: List[str] = []
                for f in files:
                    try:
                        mt = os.path.getmtime(f)
                    except OSError:
                        continue
                    prev = mtimes.get(f)
                    if prev is None or mt > prev:
                        mtimes[f] = mt
                        if prev is not None:
                            changed.append(f)

                if changed:
                    sys.stderr.write(
                        f"\n[{time.strftime('%H:%M:%S')}] "
                        f"Detected changes in {len(changed)} file(s)\n"
                    )
                    results = executor.execute(changed)
                    for r in results:
                        printer.print_result(r)
                    summary = AnalysisSummary()
                    for r in results:
                        summary.merge(r)
                    if summary.total_bugs > 0:
                        sys.stderr.write(
                            _c(
                                f"⚠ {summary.total_bugs} bug(s) found\n",
                                "yellow",
                                use_color=True,
                            )
                        )
                    else:
                        sys.stderr.write(
                            _c("✓ No bugs found\n", "green", use_color=True)
                        )

                time.sleep(debounce)
        except KeyboardInterrupt:
            pass
        finally:
            signal_handler.uninstall()

        sys.stderr.write("\nWatch stopped.\n")
        return 0


# ── CiCheckCommand ────────────────────────────────────────────────────────


class CiCheckCommand:
    """CI pipeline mode with exit codes and SARIF output."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", default=["."])
        parser.add_argument("-l", "--language", default="auto")
        parser.add_argument("--baseline", help="Baseline file")
        parser.add_argument("--sarif-output", help="SARIF output path")
        parser.add_argument("--max-new-bugs", type=int, default=0)
        parser.add_argument("--max-total-bugs", type=int, default=None)
        parser.add_argument("--fail-on-new-bugs", action="store_true", default=True)
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("-c", "--config", help="Config file path")
        parser.add_argument("-w", "--workers", type=int, default=0)
        parser.add_argument("--timeout", type=float, default=300.0)
        parser.add_argument(
            "-f", "--format",
            choices=["sarif", "json"],
            default="sarif",
        )

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)
        cfg.fail_on_new_bugs = getattr(args, "fail_on_new_bugs", True)

        try:
            discovery = FileDiscovery(
                include=cfg.include_patterns, exclude=cfg.exclude_patterns
            )
            files = discovery.discover(getattr(args, "paths", cfg.paths))
            if not files:
                sys.stderr.write("No files found.\n")
                return 0

            executor = ParallelExecutor(
                workers=cfg.effective_workers(),
                language=cfg.language,
            )
            results = executor.execute(files)

            summary = AnalysisSummary()
            for r in results:
                summary.merge(r)

            # SARIF output
            sarif_path = getattr(args, "sarif_output", None)
            if sarif_path:
                sarif = SarifGenerator().generate(results)
                pathlib.Path(sarif_path).parent.mkdir(parents=True, exist_ok=True)
                with open(sarif_path, "w") as fh:
                    json.dump(sarif, fh, indent=2)
                logger.info("SARIF written to %s", sarif_path)

            # Baseline comparison
            new_bugs = 0
            baseline_path = getattr(args, "baseline", cfg.baseline_file)
            if baseline_path:
                analyze_cmd = AnalyzeCommand()
                new_bugs = analyze_cmd._compare_baseline(baseline_path, results)
                sys.stderr.write(f"New bugs since baseline: {new_bugs}\n")

            max_new = getattr(args, "max_new_bugs", 0)
            max_total = getattr(args, "max_total_bugs", None)

            # Thresholds
            if max_total is not None and summary.total_bugs > max_total:
                sys.stderr.write(
                    f"FAIL: {summary.total_bugs} bugs exceed threshold of {max_total}\n"
                )
                return 1
            if cfg.fail_on_new_bugs and new_bugs > max_new:
                sys.stderr.write(
                    f"FAIL: {new_bugs} new bugs exceed threshold of {max_new}\n"
                )
                return 1

            # CI annotations
            self._emit_annotations(results)

            sys.stderr.write(
                f"CI check passed: {summary.total_bugs} total bugs, {new_bugs} new\n"
            )
            return 0

        except Exception as exc:
            return ErrorHandler().handle(exc, "ci-check")

    @staticmethod
    def _emit_annotations(results: List[AnalysisResult]) -> None:
        """Emit GitHub Actions annotations."""
        is_gha = os.environ.get("GITHUB_ACTIONS") == "true"
        if not is_gha:
            return
        for result in results:
            for bug in result.bugs:
                level = "error" if bug.severity == Severity.ERROR else "warning"
                msg = bug.message.replace("\n", "%0A")
                print(
                    f"::{level} file={bug.location.file},"
                    f"line={bug.location.line},"
                    f"col={bug.location.column}::{msg}"
                )


# ── InitCommand ───────────────────────────────────────────────────────────


class InitCommand:
    """Initialise .reftype.toml configuration in the project."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--language", choices=["python", "typescript", "auto"], default="auto"
        )
        parser.add_argument("--force", action="store_true")
        parser.add_argument("directory", nargs="?", default=".")

    def execute(self, args: argparse.Namespace) -> int:
        directory = getattr(args, "directory", ".")
        config_path = pathlib.Path(directory) / ".reftype.toml"

        if config_path.exists() and not getattr(args, "force", False):
            sys.stderr.write(
                f"{config_path} already exists. Use --force to overwrite.\n"
            )
            return 1

        lang_str = getattr(args, "language", "auto")
        if lang_str == "auto":
            detector = LanguageDetector()
            detected = detector.detect_project(directory)
            lang_str = detected.value

        if lang_str == "python":
            include = '["**/*.py"]'
            exclude = '["__pycache__/**", ".venv/**", "*.egg-info/**", "dist/**"]'
        else:
            include = '["**/*.ts", "**/*.tsx"]'
            exclude = '["node_modules/**", "dist/**", "build/**"]'

        template = textwrap.dedent(f"""\
        # reftype configuration
        # https://github.com/reftype/reftype

        [reftype]
        language = "{lang_str}"
        include = {include}
        exclude = {exclude}

        [reftype.domain]
        interval_precision = 64
        octagon_enabled = false

        [reftype.cegar]
        max_iterations = 50
        refinement_strategy = "counterexample-guided"
        interpolation_enabled = true

        [reftype.incremental]
        enabled = true
        cache_dir = ".reftype-cache"

        [reftype.parallel]
        workers = 0  # auto-detect
        """)

        with open(config_path, "w") as fh:
            fh.write(template)

        sys.stdout.write(f"Created {config_path}\n")
        sys.stdout.write(f"Detected language: {lang_str}\n")
        return 0


# ── ReportCommand ─────────────────────────────────────────────────────────


class ReportCommand:
    """Generate analysis reports in various formats."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="JSON analysis results file")
        parser.add_argument(
            "-f", "--format", choices=["html", "sarif", "json"], default="html"
        )
        parser.add_argument("-o", "--output", help="Output file")

    def execute(self, args: argparse.Namespace) -> int:
        input_path = getattr(args, "input", None)
        if not input_path or not pathlib.Path(input_path).exists():
            sys.stderr.write(f"Input file not found: {input_path}\n")
            return 2

        try:
            with open(input_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Failed to read input: {exc}\n")
            return 2

        results = [
            CacheManager._deserialise_result(r, r.get("file", ""))
            for r in data.get("results", [])
        ]
        summary = AnalysisSummary()
        for r in results:
            summary.merge(r)

        fmt = getattr(args, "format", "html")
        if fmt == "html":
            content = HtmlReportGenerator().generate(results, summary)
        elif fmt == "sarif":
            sarif = SarifGenerator().generate(results)
            content = json.dumps(sarif, indent=2)
        else:
            content = json.dumps(
                JsonOutputGenerator().generate(results, summary), indent=2
            )

        output_path = getattr(args, "output", None)
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(content)
            sys.stdout.write(f"Report written to {output_path}\n")
        else:
            sys.stdout.write(content + "\n")
        return 0


# ── ExportCommand ─────────────────────────────────────────────────────────


class ExportCommand:
    """Export inferred contracts to various formats."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="JSON analysis results file")
        parser.add_argument(
            "-f", "--format", choices=["pyi", "dts", "json"], default="pyi"
        )
        parser.add_argument("-o", "--output", help="Output file")

    def execute(self, args: argparse.Namespace) -> int:
        input_path = getattr(args, "input", None)
        if not input_path or not pathlib.Path(input_path).exists():
            sys.stderr.write(f"Input file not found: {input_path}\n")
            return 2

        try:
            with open(input_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Failed to read input: {exc}\n")
            return 2

        results = [
            CacheManager._deserialise_result(r, r.get("file", ""))
            for r in data.get("results", [])
        ]
        all_contracts = [c for r in results for c in r.contracts]

        fmt = getattr(args, "format", "pyi")
        if fmt == "pyi":
            content = PyiGenerator().generate(all_contracts)
        elif fmt == "dts":
            content = DtsGenerator().generate(all_contracts)
        else:
            content = json.dumps(
                [CacheManager._serialise_result(r) for r in results], indent=2
            )

        output_path = getattr(args, "output", None)
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(content)
            sys.stdout.write(f"Exported to {output_path}\n")
        else:
            sys.stdout.write(content + "\n")
        return 0


# ── DiffCommand ───────────────────────────────────────────────────────────


class DiffCommand:
    """Compare two analysis results, show new/fixed/changed bugs."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("before", help="Before analysis results (JSON)")
        parser.add_argument("after", help="After analysis results (JSON)")
        parser.add_argument("--no-color", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        before_path = getattr(args, "before", None)
        after_path = getattr(args, "after", None)

        for label, path in [("before", before_path), ("after", after_path)]:
            if not path or not pathlib.Path(path).exists():
                sys.stderr.write(f"{label} file not found: {path}\n")
                return 2

        try:
            with open(before_path) as fh:
                before_data = json.load(fh)
            with open(after_path) as fh:
                after_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Failed to read files: {exc}\n")
            return 2

        before_fps: Dict[str, Dict[str, Any]] = {}
        for r in before_data.get("results", []):
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                before_fps[fp] = {**b, "file": r.get("file", "")}

        after_fps: Dict[str, Dict[str, Any]] = {}
        for r in after_data.get("results", []):
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                after_fps[fp] = {**b, "file": r.get("file", "")}

        new_fps = set(after_fps.keys()) - set(before_fps.keys())
        fixed_fps = set(before_fps.keys()) - set(after_fps.keys())

        use_color = not getattr(args, "no_color", False)

        sys.stdout.write(
            _c(f"\n═══ Analysis Diff ═══\n", "bold", use_color=use_color)
        )

        if new_fps:
            sys.stdout.write(
                _c(f"\n  New bugs ({len(new_fps)}):\n", "red", use_color=use_color)
            )
            for fp in sorted(new_fps):
                b = after_fps[fp]
                sys.stdout.write(
                    f"    + [{b.get('severity', 'warning')}] "
                    f"{b.get('file', '')}:{b.get('location', {}).get('line', '?')}: "
                    f"{b.get('message', '')}\n"
                )

        if fixed_fps:
            sys.stdout.write(
                _c(f"\n  Fixed bugs ({len(fixed_fps)}):\n", "green", use_color=use_color)
            )
            for fp in sorted(fixed_fps):
                b = before_fps[fp]
                sys.stdout.write(
                    f"    - [{b.get('severity', 'warning')}] "
                    f"{b.get('file', '')}:{b.get('location', {}).get('line', '?')}: "
                    f"{b.get('message', '')}\n"
                )

        unchanged = set(before_fps.keys()) & set(after_fps.keys())
        sys.stdout.write(f"\n  Summary: +{len(new_fps)} new, -{len(fixed_fps)} fixed, {len(unchanged)} unchanged\n\n")

        return 1 if new_fps else 0


# ── ServerCommand ─────────────────────────────────────────────────────────


class ServerCommand:
    """Start the Language Server Protocol server."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--transport", choices=["stdio", "tcp"], default="stdio"
        )
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=2087)
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("--log-file", help="Log file path")

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(
            getattr(args, "verbose", 0),
            log_file=getattr(args, "log_file", None),
        )
        transport = getattr(args, "transport", "stdio")

        try:
            from .lsp_server import ReftypeLspServer

            server = ReftypeLspServer()
            if transport == "stdio":
                server.run_stdio()
            else:
                host = getattr(args, "host", "127.0.0.1")
                port = getattr(args, "port", 2087)
                server.run_tcp(host, port)
            return 0
        except ImportError:
            sys.stderr.write("LSP server module not available.\n")
            return 2
        except Exception as exc:
            return ErrorHandler().handle(exc, "server")


# ── VersionCommand ────────────────────────────────────────────────────────


class VersionCommand:
    """Show version information."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--json", action="store_true", dest="as_json")

    def execute(self, args: argparse.Namespace) -> int:
        info = {
            "version": _VERSION,
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        }
        if getattr(args, "as_json", False):
            sys.stdout.write(json.dumps(info, indent=2) + "\n")
        else:
            sys.stdout.write(f"tensorguard {_VERSION}\n")
            sys.stdout.write(f"Python {sys.version}\n")
            sys.stdout.write(f"Platform {platform.platform()}\n")
        return 0


# ── ConfigCommand ─────────────────────────────────────────────────────────


class OperatorConfidenceCommand:
    """Print the per-operator confidence table (sound/complete/heuristic)."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "operators",
            nargs="*",
            help="Optional operator names to query (e.g. torch.matmul F.relu). "
            "If omitted, the full table is printed.",
        )
        parser.add_argument("--json", action="store_true", dest="as_json")

    def execute(self, args: argparse.Namespace) -> int:
        from src.operator_confidence import (
            ConfidenceTag,
            confidence_table,
            rationale_for,
            tag_for,
            to_json,
        )

        queried = list(getattr(args, "operators", []) or [])

        if queried:
            rows = [
                {
                    "operator": op,
                    "confidence": tag_for(op).value,
                    "rationale": rationale_for(op),
                }
                for op in queried
            ]
        else:
            rows = confidence_table()

        if getattr(args, "as_json", False):
            if queried:
                sys.stdout.write(json.dumps({"operators": rows}, indent=2) + "\n")
            else:
                sys.stdout.write(to_json() + "\n")
            return 0

        width = max((len(r["operator"]) for r in rows), default=8)
        for r in rows:
            sys.stdout.write(f"{r['operator']:<{width}}  {r['confidence']:<9}\n")
        if not queried:
            summary: Dict[str, int] = {t.value: 0 for t in ConfidenceTag}
            for r in rows:
                summary[r["confidence"]] += 1
            sys.stdout.write(
                "\n"
                f"{len(rows)} operators: "
                f"{summary['complete']} complete, "
                f"{summary['sound']} sound, "
                f"{summary['heuristic']} heuristic "
                "(unknown ops default to heuristic)\n"
            )
        return 0


# ── ConfigCommand ─────────────────────────────────────────────────────────


class ConfigCommand:
    """Show or edit configuration."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "action", choices=["show", "path", "defaults"], nargs="?", default="show"
        )
        parser.add_argument("-c", "--config", help="Config file path")

    def execute(self, args: argparse.Namespace) -> int:
        action = getattr(args, "action", "show")
        loader = ConfigLoader()

        if action == "path":
            path = loader.find_config_file()
            if path:
                sys.stdout.write(f"{path}\n")
            else:
                sys.stdout.write("No config file found.\n")
            return 0

        if action == "defaults":
            cfg = Configuration()
            sys.stdout.write(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
            return 0

        # show
        config_path = getattr(args, "config", None) or loader.find_config_file()
        if config_path and pathlib.Path(config_path).exists():
            file_cfg = loader.load(config_path)
            sys.stdout.write(f"# Loaded from {config_path}\n")
            sys.stdout.write(json.dumps(file_cfg, indent=2) + "\n")
        else:
            sys.stdout.write("No configuration file found. Using defaults.\n")
            cfg = Configuration()
            sys.stdout.write(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
        return 0


# ── PackageAnalyzeCommand ─────────────────────────────────────────────────


# Known type stub info for popular libraries (used by --requirements)
_KNOWN_LIBRARY_STUBS: Dict[str, Dict[str, str]] = {
    "numpy": {"stub_package": "numpy", "types": "ndarray, dtype, int64, float64"},
    "pandas": {"stub_package": "pandas-stubs", "types": "DataFrame, Series, Index"},
    "scipy": {"stub_package": "scipy", "types": "sparse.csr_matrix, optimize.OptimizeResult"},
    "requests": {"stub_package": "types-requests", "types": "Response, Session"},
    "flask": {"stub_package": "flask", "types": "Flask, Request, Response"},
    "django": {"stub_package": "django-stubs", "types": "HttpRequest, HttpResponse, QuerySet"},
    "sqlalchemy": {"stub_package": "sqlalchemy-stubs", "types": "Session, Engine, Column"},
    "torch": {"stub_package": "torch", "types": "Tensor, nn.Module, optim.Optimizer"},
    "tensorflow": {"stub_package": "tensorflow", "types": "Tensor, Variable, keras.Model"},
    "pydantic": {"stub_package": "pydantic", "types": "BaseModel, Field, validator"},
}


def _parse_requirements_txt(path: pathlib.Path) -> List[str]:
    """Extract package names from a requirements.txt file."""
    packages: List[str] = []
    if not path.is_file():
        return packages
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Strip version specifiers: numpy>=1.21 -> numpy
            name = re.split(r"[>=<!\[;]", line)[0].strip()
            if name:
                packages.append(name.lower())
    return packages


def _parse_pyproject_dependencies(path: pathlib.Path) -> List[str]:
    """Extract dependency names from pyproject.toml [project.dependencies]."""
    packages: List[str] = []
    if not path.is_file():
        return packages
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return packages
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            continue
        if re.match(r"^\[", stripped) and stripped != "[project]":
            in_deps = False
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps and stripped.startswith('"'):
            name = re.split(r'[>=<!\[;"\']', stripped.strip('", '))[0].strip()
            if name:
                packages.append(name.lower())
        if in_deps and stripped == "]":
            in_deps = False
    return packages


# ---------------------------------------------------------------------------
# .pyi stub parsing and checker baseline import utilities
# ---------------------------------------------------------------------------


@dataclass
class StubType:
    """A type annotation extracted from a .pyi stub file."""
    module: str
    qualified_name: str
    annotation: str
    line: int = 0


def _scan_pyi_stubs(stubs_dir: str) -> List[StubType]:
    """Scan a directory for .pyi files and extract type annotations via ast."""
    stubs: List[StubType] = []
    root = pathlib.Path(stubs_dir)
    if not root.is_dir():
        logger.warning("Stubs directory does not exist: %s", stubs_dir)
        return stubs
    for pyi_path in root.rglob("*.pyi"):
        rel = pyi_path.relative_to(root)
        module = str(rel.with_suffix("")).replace(os.sep, ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        try:
            tree = ast.parse(pyi_path.read_text(encoding="utf-8"), filename=str(pyi_path))
        except (SyntaxError, OSError) as exc:
            logger.debug("Skipping stub %s: %s", pyi_path, exc)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ret = ast.dump(node.returns) if node.returns else "None"
                stubs.append(StubType(
                    module=module,
                    qualified_name=f"{module}.{node.name}",
                    annotation=ret,
                    line=node.lineno,
                ))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                ann = ast.dump(node.annotation) if node.annotation else "Any"
                stubs.append(StubType(
                    module=module,
                    qualified_name=f"{module}.{node.target.id}",
                    annotation=ann,
                    line=node.lineno,
                ))
    return stubs


def _detect_typeshed() -> Optional[str]:
    """Try to find typeshed stubs bundled with the Python installation."""
    # mypy bundles typeshed; also check the stdlib pyi path
    for candidate in [
        pathlib.Path(sys.prefix) / "lib" / "mypy" / "typeshed",
        pathlib.Path(sys.prefix) / "lib" / "typeshed",
        pathlib.Path(shutil.which("mypy") or "/nonexistent").parent.parent / "lib" / "mypy" / "typeshed",
    ]:
        if candidate.is_dir():
            return str(candidate)
    return None


@dataclass
class CheckerIssue:
    """An issue imported from an external type checker (mypy / pyright)."""
    file: str
    line: int
    message: str
    error_code: str
    source: str  # "mypy" or "pyright"
    severity: str = "error"

    def fingerprint(self) -> str:
        raw = f"{self.source}:{self.file}:{self.line}:{self.error_code}:{self.message}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class BaselineComparison:
    """Result of comparing refinement type results against a checker baseline."""
    confirmed: List[CheckerIssue] = field(default_factory=list)
    false_positives: List[CheckerIssue] = field(default_factory=list)
    new_issues: List[Bug] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            f"  Checker errors confirmed   : {len(self.confirmed)}",
            f"  Checker false positives     : {len(self.false_positives)}",
            f"  New issues (refinement type): {len(self.new_issues)}",
        ]
        return "\n".join(lines)


def _parse_mypy_text(text: str) -> List[CheckerIssue]:
    """Parse mypy's default text output: file:line: error: message [code]."""
    issues: List[CheckerIssue] = []
    pattern = re.compile(
        r"^(?P<file>[^:]+):(?P<line>\d+):\s*(?P<sev>error|warning|note):\s*"
        r"(?P<msg>.+?)(?:\s+\[(?P<code>[^\]]+)\])?\s*$"
    )
    for raw_line in text.splitlines():
        m = pattern.match(raw_line.strip())
        if m:
            issues.append(CheckerIssue(
                file=m.group("file"),
                line=int(m.group("line")),
                message=m.group("msg"),
                error_code=m.group("code") or "unknown",
                source="mypy",
                severity=m.group("sev"),
            ))
    return issues


def _parse_mypy_json(data: Any) -> List[CheckerIssue]:
    """Parse mypy's JSON output (list of dicts with file, line, message, code)."""
    issues: List[CheckerIssue] = []
    items = data if isinstance(data, list) else data.get("errors", data.get("messages", []))
    for item in items:
        if not isinstance(item, dict):
            continue
        issues.append(CheckerIssue(
            file=item.get("file", "<unknown>"),
            line=int(item.get("line", 0)),
            message=item.get("message", ""),
            error_code=item.get("code", item.get("error_code", "unknown")),
            source="mypy",
            severity=item.get("severity", "error"),
        ))
    return issues


def _parse_mypy_baseline(path: str) -> List[CheckerIssue]:
    """Load a mypy baseline file (text or JSON format)."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    text_stripped = text.strip()
    if text_stripped.startswith(("{", "[")):
        try:
            data = json.loads(text_stripped)
            return _parse_mypy_json(data)
        except json.JSONDecodeError:
            pass
    return _parse_mypy_text(text)


def _parse_pyright_baseline(path: str) -> List[CheckerIssue]:
    """Parse pyright's JSON output format."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    data = json.loads(text)
    issues: List[CheckerIssue] = []
    diagnostics = data.get("generalDiagnostics", data.get("diagnostics", []))
    for diag in diagnostics:
        if not isinstance(diag, dict):
            continue
        rng = diag.get("range", {}).get("start", {})
        issues.append(CheckerIssue(
            file=diag.get("file", diag.get("uri", "<unknown>")),
            line=int(rng.get("line", 0)) + 1,  # pyright uses 0-based lines
            message=diag.get("message", ""),
            error_code=diag.get("rule", diag.get("code", "unknown")),
            source="pyright",
            severity=diag.get("severity", "error"),
        ))
    return issues


def _compare_checker_baseline(
    checker_issues: List[CheckerIssue],
    analysis_results: List[AnalysisResult],
) -> BaselineComparison:
    """Compare external checker issues against reftype analysis results."""
    result_bugs_by_loc: Dict[str, List[Bug]] = {}
    for r in analysis_results:
        for b in r.bugs:
            key = f"{b.location.file}:{b.location.line}"
            result_bugs_by_loc.setdefault(key, []).append(b)

    confirmed: List[CheckerIssue] = []
    false_positives: List[CheckerIssue] = []
    matched_bug_fps: Set[str] = set()

    for issue in checker_issues:
        key = f"{issue.file}:{issue.line}"
        bugs_at_loc = result_bugs_by_loc.get(key, [])
        if bugs_at_loc:
            confirmed.append(issue)
            for b in bugs_at_loc:
                matched_bug_fps.add(b.fingerprint())
        else:
            false_positives.append(issue)

    new_issues: List[Bug] = []
    for r in analysis_results:
        for b in r.bugs:
            if b.fingerprint() not in matched_bug_fps:
                new_issues.append(b)

    return BaselineComparison(
        confirmed=confirmed,
        false_positives=false_positives,
        new_issues=new_issues,
    )


class PackageAnalyzeCommand:
    """Analyze an entire Python package or directory for refinement type bugs.

    Recursively discovers .py files, respects .gitignore, processes all
    files, and aggregates results with a summary.
    """

    # Directories always skipped
    _SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules",
                  ".mypy_cache", ".pytest_cache", "dist", "build", "*.egg-info"}

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "directory", nargs="?", default=".",
            help="Directory or Python package to analyze (default: current dir)",
        )
        parser.add_argument(
            "--requirements", metavar="FILE",
            help="Path to requirements.txt or pyproject.toml for library type stubs",
        )
        parser.add_argument(
            "--output-format", choices=["text", "json", "sarif"], default="text",
            help="Output format (default: text)",
        )
        parser.add_argument(
            "-o", "--output", help="Output file (default: stdout)",
        )
        parser.add_argument(
            "-v", "--verbose", action="count", default=0,
        )
        parser.add_argument(
            "--include", nargs="*", default=["**/*.py"],
            help="Glob patterns to include (default: **/*.py)",
        )
        parser.add_argument(
            "--exclude", nargs="*", default=[],
            help="Additional glob patterns to exclude",
        )
        parser.add_argument(
            "-w", "--workers", type=int, default=0,
            help="Parallel workers (0 = auto)",
        )
        parser.add_argument(
            "--timeout", type=float, default=300.0,
            help="Per-file timeout in seconds",
        )
        parser.add_argument(
            "-c", "--config", help="Config file path",
        )
        parser.add_argument(
            "--stubs-dir", metavar="PATH",
            help="Directory containing .pyi stub files to use as known types",
        )
        parser.add_argument(
            "--mypy-baseline", metavar="FILE",
            help="mypy output file (text or JSON) for baseline comparison",
        )
        parser.add_argument(
            "--pyright-baseline", metavar="FILE",
            help="pyright JSON output file for baseline comparison",
        )

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))

        directory = pathlib.Path(getattr(args, "directory", ".")).resolve()
        if not directory.is_dir():
            sys.stderr.write(f"Not a directory: {directory}\n")
            return 2

        # Build exclusion patterns
        exclude = list(self._SKIP_DIRS) + (getattr(args, "exclude", None) or [])
        include = getattr(args, "include", None) or ["**/*.py"]

        # Discover files
        discovery = FileDiscovery(include=include, exclude=exclude)
        files = discovery.discover([str(directory)])

        if not files:
            sys.stderr.write("No Python files found in directory.\n")
            return 0

        sys.stderr.write(f"Discovered {len(files)} Python file(s) in {directory}\n")

        # Parse requirements if provided
        req_info: Optional[Dict[str, Any]] = None
        req_path_str = getattr(args, "requirements", None)
        if req_path_str:
            req_path = pathlib.Path(req_path_str)
            if req_path.name == "pyproject.toml":
                dep_names = _parse_pyproject_dependencies(req_path)
            else:
                dep_names = _parse_requirements_txt(req_path)

            stubs_available: List[Dict[str, str]] = []
            for dep in dep_names:
                if dep in _KNOWN_LIBRARY_STUBS:
                    stubs_available.append(
                        {"package": dep, **_KNOWN_LIBRARY_STUBS[dep]}
                    )

            req_info = {
                "source": str(req_path),
                "dependencies": dep_names,
                "known_stubs": stubs_available,
            }
            sys.stderr.write(
                f"Parsed {len(dep_names)} dependencies from {req_path.name}, "
                f"{len(stubs_available)} with known type stubs\n"
            )

        # Load .pyi stubs if provided
        known_stubs: List[StubType] = []
        stubs_dir = getattr(args, "stubs_dir", None)
        if stubs_dir:
            known_stubs = _scan_pyi_stubs(stubs_dir)
            sys.stderr.write(f"Loaded {len(known_stubs)} type annotations from stubs\n")
        typeshed = _detect_typeshed()
        if typeshed and not stubs_dir:
            logger.debug("Auto-detected typeshed at %s", typeshed)

        # Load config and run analysis
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)

        executor = ParallelExecutor(
            workers=cfg.effective_workers(),
            chunk_size=cfg.parallel.chunk_size,
            timeout_per_file=cfg.parallel.timeout_per_file,
            language=cfg.language,
        )

        progress = ProgressReporter(total=len(files))
        results = executor.execute(files, progress)
        progress.finish()

        summary = AnalysisSummary()
        for r in results:
            summary.merge(r)

        # Compute refinement stats
        total_types_inferred = sum(len(r.contracts) for r in results)
        total_refinements = sum(
            sum(len(c.preconditions) + len(c.postconditions) for c in r.contracts)
            for r in results
        )

        # Checker baseline comparisons (mypy / pyright)
        checker_comparison: Optional[BaselineComparison] = None
        mypy_bl = getattr(args, "mypy_baseline", None)
        pyright_bl = getattr(args, "pyright_baseline", None)
        if mypy_bl:
            mypy_issues = _parse_mypy_baseline(mypy_bl)
            checker_comparison = _compare_checker_baseline(mypy_issues, results)
            sys.stderr.write(f"\n── mypy baseline comparison ({len(mypy_issues)} issues) ──\n")
            sys.stderr.write(checker_comparison.summary_text() + "\n")
        if pyright_bl:
            pyright_issues = _parse_pyright_baseline(pyright_bl)
            checker_comparison = _compare_checker_baseline(pyright_issues, results)
            sys.stderr.write(f"\n── pyright baseline comparison ({len(pyright_issues)} issues) ──\n")
            sys.stderr.write(checker_comparison.summary_text() + "\n")

        # Output
        out_format = getattr(args, "output_format", "text")
        output_path = getattr(args, "output", None)

        content: str
        if out_format == "json":
            data = JsonOutputGenerator().generate(results, summary)
            data["package_directory"] = str(directory)
            if req_info:
                data["requirements"] = req_info
            content = json.dumps(data, indent=2)
        elif out_format == "sarif":
            sarif = SarifGenerator().generate(results)
            content = json.dumps(sarif, indent=2)
        else:
            # Text summary
            lines: List[str] = []
            lines.append("")
            lines.append("═══ Package Analysis Summary ═══")
            lines.append(f"  Directory        : {directory}")
            lines.append(f"  Files analyzed   : {summary.total_files}")
            lines.append(f"  Functions        : {summary.total_functions}")
            lines.append(f"  Types inferred   : {total_types_inferred}")
            lines.append(f"  Refinements found: {total_refinements}")
            lines.append(f"  Bugs found       : {summary.total_bugs}")
            if summary.bugs_by_severity:
                for sev, cnt in sorted(summary.bugs_by_severity.items()):
                    lines.append(f"    {sev:10s}: {cnt}")
            if summary.bugs_by_category:
                lines.append("  By category:")
                for cat, cnt in sorted(summary.bugs_by_category.items(), key=lambda x: -x[1]):
                    lines.append(f"    {cat:25s}: {cnt}")
            lines.append(f"  CEGAR iterations : {summary.total_cegar_iterations}")
            lines.append(f"  Duration         : {summary.duration_ms:.0f}ms")
            if req_info:
                lines.append(f"  Dependencies     : {len(req_info['dependencies'])}")
                lines.append(f"  Known type stubs : {len(req_info['known_stubs'])}")
            if summary.files_timed_out:
                lines.append(f"  Timed-out files  : {summary.files_timed_out}")
            lines.append("")

            # Per-file details in verbose mode
            if getattr(args, "verbose", 0) >= 1:
                printer = ResultPrinter(stream=sys.stderr, color=True)
                for r in results:
                    printer.print_result(r)

            content = "\n".join(lines)

        if output_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(content)
            sys.stderr.write(f"Output written to {output_path}\n")
        else:
            sys.stdout.write(content + "\n")

        return 0 if summary.total_bugs == 0 else 1


# ── VerifyCommand ─────────────────────────────────────────────────────────


class VerifyCommand:
    """Verify nn.Module architecture via constraint-based verification.

    Extracts the computation graph from an nn.Module class, then verifies
    shape compatibility, device consistency, and gradient flow using Z3-backed
    symbolic constraint propagation. Produces either a safety verification condition or a counterexample trace.
    """

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Python file containing nn.Module class")
        parser.add_argument(
            "--input-shape", "-s", action="append", default=[],
            help="Input shape as name=dim1,dim2,... (e.g., x=batch,3,224,224). "
            "When omitted, shapes are auto-inferred from forward annotations, "
            "docstrings, example inputs, or the first rank-determining layer."
        )
        parser.add_argument(
            "--no-infer", action="store_true",
            help="Disable automatic input-shape inference (treat unspecified "
            "inputs as fully symbolic, as in pre-Step-56 behavior)."
        )
        parser.add_argument(
            "--no-device-check", action="store_true",
            help="Disable device consistency checking"
        )
        parser.add_argument(
            "--no-phase-check", action="store_true",
            help="Disable train/eval phase checking"
        )
        parser.add_argument(
            "--no-grad-check", action="store_true",
            help="Disable gradient-flow checking"
        )
        parser.add_argument(
            "--cegar-iterations", type=int, default=10,
            help="Max CEGAR refinement iterations (default: 10)"
        )
        parser.add_argument(
            "--format", "-f", choices=["text", "json", "sarif"], default="text",
            help="Output format"
        )
        parser.add_argument(
            "--no-color", action="store_true",
            help="Disable ANSI color in text diagnostics."
        )
        parser.add_argument(
            "--explain", action="store_true",
            help="Print the inference chain (input -> op -> shape -> ... -> "
            "failing step) that led to each reported bug."
        )
        parser.add_argument(
            "--fix", action="store_true",
            help="Print mechanical autofix suggestions (concrete single-line "
            "edits) for repairable bugs such as a wrong nn.Linear in_features."
        )
        parser.add_argument(
            "--write", action="store_true",
            help="With --fix, apply the suggested edits to the file in place."
        )
        parser.add_argument(
            "--lsp", action="store_true",
            help="Emit an editor-ready LSP report (diagnostics/squiggles, hover "
            "shapes, and quick-fix code actions) as JSON for a VSCode/LSP "
            "extension to consume."
        )
        parser.add_argument(
            "--watch", action="store_true",
            help="Re-verify the file every time it changes (live feedback). "
            "Also watches sibling .py files in the same directory. Ctrl-C to stop."
        )
        parser.add_argument(
            "--debounce", type=float, default=0.4,
            help="Polling interval in seconds for --watch (default: 0.4)."
        )
        parser.add_argument(
            "--high-confidence", action="store_true",
            help="Only report high-confidence (Z3-proven) bugs. Reduces FP rate to 0%% for CI/CD gating."
        )
        parser.add_argument(
            "--soundness-mode", choices=["sound", "balanced", "heuristic"],
            default=None,
            help="Verdict strictness (Step 7). 'sound' emits SAFE only when the "
            "module is fully in the verifiable fragment (else UNKNOWN, exit 2); "
            "'balanced' (default) abstains on opaque layers; 'heuristic' "
            "tolerates abstention. Does not change which bugs are reported. "
            "Overrides any tensorguard.toml setting."
        )
        parser.add_argument(
            "--config", default=None,
            help="Path to a tensorguard.toml config file (default: auto-discover "
            "by walking up from the file under analysis)."
        )
        parser.add_argument(
            "--no-config", action="store_true",
            help="Ignore any tensorguard.toml / pyproject [tool.tensorguard] config."
        )

    def execute(self, args: argparse.Namespace) -> int:
        if getattr(args, "lsp", False):
            return self._emit_lsp(args)
        if getattr(args, "watch", False):
            return self._watch(args)
        return self._verify_once(args)

    def _emit_lsp(self, args: argparse.Namespace) -> int:
        """Print the editor-ready LSP report (diagnostics/hover/code actions)."""
        filepath = pathlib.Path(args.file)
        if not filepath.exists():
            sys.stderr.write(f"File not found: {args.file}\n")
            return 1
        try:
            result = self._verify_value(str(filepath), args)
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            return 1
        from src.lsp_provider import build_lsp_report
        uri = filepath.resolve().as_uri()
        report = build_lsp_report(result, uri)
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        return 1 if result.bugs else 0

    def _watch(self, args: argparse.Namespace) -> int:
        """Re-verify ``args.file`` (and sibling .py files) on every change."""
        from src.watch_mode import (
            format_watch_result, poll_once, run_verification, snapshot_mtimes,
        )

        filepath = pathlib.Path(args.file)
        if not filepath.exists():
            sys.stderr.write(f"File not found: {args.file}\n")
            return 1

        # Watch the target plus its sibling .py files (common multi-file models).
        watched = [str(filepath)]
        try:
            for sib in sorted(filepath.parent.glob("*.py")):
                if str(sib) != str(filepath):
                    watched.append(str(sib))
        except Exception:
            pass

        debounce = float(getattr(args, "debounce", 0.4) or 0.4)
        use_color = not getattr(args, "no_color", False) and sys.stdout.isatty()

        sys.stderr.write(
            f"Watching {filepath.name} (and {len(watched) - 1} sibling file(s)); "
            f"Ctrl-C to stop.\n"
        )

        # Initial pass so the developer sees the current state immediately.
        wr = run_verification(str(filepath), lambda p: self._verify_value(p, args))
        sys.stdout.write(format_watch_result(wr, use_color) + "\n")
        sys.stdout.flush()

        mtimes = snapshot_mtimes(watched)
        try:
            while True:
                changed, mtimes = poll_once(watched, mtimes)
                if changed:
                    stamp = time.strftime("%H:%M:%S")
                    sys.stdout.write(
                        f"\n[{stamp}] change detected; re-verifying...\n"
                    )
                    wr = run_verification(
                        str(filepath), lambda p: self._verify_value(p, args)
                    )
                    sys.stdout.write(format_watch_result(wr, use_color) + "\n")
                    sys.stdout.flush()
                time.sleep(debounce)
        except KeyboardInterrupt:
            sys.stderr.write("\nWatch stopped.\n")
        return 0

    def _resolve_config(self, args: argparse.Namespace, path: str):
        """Load tensorguard.toml (unless --no-config) for *path*."""
        from src.tg_config import TGConfig, load_tg_config
        if getattr(args, "no_config", False):
            return TGConfig()
        return load_tg_config(path, explicit_path=getattr(args, "config", None))

    def _effective_verify_kwargs(self, args: argparse.Namespace, cfg) -> dict:
        """Merge CLI flags over config defaults (CLI wins).

        The CLI ``--no-*`` flags can only *disable* a check, so the effective
        value is ``config_value and not cli_disabled``.  ``--soundness-mode`` and
        explicit ``--cegar-iterations`` override the config; otherwise the config
        value (or the built-in default) applies.
        """
        soundness = getattr(args, "soundness_mode", None)
        if soundness is None:
            soundness = cfg.soundness_mode or "balanced"
        cegar = getattr(args, "cegar_iterations", 10)
        if cegar == 10 and cfg.cegar_iterations:
            cegar = cfg.cegar_iterations
        kwargs = {
            "check_devices": cfg.check_devices and not args.no_device_check,
            "check_phases": cfg.check_phases and not args.no_phase_check,
            "check_gradients": cfg.check_gradients and not args.no_grad_check,
            "max_cegar_iterations": cegar,
            "high_confidence_only": args.high_confidence or cfg.high_confidence,
            "soundness_mode": soundness,
            "infer_inputs": cfg.infer_inputs and not getattr(args, "no_infer", False),
        }
        if getattr(cfg, "max_loop_unrolls", None) is not None:
            kwargs["max_loop_unrolls"] = cfg.max_loop_unrolls
        return kwargs

    def _verify_value(self, path: str, args: argparse.Namespace):
        """Verify *path* and return the AnalysisResult (used by watch mode)."""
        from src.api import verify_architecture
        from src.tg_config import filter_result
        source = pathlib.Path(path).read_text(encoding="utf-8")
        input_shapes = self._parse_input_shapes(args.input_shape)
        cfg = self._resolve_config(args, path)
        result = verify_architecture(
            source,
            input_shapes=input_shapes,
            filename=path,
            **self._effective_verify_kwargs(args, cfg),
        )
        return filter_result(cfg, result)

    @staticmethod
    def _parse_input_shapes(specs) -> Dict[str, tuple]:
        input_shapes: Dict[str, tuple] = {}
        for spec in specs:
            if "=" not in spec:
                continue
            name, dims_str = spec.split("=", 1)
            dims = []
            for d in dims_str.split(","):
                d = d.strip()
                try:
                    dims.append(int(d))
                except ValueError:
                    dims.append(d)
            input_shapes[name] = tuple(dims)
        return input_shapes

    def _verify_once(self, args: argparse.Namespace) -> int:
        filepath = pathlib.Path(args.file)
        if not filepath.exists():
            sys.stderr.write(f"File not found: {args.file}\n")
            return 1

        try:
            source = filepath.read_text(encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Cannot read file: {e}\n")
            return 1

        # Parse input shapes
        input_shapes: Dict[str, tuple] = {}
        for spec in args.input_shape:
            if "=" not in spec:
                sys.stderr.write(f"Invalid shape spec: {spec} (use name=d1,d2,...)\n")
                return 1
            name, dims_str = spec.split("=", 1)
            dims = []
            for d in dims_str.split(","):
                d = d.strip()
                try:
                    dims.append(int(d))
                except ValueError:
                    dims.append(d)  # symbolic dim
            input_shapes[name] = tuple(dims)

        # Load per-repo configuration and honor file ignores up-front.
        cfg = self._resolve_config(args, str(filepath))
        from src.tg_config import is_ignored_file
        if is_ignored_file(cfg, str(filepath)):
            sys.stdout.write(
                f"- {filepath.name}: ignored by tensorguard config\n"
            )
            return 0

        try:
            from src.api import verify_architecture
            from src.tg_config import filter_result
            result = verify_architecture(
                source,
                input_shapes=input_shapes,
                filename=str(filepath),
                **self._effective_verify_kwargs(args, cfg),
            )
            result = filter_result(cfg, result)
        except RuntimeError as e:
            sys.stderr.write(f"Error: {e}\n")
            return 1

        fmt = getattr(args, "format", "text")
        verdict = getattr(result, "verdict", "SAFE" if not result.bugs else "UNSAFE")
        unknown_reasons = list(getattr(result, "unknown_reasons", []))
        inferred_shapes = dict(getattr(result, "inferred_input_shapes", {}) or {})
        inferred_sources = dict(getattr(result, "inferred_input_sources", {}) or {})
        if fmt == "json":
            out = {
                "file": str(filepath),
                "bugs": [
                    {"line": b.location.line, "message": b.message, "severity": b.severity}
                    for b in result.bugs
                ],
                "duration_ms": result.duration_ms,
                "status": "SAFE" if not result.bugs else "UNSAFE",
                "verdict": verdict,
                "soundness_mode": getattr(result, "soundness_mode", "balanced"),
                "abstained": bool(getattr(result, "abstained", False)),
                "opaque_layer_count": int(getattr(result, "opaque_layer_count", 0)),
                "unknown_reasons": unknown_reasons,
                "inferred_input_shapes": {
                    name: list(shape) for name, shape in inferred_shapes.items()
                },
                "inferred_input_sources": inferred_sources,
            }
            if getattr(args, "explain", False):
                chain = getattr(result, "inference_chain", None)
                if chain is not None:
                    out["inference_chain"] = {
                        "model_name": chain.model_name,
                        "failing_step": chain.failing_step,
                        "summary": chain.summary,
                        "concrete_dims": chain.concrete_dims,
                        "links": [
                            {
                                "step_index": l.step_index,
                                "op": l.op,
                                "layer": l.layer,
                                "line": l.line,
                                "inputs": l.inputs,
                                "input_shapes": l.input_shapes,
                                "output": l.output,
                                "output_shape": l.output_shape,
                                "is_failing": l.is_failing,
                                "expected_shape": l.expected_shape,
                                "actual_shape": l.actual_shape,
                            }
                            for l in chain.links
                        ],
                    }
            if getattr(args, "fix", False):
                fixes = list(getattr(result, "autofixes", []) or [])
                out["autofixes"] = [
                    {
                        "layer": f.layer,
                        "kind": f.kind,
                        "line": f.line,
                        "original": f.original,
                        "suggested": f.suggested,
                        "description": f.description,
                        "old_value": f.old_value,
                        "new_value": f.new_value,
                    }
                    for f in fixes
                ]
            sys.stdout.write(json.dumps(out, indent=2) + "\n")
        elif fmt == "text":
            if inferred_shapes and not args.input_shape:
                sys.stdout.write("Auto-inferred input shapes (no -s given):\n")
                for name in sorted(inferred_shapes):
                    dims = ", ".join(str(d) for d in inferred_shapes[name])
                    src_label = inferred_sources.get(name, "inferred")
                    sys.stdout.write(
                        f"  {name} = ({dims})  [from {src_label}]\n"
                    )
            if not result.bugs:
                if verdict == "UNKNOWN":
                    sys.stdout.write(
                        f"? {filepath.name}: Cannot certify safe — UNKNOWN "
                        f"({result.duration_ms:.1f}ms)\n"
                    )
                    for reason in unknown_reasons:
                        sys.stdout.write(f"  - {reason}\n")
                else:
                    sys.stdout.write(
                        f"✓ {filepath.name}: Architecture verified safe "
                        f"({result.duration_ms:.1f}ms)\n"
                    )
                    # Show discovered contracts if any
                    contracts = getattr(result, "_shape_contracts", [])
                    if contracts:
                        sys.stdout.write(f"  Discovered {len(contracts)} shape contracts:\n")
                        for c in contracts[:5]:
                            sys.stdout.write(f"    {c}\n")
            else:
                diagnostics = list(getattr(result, "diagnostics", []) or [])
                count = len(diagnostics) if diagnostics else len(result.bugs)
                noun = "issue" if count == 1 else "issues"
                sys.stdout.write(
                    f"✗ {filepath.name}: {count} verification {noun} "
                    f"({result.duration_ms:.1f}ms)\n"
                )
                if diagnostics:
                    from src.source_mapped_errors import format_ansi, format_plain
                    use_color = (
                        not getattr(args, "no_color", False)
                        and sys.stdout.isatty()
                    )
                    rendered = (
                        format_ansi(diagnostics) if use_color
                        else format_plain(diagnostics)
                    )
                    for ln in rendered.split("\n"):
                        sys.stdout.write(f"  {ln}\n")
                else:
                    for b in result.bugs:
                        sys.stdout.write(
                            f"  L{b.location.line}: {b.message}\n"
                        )
                if getattr(args, "explain", False):
                    chain = getattr(result, "inference_chain", None)
                    if chain is not None:
                        from src.inference_chain import (
                            format_chain_ansi, format_chain_plain,
                        )
                        use_color = (
                            not getattr(args, "no_color", False)
                            and sys.stdout.isatty()
                        )
                        rendered = (
                            format_chain_ansi(chain) if use_color
                            else format_chain_plain(chain)
                        )
                        sys.stdout.write("\n")
                        for ln in rendered.split("\n"):
                            sys.stdout.write(f"  {ln}\n")
                if getattr(args, "fix", False):
                    fixes = list(getattr(result, "autofixes", []) or [])
                    if fixes:
                        from src.autofix import (
                            format_autofixes_ansi, format_autofixes_plain,
                        )
                        use_color = (
                            not getattr(args, "no_color", False)
                            and sys.stdout.isatty()
                        )
                        rendered = (
                            format_autofixes_ansi(fixes) if use_color
                            else format_autofixes_plain(fixes)
                        )
                        sys.stdout.write("\n")
                        for ln in rendered.split("\n"):
                            sys.stdout.write(f"  {ln}\n")
        else:
            sarif = result.to_sarif()
            sys.stdout.write(json.dumps(sarif, indent=2) + "\n")

        # Step 59 — optionally apply the mechanical fixes in place.
        if getattr(args, "fix", False) and getattr(args, "write", False):
            fixes = list(getattr(result, "autofixes", []) or [])
            if fixes:
                from src.autofix import apply_autofixes
                new_source = apply_autofixes(source, fixes)
                if new_source != source:
                    try:
                        filepath.write_text(new_source, encoding="utf-8")
                        sys.stdout.write(
                            f"Applied {len(fixes)} fix"
                            f"{'' if len(fixes) == 1 else 'es'} to "
                            f"{filepath.name}\n"
                        )
                    except Exception as e:
                        sys.stderr.write(f"Could not write fixes: {e}\n")

        if result.bugs:
            return 1
        # Sound mode is an opt-in CI gate: an UNKNOWN verdict cannot certify
        # safety, so it fails with a distinct exit code. Other modes preserve
        # the legacy "no bugs → exit 0" behavior for backward compatibility.
        if verdict == "UNKNOWN" and getattr(args, "soundness_mode", "balanced") == "sound":
            return 2
        return 0


# ── ExplainCommand ─────────────────────────────────────────────────────────


class ExplainCommand:
    """Generate a self-contained HTML explanation report for one nn.Module file."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Python file containing nn.Module class")
        parser.add_argument(
            "--input-shape", "-s", action="append", default=[],
            help="Input shape as name=dim1,dim2,... (e.g., x=batch,3,224,224).",
        )
        parser.add_argument("-o", "--output", help="Write HTML report to this path")
        parser.add_argument("--no-infer", action="store_true", help="Disable automatic input-shape inference")
        parser.add_argument("--no-device-check", action="store_true", help="Disable device consistency checking")
        parser.add_argument("--no-phase-check", action="store_true", help="Disable train/eval phase checking")
        parser.add_argument("--no-grad-check", action="store_true", help="Disable gradient-flow checking")
        parser.add_argument("--cegar-iterations", type=int, default=10, help="Max CEGAR refinement iterations")
        parser.add_argument("--high-confidence", action="store_true", help="Only report high-confidence bugs")
        parser.add_argument(
            "--soundness-mode", choices=["sound", "balanced", "heuristic"],
            default=None,
            help="Verdict strictness; sound mode exits 2 on UNKNOWN.",
        )
        parser.add_argument("--config", default=None, help="Path to tensorguard.toml")
        parser.add_argument("--no-config", action="store_true", help="Ignore TensorGuard config files")

    @staticmethod
    def _parse_input_shapes_strict(specs: Sequence[str]) -> Optional[Dict[str, tuple]]:
        input_shapes: Dict[str, tuple] = {}
        for spec in specs:
            if "=" not in spec:
                sys.stderr.write(f"Invalid shape spec: {spec} (use name=d1,d2,...)\n")
                return None
            name, dims_str = spec.split("=", 1)
            dims = []
            for d in dims_str.split(","):
                d = d.strip()
                try:
                    dims.append(int(d))
                except ValueError:
                    dims.append(d)
            input_shapes[name] = tuple(dims)
        return input_shapes

    def execute(self, args: argparse.Namespace) -> int:
        filepath = pathlib.Path(args.file)
        if not filepath.exists():
            sys.stderr.write(f"File not found: {args.file}\n")
            return 1
        try:
            source = filepath.read_text(encoding="utf-8")
        except Exception as exc:
            sys.stderr.write(f"Cannot read file: {exc}\n")
            return 1

        input_shapes = self._parse_input_shapes_strict(getattr(args, "input_shape", []) or [])
        if input_shapes is None:
            return 1

        verify_cmd = VerifyCommand()
        cfg = verify_cmd._resolve_config(args, str(filepath))
        from src.tg_config import filter_result, is_ignored_file
        if is_ignored_file(cfg, str(filepath)):
            sys.stderr.write(f"{filepath} is ignored by TensorGuard config; no report generated.\n")
            return 0

        try:
            from src.api import verify_architecture
            result = verify_architecture(
                source,
                input_shapes=input_shapes,
                filename=str(filepath),
                **verify_cmd._effective_verify_kwargs(args, cfg),
            )
            result = filter_result(cfg, result)
        except RuntimeError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1

        from src.inference_chain import format_explain_html, write_explain_html
        title = f"TensorGuard explain: {filepath.name}"
        output_path = getattr(args, "output", None)
        if output_path:
            written = write_explain_html(output_path, result, source=source, title=title)
            sys.stdout.write(f"TensorGuard explain report written to {written}\n")
        else:
            sys.stdout.write(format_explain_html(result, source=source, title=title))

        verdict = getattr(result, "verdict", "SAFE" if not result.bugs else "UNSAFE")
        if result.bugs:
            return 1
        if verdict == "UNKNOWN" and getattr(result, "soundness_mode", "balanced") == "sound":
            return 2
        return 0


# ── ModelHubBadgeCommand ───────────────────────────────────────────────────


class ModelHubBadgeCommand:
    """Write a TensorGuard-verified model-hub badge certificate bundle."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Python file containing nn.Module class")
        parser.add_argument(
            "--input-shape",
            "-s",
            action="append",
            default=[],
            help="Input shape as name=dim1,dim2,... (repeatable).",
        )
        parser.add_argument("--model-id", required=True, help="Model hub id, e.g. org/model")
        parser.add_argument(
            "-o",
            "--output",
            default="tensorguard_verified",
            help="Output directory for manifest, badge SVG, certificate, and model-card snippet",
        )
        parser.add_argument(
            "--secret-env",
            default="TENSORGUARD_CERT_SECRET",
            help="Environment variable containing the signing secret",
        )
        parser.add_argument(
            "--secret",
            default=None,
            help="Signing secret value (prefer --secret-env outside tests).",
        )
        parser.add_argument(
            "--issued-at",
            default="1970-01-01T00:00:00+00:00",
            help="Certificate issue timestamp; fixed by default for reproducible bundles.",
        )
        parser.add_argument("--issuer", default="tensorguard-model-hub")
        parser.add_argument("--key-id", default=None)
        parser.add_argument("--cegar-iterations", type=int, default=10)
        parser.add_argument("--no-infer", action="store_true", help="Disable input-shape inference")
        parser.add_argument(
            "--include-proof",
            action="store_true",
            help="Embed the solver proof DAG in the signed certificate.",
        )
        parser.add_argument("--json", action="store_true", dest="as_json")

    @staticmethod
    def _parse_input_shapes(specs: Sequence[str]) -> Optional[Dict[str, tuple]]:
        input_shapes: Dict[str, tuple] = {}
        for spec in specs:
            if "=" not in spec:
                sys.stderr.write(f"Invalid shape spec: {spec} (use name=d1,d2,...)\n")
                return None
            name, dims_str = spec.split("=", 1)
            dims = []
            for d in dims_str.split(","):
                d = d.strip()
                try:
                    dims.append(int(d))
                except ValueError:
                    dims.append(d)
            input_shapes[name] = tuple(dims)
        return input_shapes

    def execute(self, args: argparse.Namespace) -> int:
        filepath = pathlib.Path(args.file)
        if not filepath.exists():
            sys.stderr.write(f"File not found: {args.file}\n")
            return 1
        secret = getattr(args, "secret", None)
        if secret is None:
            secret = os.environ.get(getattr(args, "secret_env", "TENSORGUARD_CERT_SECRET"))
        if not secret:
            sys.stderr.write("Signing secret missing; set --secret or --secret-env.\n")
            return 2
        input_shapes = self._parse_input_shapes(getattr(args, "input_shape", []) or [])
        if input_shapes is None:
            return 1
        try:
            source = filepath.read_text(encoding="utf-8")
            from src.model_hub_badge import write_model_hub_badge_bundle

            bundle = write_model_hub_badge_bundle(
                source,
                input_shapes=input_shapes,
                output_dir=getattr(args, "output", "tensorguard_verified"),
                model_id=getattr(args, "model_id"),
                secret=secret,
                filename=str(filepath),
                issued_at=getattr(args, "issued_at"),
                issuer=getattr(args, "issuer"),
                key_id=getattr(args, "key_id", None),
                max_cegar_iterations=getattr(args, "cegar_iterations", 10),
                infer_inputs=not getattr(args, "no_infer", False),
                include_proof=getattr(args, "include_proof", False),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"Could not write TensorGuard model-hub badge: {exc}\n")
            return 1
        if getattr(args, "as_json", False):
            sys.stdout.write(json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(
                "TensorGuard-verified model-hub bundle written to "
                f"{bundle.output_dir}\n"
            )
            sys.stdout.write(f"Model-card snippet: {bundle.model_card_snippet_path}\n")
            sys.stdout.write(f"Badge: {bundle.badge_markdown}\n")
        return 0


# ── PlaygroundCommand ──────────────────────────────────────────────────────


class PlaygroundCommand:
    """Generate the no-upload local TensorGuard playground."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-o",
            "--output",
            default="tensorguard_playground",
            help="Output directory for index.html and manifest.json",
        )
        parser.add_argument(
            "--open",
            action="store_true",
            help="Open the generated index.html in the default browser",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from src.playground import write_playground

        paths = write_playground(getattr(args, "output", "tensorguard_playground"))
        html_path = paths["html"]
        sys.stdout.write(f"TensorGuard local playground written to {html_path}\n")
        sys.stdout.write("Privacy mode: local-static, no upload, no import, no execution.\n")
        if getattr(args, "open", False):
            import webbrowser

            webbrowser.open(html_path.resolve().as_uri())
        return 0


# ── AdoptionRecipesCommand ─────────────────────────────────────────────────


class AdoptionRecipesCommand:
    """Print one-line setup recipes for TensorGuard integrations."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "targets",
            nargs="*",
            help=(
                "Recipe target(s): github-actions, pre-commit, pytest, nox, tox, "
                "makefile, vscode, jetbrains, neovim, jupyter. Omit for all."
            ),
        )
        parser.add_argument("--json", action="store_true", dest="as_json")
        parser.add_argument(
            "--check",
            action="store_true",
            help="Validate that every advertised recipe is backed by repo files.",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from src.setup_recipes import (
            recipe_for,
            recipes,
            render_json,
            render_text,
            validate_recipes,
        )

        if getattr(args, "check", False):
            repo = pathlib.Path(__file__).resolve().parents[2]
            errors = validate_recipes(repo)
            if errors:
                sys.stderr.write("\n".join(errors) + "\n")
                return 1

        targets = list(getattr(args, "targets", []) or [])
        try:
            selected = [recipe_for(t) for t in targets] if targets else recipes()
        except KeyError as exc:
            sys.stderr.write(str(exc) + "\n")
            return 2

        if getattr(args, "as_json", False):
            sys.stdout.write(render_json(selected))
        else:
            sys.stdout.write(render_text(selected))
        return 0


# ── SarifTrendsCommand ──────────────────────────────────────────────────────


class SarifTrendsCommand:
    """Build a Code Scanning trend dashboard from ordered SARIF snapshots."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "snapshots",
            nargs="+",
            help="Ordered release snapshots as RELEASE=path/to/tensorguard.sarif",
        )
        parser.add_argument("-o", "--output", help="Write JSON dashboard here")
        parser.add_argument("-m", "--markdown", help="Write Markdown dashboard here")

    def execute(self, args: argparse.Namespace) -> int:
        from src.sarif_trend_dashboard import load_snapshot, write_dashboard

        try:
            snapshots = [load_snapshot(s) for s in getattr(args, "snapshots", [])]
            dashboard = write_dashboard(
                snapshots,
                json_path=getattr(args, "output", None),
                markdown_path=getattr(args, "markdown", None),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"Could not build SARIF trend dashboard: {exc}\n")
            return 2
        if not getattr(args, "output", None):
            sys.stdout.write(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")
        return 0


# ── UsageMetricsCommand ──────────────────────────────────────────────────────


class UsageMetricsCommand:
    """Summarize TensorGuard JSON reports locally without telemetry."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("reports", nargs="+", help="TensorGuard JSON report(s)")
        parser.add_argument(
            "--format",
            choices=("json", "markdown"),
            default="json",
            help="Output format (default: json)",
        )
        parser.add_argument(
            "--top",
            type=int,
            default=10,
            help="Number of unsupported operators to include",
        )
        parser.add_argument("-o", "--output", help="Write summary to this file")

    def execute(self, args: argparse.Namespace) -> int:
        from src.local_usage_metrics import summarize_files

        try:
            summary = summarize_files(getattr(args, "reports", []), limit=args.top)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"Could not summarize local usage metrics: {exc}\n")
            return 2
        if args.format == "markdown":
            rendered = summary.to_markdown()
        else:
            rendered = json.dumps(summary.to_json_dict(), indent=2, sort_keys=True) + "\n"
        if args.output:
            pathlib.Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0


# ---------------------------------------------------------------------------
# ReftypeCliApp — main application
# ---------------------------------------------------------------------------


class ScanTorchDataCommand:
    """Scan PyTorch data-pipeline source for silent data-misuse bugs."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("path", help="a .py file or directory tree to scan")
        parser.add_argument("--format", choices=("text", "json"), default="text")
        parser.add_argument("--fail-on-finding", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        from src.interface_layer import (
            analyze_torch_data_tree,
            render_torch_data_report_json,
            render_torch_data_report_text,
        )

        try:
            reports = analyze_torch_data_tree(args.path)
        except OSError as exc:
            sys.stderr.write(f"tensorguard: cannot read source: {exc}\n")
            return 2
        count = sum(len(r.findings) for r in reports)
        if args.format == "json":
            sys.stdout.write(render_torch_data_report_json(reports) + "\n")
        else:
            sys.stdout.write(render_torch_data_report_text(reports) + "\n")
        return 1 if (getattr(args, "fail_on_finding", False) and count) else 0


class ProveSurfaceBanCommand:
    """Prove whether an id-level ban prevents a forbidden surface string."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("target", help="the forbidden surface string")
        parser.add_argument("tokenizer_json", help="path to a HF tokenizer.json")
        parser.add_argument("--suppress-id", action="append", type=int, default=[], metavar="ID")
        parser.add_argument("--suppress-substring-tokens", action="store_true")
        parser.add_argument("--bad-word-ids", action="append", default=[], metavar="ID,ID,...")
        parser.add_argument("--format", choices=("text", "json"), default="text")
        parser.add_argument("--fail-on-bypass", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        from src.interface_layer import (
            BanSoundnessStatus,
            load_id_surfaces_from_tokenizer_json,
            naive_substring_suppression,
            prove_surface_ban,
            render_surface_ban_report_json,
            render_surface_ban_report_text,
        )

        try:
            id_surfaces = load_id_surfaces_from_tokenizer_json(args.tokenizer_json)
            suppressed = list(args.suppress_id)
            if args.suppress_substring_tokens:
                suppressed.extend(naive_substring_suppression(id_surfaces, args.target))
            bad_words = [
                [int(x) for x in raw.split(",") if x.strip() != ""]
                for raw in args.bad_word_ids
            ]
            report = prove_surface_ban(
                id_surfaces, args.target, suppressed_ids=suppressed, bad_word_id_seqs=bad_words
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"tensorguard: cannot analyze surface ban: {exc}\n")
            return 2
        if args.format == "json":
            sys.stdout.write(render_surface_ban_report_json(report) + "\n")
        else:
            sys.stdout.write(render_surface_ban_report_text(report) + "\n")
        bypassable = report.status is BanSoundnessStatus.BYPASS_FOUND
        return 1 if (getattr(args, "fail_on_bypass", False) and bypassable) else 0


class ProveStreamingStopCommand:
    """Prove whether a server can truncate output exactly at a stop string."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("stop", help="the stop string")
        parser.add_argument("tokenizer_json", help="path to a HF tokenizer.json")
        parser.add_argument("--format", choices=("text", "json"), default="text")
        parser.add_argument("--fail-on-hazard", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        from src.interface_layer import (
            StopSoundnessStatus,
            load_id_surfaces_from_tokenizer_json,
            prove_streaming_stop,
            render_streaming_stop_report_json,
            render_streaming_stop_report_text,
        )

        try:
            id_surfaces = load_id_surfaces_from_tokenizer_json(args.tokenizer_json)
            report = prove_streaming_stop(id_surfaces, args.stop)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"tensorguard: cannot analyze streaming stop: {exc}\n")
            return 2
        if args.format == "json":
            sys.stdout.write(render_streaming_stop_report_json(report) + "\n")
        else:
            sys.stdout.write(render_streaming_stop_report_text(report) + "\n")
        hazardous = report.status is StopSoundnessStatus.HAZARDS_FOUND
        return 1 if (getattr(args, "fail_on_hazard", False) and hazardous) else 0


class ProveDecodingFeasibilityCommand:
    """Prove whether a guided-decoding grammar admits a tokenization."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("regex", help="the guided_regex grammar (a regular expression)")
        parser.add_argument("tokenizer_json", help="path to a HF tokenizer.json")
        parser.add_argument("--format", choices=("text", "json"), default="text")
        parser.add_argument("--fail-on-infeasible", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        from src.interface_layer import (
            load_vocab_surfaces_from_tokenizer_json,
            prove_decoding_feasibility,
            regex_to_dfa,
            render_decoding_feasibility_report_json,
            render_decoding_feasibility_report_text,
        )

        try:
            vocab = load_vocab_surfaces_from_tokenizer_json(args.tokenizer_json)
            dfa = regex_to_dfa(args.regex, extra_alphabet=set("".join(vocab)), name=args.regex)
            report = prove_decoding_feasibility(dfa, vocab, grammar_name=args.regex)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"tensorguard: cannot analyze decoding feasibility: {exc}\n")
            return 2
        if args.format == "json":
            sys.stdout.write(render_decoding_feasibility_report_json(report) + "\n")
        else:
            sys.stdout.write(render_decoding_feasibility_report_text(report) + "\n")
        infeasible = not getattr(report, "feasible", True)
        return 1 if (getattr(args, "fail_on_infeasible", False) and infeasible) else 0


class SymexecCommand:
    """Run the Python symbolic-execution engine over files/directories.

    ``tensorguard symexec <path> [--engine symexec|fx|both]`` surfaces the
    symexec findings (tuple-unpacking arity, rank-indexing, broadcast/reshape/
    matmul shape faults, …) on their own, with on-demand ``--explain`` provenance
    derivations, the abstain-coverage profile, and the deterministic proof
    fingerprint.  ``--engine fx`` runs only the FX/SMT shape path, ``both`` runs
    both and labels each section."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "paths", nargs="*", default=["."],
            help="Python files or directories to analyze (default: .)",
        )
        parser.add_argument(
            "--engine", choices=["symexec", "fx", "both"], default="symexec",
            help="Which engine(s) to run (default: symexec).",
        )
        parser.add_argument(
            "--format", choices=["text", "json", "sarif", "github", "lsp"],
            default="text", dest="fmt",
            help="Output format (default: text). 'sarif' emits a SARIF 2.1.0 log; "
                 "'github' emits GitHub Actions annotations; 'lsp' emits LSP "
                 "diagnostics JSON (all symexec engine only).",
        )
        parser.add_argument(
            "--explain", action="store_true",
            help="Render the full provenance derivation for each symexec bug.",
        )
        parser.add_argument(
            "--fingerprint", action="store_true",
            help="Print the deterministic proof fingerprint + abstain coverage.",
        )
        parser.add_argument(
            "--coverage", action="store_true",
            help="Print the statement-coverage profile (fraction of statements "
                 "the engine interpreted with a non-Top value).",
        )
        parser.add_argument(
            "--benchmark", action="store_true",
            help="Benchmark per-file analysis latency and print a profile "
                 "(mean/p95/max wall-time + iteration caps) instead of findings.",
        )
        parser.add_argument(
            "--budget-ms", type=float, default=None,
            help="Coarse per-file wall-clock budget in ms; analysis abstains on "
                 "remaining units once exceeded (sound, off by default).",
        )
        parser.add_argument("-o", "--output", help="Write output to this file (default: stdout).")

    @staticmethod
    def _collect_files(paths: Sequence[str]) -> List[pathlib.Path]:
        files: List[pathlib.Path] = []
        seen: set = set()
        for raw in paths:
            p = pathlib.Path(raw)
            if p.is_dir():
                found = sorted(p.rglob("*.py"))
            elif p.is_file():
                found = [p]
            else:
                found = []
            for f in found:
                key = str(f.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(f)
        return files

    def execute(self, args: argparse.Namespace) -> int:
        from src.symexec import analyze_source as symexec_analyze

        paths = getattr(args, "paths", None) or ["."]
        engine = getattr(args, "engine", "symexec")
        fmt = getattr(args, "fmt", "text")
        files = self._collect_files(paths)
        if not files:
            sys.stderr.write("tensorguard symexec: no Python files found\n")
            return 1

        if getattr(args, "benchmark", False):
            return self._run_benchmark(files, getattr(args, "budget_ms", None),
                                       fmt, getattr(args, "output", None))

        budget_ms = getattr(args, "budget_ms", None)
        records: List[dict] = []
        sarif_items: List[tuple] = []
        total_bugs = 0
        for f in files:
            try:
                source = f.read_text(encoding="utf-8")
            except Exception as exc:
                sys.stderr.write(f"tensorguard symexec: cannot read {f}: {exc}\n")
                continue
            fname = str(f)
            rec: dict = {"file": fname, "engines": {}}

            if engine in ("symexec", "both"):
                sr = symexec_analyze(source, filename=fname, budget_ms=budget_ms)
                sarif_items.append((fname, sr))
                rec["engines"]["symexec"] = {
                    "bugs": [b.to_dict() for b in sr.bugs],
                    "fingerprint": sr.fingerprint(),
                    "abstain_coverage": {
                        c.value: n for c, n in sr.abstentions.coverage().items()
                    },
                    "abstain_total": sr.abstentions.total,
                    "coverage": sr.coverage.to_dict(),
                    "explain": sr.explain(filename=fname) if getattr(args, "explain", False) else None,
                }
                total_bugs += len(sr.bugs)

            if engine in ("fx", "both"):
                fx_bugs = self._run_fx(source, fname)
                rec["engines"]["fx"] = {"bugs": fx_bugs}
                total_bugs += len(fx_bugs)

            records.append(rec)

        symexec_only = {"sarif", "github", "lsp"}
        if fmt in symexec_only and engine == "fx":
            sys.stderr.write(
                f"tensorguard symexec: --format {fmt} covers the symexec engine "
                "only; ignoring --engine fx.\n"
            )
        if fmt == "sarif":
            out = self._render_sarif(sarif_items)
        elif fmt == "github":
            out = self._render_github(sarif_items)
        elif fmt == "lsp":
            out = self._render_lsp(sarif_items)
        elif fmt == "json":
            out = self._render_json(records)
        else:
            out = self._render_text(records, engine, getattr(args, "fingerprint", False),
                                    getattr(args, "coverage", False))
        self._write(out, getattr(args, "output", None))
        # Linter convention: non-zero exit when any bug is found.
        return 1 if total_bugs else 0

    @staticmethod
    def _run_benchmark(files, budget_ms, fmt, output) -> int:
        """Benchmark per-file analysis latency and emit a profile (Step 78)."""
        from src.symexec import benchmark_paths, summarise

        records = benchmark_paths(
            [str(f) for f in files], repeats=3, budget_ms=budget_ms
        )
        profile = summarise(records)
        if fmt == "json":
            out = json.dumps(
                {"summary": profile, "files": [r.to_dict() for r in records]},
                indent=2,
                sort_keys=True,
            ) + "\n"
        else:
            lines = ["symexec performance benchmark", ""]
            lines.append(f"  files          : {profile['files']}")
            lines.append(f"  mean latency   : {profile['mean_ms']:.3f} ms")
            lines.append(f"  p95 latency    : {profile['p95_ms']:.3f} ms")
            lines.append(f"  max latency    : {profile['max_ms']:.3f} ms")
            lines.append(f"  slowest file   : {profile['slowest_file']}")
            if profile["errors"]:
                lines.append(f"  errors         : {profile['errors']}")
            if profile["budget_exceeded"]:
                lines.append(f"  budget tripped : {profile['budget_exceeded']}")
            caps = ", ".join(f"{k}={v}" for k, v in sorted(profile["iteration_caps"].items()))
            lines.append(f"  iteration caps : {caps}")
            out = "\n".join(lines) + "\n"
        SymexecCommand._write(out, output)
        return 0

    @staticmethod
    def _run_fx(source: str, filename: str) -> List[dict]:
        """Run only the FX/SMT shape path (no symexec) and return bug dicts."""
        try:
            from src.api import analyze as api_analyze

            res = api_analyze(source, filename=filename, use_symexec=False)
            out = []
            for b in res.bugs:
                loc = getattr(b, "location", None)
                out.append(
                    {
                        "category": getattr(getattr(b, "category", None), "value", str(getattr(b, "category", ""))),
                        "message": getattr(b, "message", ""),
                        "line": getattr(loc, "line", 0) if loc else 0,
                        "col": getattr(loc, "column", 0) if loc else 0,
                        "severity": getattr(b, "severity", "error"),
                    }
                )
            return out
        except Exception as exc:  # FX path may require torch; degrade gracefully.
            sys.stderr.write(f"tensorguard symexec: FX engine unavailable: {exc}\n")
            return []

    @staticmethod
    def _render_json(records: List[dict]) -> str:
        return json.dumps({"results": records}, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _render_sarif(items: List[tuple]) -> str:
        """Render the symexec results as a SARIF 2.1.0 log (Step 68)."""
        from src.symexec import to_sarif

        return json.dumps(to_sarif(items), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _render_github(items: List[tuple]) -> str:
        """Render GitHub Actions annotation commands for the symexec results
        (Step 69)."""
        from src.symexec import render_github_annotations

        lines = [
            render_github_annotations(sr, filename=fname)
            for fname, sr in items
            if sr.bugs
        ]
        return ("\n".join(l for l in lines if l) + "\n") if lines else ""

    @staticmethod
    def _render_lsp(items: List[tuple]) -> str:
        """Render LSP diagnostics JSON (one entry per file) for the symexec
        results (Step 69)."""
        from src.symexec import to_lsp_diagnostics

        payload = [
            {"uri": fname, "diagnostics": to_lsp_diagnostics(sr, uri=fname)}
            for fname, sr in items
        ]
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _render_text(records: List[dict], engine: str, show_fingerprint: bool,
                     show_coverage: bool = False) -> str:
        lines: List[str] = []
        total = 0
        for rec in records:
            sym = rec["engines"].get("symexec")
            fx = rec["engines"].get("fx")
            header_written = False

            def _header():
                nonlocal header_written
                if not header_written:
                    lines.append(f"== {rec['file']} ==")
                    header_written = True

            if sym is not None:
                if sym["bugs"] or sym["abstain_total"] or show_fingerprint or show_coverage:
                    _header()
                if engine == "both":
                    lines.append("  [symexec]")
                for b in sym["bugs"]:
                    total += 1
                    lines.append(
                        f"  {b['kind']} at {b['line']}:{b['col']} "
                        f"(conf {b['confidence']:.2f}) — {b['message']}"
                    )
                if sym.get("explain"):
                    lines.append(_indent(sym["explain"], "  "))
                if show_coverage and sym.get("coverage"):
                    cov = sym["coverage"]
                    lines.append(
                        f"  coverage: {cov['non_top_statements']}/{cov['total_statements']} "
                        f"statements non-Top ({cov['coverage']:.0%}); "
                        f"value {cov['non_top_bindings']}/{cov['binding_statements']} "
                        f"({cov['value_coverage']:.0%})"
                    )
                    if cov.get("unmodeled_kinds"):
                        gaps = ", ".join(
                            f"{k}={v}" for k, v in sorted(cov["unmodeled_kinds"].items())
                        )
                        lines.append(f"  unmodeled statements: {gaps}")
                if show_fingerprint:
                    lines.append(f"  fingerprint: {sym['fingerprint']}")
                    if sym["abstain_total"]:
                        cov = ", ".join(
                            f"{k}={v}" for k, v in sorted(sym["abstain_coverage"].items())
                        )
                        lines.append(f"  abstain: {sym['abstain_total']} ({cov})")

            if fx is not None:
                if fx["bugs"]:
                    _header()
                if engine == "both":
                    lines.append("  [fx]")
                for b in fx["bugs"]:
                    total += 1
                    lines.append(
                        f"  {b['category']} at {b['line']}:{b['col']} — {b['message']}"
                    )

        bug_count = sum(
            len(rec["engines"].get("symexec", {}).get("bugs", []))
            + len(rec["engines"].get("fx", {}).get("bugs", []))
            for rec in records
        )
        lines.append("")
        lines.append(
            f"analyzed {len(records)} file(s); "
            f"{bug_count} bug(s) found."
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write(text: str, output: Optional[str]) -> None:
        if output:
            with open(output, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            sys.stdout.write(text)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class FixCommand:
    """Apply machine-verified source repairs for the bugs the symexec engine finds.

    ``tensorguard fix <path>`` proposes a minimal, canonical edit for each
    repairable bug, **re-runs the analyzer on the patched source**, and only
    surfaces a fix when the targeted bug is gone *and* no new bug kind appears —
    so every fix is verified, not guessed. By default it prints the unified
    diffs; ``--write`` applies them in place. ``--soundness-mode heuristic``
    (the default for ``fix``) also repairs intent bugs such as a missing
    ``super().__init__()`` or a ``module.forward(x)`` call.
    """

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "paths", nargs="*", default=["."],
            help="Python files or directories to repair (default: .)",
        )
        parser.add_argument(
            "--soundness-mode", choices=["sound", "balanced", "heuristic"],
            default="heuristic", dest="soundness_mode",
            help="Which findings to attempt to repair (default: heuristic, which "
                 "includes intent bugs like missing super().__init__()).",
        )
        parser.add_argument(
            "--write", "-w", action="store_true",
            help="Apply the verified fixes in place instead of only printing diffs.",
        )
        parser.add_argument(
            "--unverified", action="store_true",
            help="Also show candidate fixes that failed re-verification (with the "
                 "reason), for diagnostics. Never written to disk.",
        )
        parser.add_argument(
            "--format", choices=["text", "json", "sarif", "patch"], default="text",
            dest="fmt",
            help="Output format (default: text). 'sarif' emits SARIF 2.1.0 with "
                 "an 'Apply suggested fix' for each verified repair; 'patch' emits "
                 "a `git apply`-able unified patch of all verified fixes.",
        )
        parser.add_argument("-o", "--output", help="Write output to this file (default: stdout).")

    @staticmethod
    def _collect_files(paths: Sequence[str]) -> List[pathlib.Path]:
        return SymexecCommand._collect_files(paths)

    def execute(self, args: argparse.Namespace) -> int:
        from src.symexec import SymConfig, repair

        paths = getattr(args, "paths", None) or ["."]
        mode = getattr(args, "soundness_mode", "heuristic")
        config = SymConfig.for_mode(mode)
        write = bool(getattr(args, "write", False))
        show_unverified = bool(getattr(args, "unverified", False))
        fmt = getattr(args, "fmt", "text")

        files = self._collect_files(paths)
        if not files:
            sys.stderr.write("tensorguard fix: no Python files found\n")
            return 1

        records: List[dict] = []
        text_chunks: List[str] = []
        sarif_runs: List[dict] = []
        patch_chunks: List[str] = []
        total_verified = 0
        total_written = 0

        for f in files:
            try:
                source = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                sys.stderr.write(f"tensorguard fix: cannot read {f}: {exc}\n")
                continue

            fixes = repair(
                source, filename=str(f), config=config,
                verified_only=not show_unverified,
            )
            verified = [x for x in fixes if x.verified]
            if not fixes:
                continue
            total_verified += len(verified)

            # Apply verified fixes in place, one at a time, re-deriving the diff
            # against the evolving file so multiple fixes compose deterministically.
            wrote_this_file = False
            if write and verified:
                patched = source
                applied = 0
                from src.symexec import repair as _repair
                # Re-run repair against the (possibly already-edited) buffer so
                # line numbers stay consistent after each applied edit.
                while True:
                    step = _repair(patched, filename=str(f), config=config)
                    if not step:
                        break
                    patched = step[0].patched_source
                    applied += 1
                    if applied > 1000:  # pathological guard
                        break
                if patched != source:
                    f.write_text(patched, encoding="utf-8")
                    wrote_this_file = True
                    total_written += applied

            records.append({
                "file": str(f),
                "written": wrote_this_file,
                "fixes": [
                    {
                        "kind": x.kind,
                        "line": x.line,
                        "strategy": x.strategy,
                        "description": x.description,
                        "verified": x.verified,
                        "detail": x.detail,
                        "diff": x.diff,
                    }
                    for x in fixes
                ],
            })

            if fmt == "text":
                text_chunks.append(self._render_file(f, fixes, wrote_this_file))
            elif fmt == "sarif":
                run = self._sarif_run_for_file(str(f), source, verified, config)
                if run is not None:
                    sarif_runs.append(run)
            elif fmt == "patch" and verified:
                patch = self._cumulative_patch(str(f), source, config)
                if patch:
                    patch_chunks.append(patch)

        if fmt == "json":
            out = json.dumps({
                "files": records,
                "verified_fixes": total_verified,
                "applied": total_written,
            }, indent=2) + "\n"
        elif fmt == "sarif":
            from src.symexec.export import SARIF_VERSION, SARIF_SCHEMA
            out = json.dumps({
                "version": SARIF_VERSION,
                "$schema": SARIF_SCHEMA,
                "runs": sarif_runs,
            }, indent=2) + "\n"
        elif fmt == "patch":
            out = "".join(patch_chunks)
        else:
            if not text_chunks:
                out = "tensorguard fix: no repairable bugs found.\n"
            else:
                summary = (
                    f"\n{total_verified} verified fix(es) across "
                    f"{len(records)} file(s)"
                )
                if write:
                    summary += f"; {total_written} applied"
                else:
                    summary += "; re-run with --write to apply"
                out = "\n".join(text_chunks) + summary + ".\n"

        self._write(out, getattr(args, "output", None))
        return 0

    @staticmethod
    def _cumulative_patch(filename: str, source: str, config) -> str:
        """Apply every verified fix iteratively (without touching disk) and emit a
        single ``git apply``-able unified patch of original → fully-repaired, or
        an empty string when nothing changed."""
        from src.symexec import repair as _repair

        patched = source
        applied = 0
        while True:
            step = _repair(patched, filename=filename, config=config)
            if not step:
                break
            patched = step[0].patched_source
            applied += 1
            if applied > 1000:  # pathological guard
                break
        if patched == source:
            return ""
        import difflib
        a = source.splitlines(keepends=True)
        b = patched.splitlines(keepends=True)
        body = "".join(
            difflib.unified_diff(
                a, b, fromfile=f"a/{filename}", tofile=f"b/{filename}"
            )
        )
        if body and not body.endswith("\n"):
            body += "\n"
        return f"diff --git a/{filename} b/{filename}\n{body}"

    @staticmethod
    def _sarif_run_for_file(filename: str, source: str, verified, config) -> Optional[dict]:
        """Build a SARIF ``run`` for one file: re-analyze for the findings, then
        attach an 'Apply suggested fix' to each finding that has a verified
        repair (matched by ``(kind, line)``). Returns ``None`` on analysis
        failure."""
        from src.symexec import analyze_source
        from src.symexec.export import result_to_sarif_run, sarif_replacement

        try:
            result = analyze_source(source, filename=filename, config=config)
        except Exception:
            return None
        fixes_by_loc: Dict[tuple, tuple] = {}
        for x in verified:
            replacement = sarif_replacement(source, x.patched_source)
            if replacement is None:
                continue
            fixes_by_loc[(x.kind, int(x.line))] = (x.description, replacement)
        return result_to_sarif_run(result, filename, fixes_by_loc=fixes_by_loc)

    @staticmethod
    def _render_file(path: pathlib.Path, fixes, written: bool) -> str:
        lines = [f"=== {path} ==="]
        for x in fixes:
            mark = "✓ verified" if x.verified else "✗ rejected"
            lines.append(f"  [{mark}] {x.kind} (line {x.line}) — {x.strategy}")
            lines.append(f"    {x.description}")
            if not x.verified:
                lines.append(f"    reason: {x.detail}")
            if x.diff:
                for d in x.diff.splitlines():
                    lines.append(f"    {d}")
        if written:
            lines.append("  → applied in place.")
        return "\n".join(lines)

    @staticmethod
    def _write(text: str, output: Optional[str]) -> None:
        if output:
            with open(output, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            sys.stdout.write(text)


class ReftypeCliApp:
    """Main CLI application class that wires subcommands to argparse."""

    COMMANDS: Dict[str, Callable[[], Command]] = {
        "analyze": lambda: AnalyzeCommand(),
        "analyze-package": lambda: PackageAnalyzeCommand(),
        "verify": lambda: VerifyCommand(),
        "symexec": lambda: SymexecCommand(),
        "fix": lambda: FixCommand(),
        "explain": lambda: ExplainCommand(),
        "watch": lambda: WatchCommand(),
        "ci-check": lambda: CiCheckCommand(),
        "init": lambda: InitCommand(),
        "report": lambda: ReportCommand(),
        "export": lambda: ExportCommand(),
        "diff": lambda: DiffCommand(),
        "server": lambda: ServerCommand(),
        "version": lambda: VersionCommand(),
        "config": lambda: ConfigCommand(),
        "operator-confidence": lambda: OperatorConfidenceCommand(),
        "model-hub-badge": lambda: ModelHubBadgeCommand(),
        "playground": lambda: PlaygroundCommand(),
        "adoption-recipes": lambda: AdoptionRecipesCommand(),
        "sarif-trends": lambda: SarifTrendsCommand(),
        "usage-metrics": lambda: UsageMetricsCommand(),
        "scan-torch-data": lambda: ScanTorchDataCommand(),
        "prove-surface-ban": lambda: ProveSurfaceBanCommand(),
        "prove-streaming-stop": lambda: ProveStreamingStopCommand(),
        "prove-decoding-feasibility": lambda: ProveDecodingFeasibilityCommand(),
    }

    def __init__(self) -> None:
        self.parser = self._build_parser()
        self._update_checker = UpdateChecker()

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="tensorguard",
            description=(
                "Refinement type inference for dynamically-typed languages "
                "(Python & TypeScript) using CEGAR."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""\
            examples:
              reftype analyze .
              reftype analyze src/ --format sarif -o results.sarif
              reftype analyze-package my_project/ --output-format json
              reftype analyze-package . --requirements requirements.txt --output-format sarif
              reftype watch src/ --debounce 1.0
              reftype ci-check --baseline baseline.json --sarif-output results.sarif
              reftype init --language python
              reftype server --transport stdio
              reftype diff before.json after.json
            """),
        )
        parser.add_argument(
            "--version", action="version", version=f"tensorguard {_VERSION}"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        for name, factory in self.COMMANDS.items():
            cmd = factory()
            sub = subparsers.add_parser(name, help=self._command_help(name))
            cmd.register(sub)

        return parser

    @staticmethod
    def _command_help(name: str) -> str:
        helps: Dict[str, str] = {
            "analyze": "Analyse files/directories for refinement type bugs",
            "analyze-package": "Analyse an entire Python package/directory with summary",
            "verify": "Verify nn.Module architecture via constraint-based verification",
            "symexec": "Run the Python symbolic-execution engine (--engine symexec|fx|both, --explain, --fingerprint)",
            "fix": "Apply machine-verified source repairs for symexec findings (--write to apply, --soundness-mode, --unverified)",
            "explain": "Generate an HTML inference-chain explanation report",
            "watch": "Watch files for changes and re-analyse incrementally",
            "ci-check": "Run analysis in CI mode with exit codes",
            "init": "Initialise .reftype.toml configuration",
            "report": "Generate analysis reports",
            "export": "Export inferred contracts",
            "diff": "Compare two analysis results",
            "server": "Start the LSP server",
            "version": "Show version information",
            "config": "Show or edit configuration",
            "operator-confidence": "Show per-operator confidence tags (sound/complete/heuristic)",
            "model-hub-badge": "Write a TensorGuard-verified model-hub certificate bundle",
            "playground": "Generate a no-upload local static TensorGuard playground",
            "adoption-recipes": "Print one-line setup recipes for CI, hooks, editors, and notebooks",
            "sarif-trends": "Build a Code Scanning trend dashboard from SARIF snapshots",
            "usage-metrics": "Summarize local JSON reports without telemetry or source code",
            "scan-torch-data": "Scan PyTorch data-pipeline source for silent data-misuse bugs (worker-RNG, drop_last-on-eval, fit-before-split leakage)",
            "prove-surface-ban": "Prove whether an id-level bad_words/suppress_tokens ban prevents a forbidden surface string",
            "prove-streaming-stop": "Prove whether a server can truncate output exactly at a stop string (overshoot / split-stop)",
            "prove-decoding-feasibility": "Prove whether a guided-decoding grammar admits a tokenization under a vocabulary",
        }
        return helps.get(name, "")

    def run(self, argv: Optional[Sequence[str]] = None) -> int:
        args = self.parser.parse_args(argv)
        command_name = getattr(args, "command", None)

        if not command_name:
            self.parser.print_help()
            return 0

        self._update_checker.check_async()

        factory = self.COMMANDS.get(command_name)
        if factory is None:
            sys.stderr.write(f"Unknown command: {command_name}\n")
            return 2

        cmd = factory()
        try:
            exit_code = cmd.execute(args)
        except Exception as exc:
            exit_code = ErrorHandler().handle(exc, command_name)

        update_msg = self._update_checker.notify_if_outdated()
        if update_msg:
            sys.stderr.write(f"\n{update_msg}\n")

        return exit_code


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    app = ReftypeCliApp()
    return app.run(argv)


if __name__ == "__main__":
    sys.exit(main())
