"""Step 252: broad same-case head-to-head benchmark.

This benchmark scores TensorGuard and practical alternatives on the exact 20
cases used by the frozen GPT-4.1-nano LLM baseline in
``experiments/llm_baseline_results.json``.  The script does not call an external
LLM; it verifies and reuses that committed artifact so reproduction never uploads
repository code.

The live tools are intentionally heterogeneous.  Some are static source checkers
(Pyright, TensorGuard, PyTea), some are dynamic smoke/guard paths
(runtime/jaxtyping/torch.export), and some are environment-qualified
(torchtyping if absent, TorchDynamo on Python versions unsupported by PyTorch).
The artifact records coverage/NA explicitly instead of pretending every tool can
answer every case.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.run_llm_baseline import BENCHMARKS, OUTPUT_FILE as LLM_OUTPUT  # noqa: E402
from src.unified import analyze_unified  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "head_to_head_step252.json"
OUT_MD = REPO / "reproducibility" / "head_to_head_step252.md"
PYTEA_BIN = REPO / "experiments_v5" / "_pytea_src" / "bin" / "pytea.py"

BUGGY = "buggy"
CLEAN = "clean"
NA = "na"
TORCH_SEED = 0
PYTEA_TIMEOUT_S = 120
SUBPROCESS_TIMEOUT_S = 60
_JAXTYPING_COMPATIBLE: Optional[bool] = None

TOOLS = [
    "tensorguard_unified",
    "pytea",
    "pyright",
    "jaxtyping_runtime",
    "torchtyping_runtime",
    "torch_export_guards",
    "torch_dynamo_guards",
    "runtime_forward_smoke",
    "llm_gpt4_1_nano_frozen",
]


@dataclass(frozen=True)
class EntrySpec:
    kind: str  # "function" | "module"
    function: Optional[str] = None
    class_name: Optional[str] = None
    args: Tuple[str, ...] = ()
    input_shapes: Tuple[Tuple[int, ...], ...] = ()
    note: str = ""


# Every benchmark case gets an explicit deterministic entrypoint.  The two
# Optional/None bug cases deliberately use cond=True: this is the "single happy
# path" a smoke test would often try, so runtime tools miss the latent cond=False
# bug while static tools can still inspect both branches.
ENTRY_SPECS: Dict[str, EntrySpec] = {
    "matmul_2d_mismatch": EntrySpec("function", function="f"),
    "linear_chain_mismatch": EntrySpec(
        "module", class_name="M", input_shapes=((2, 10),)
    ),
    "conv_chain_mismatch": EntrySpec(
        "module", class_name="M", input_shapes=((2, 3, 16, 16),)
    ),
    "null_optional_tensor_shape": EntrySpec(
        "function", function="f", args=("True",), note="single-path cond=True smoke"
    ),
    "reshape_element_mismatch": EntrySpec("function", function="f"),
    "cat_dim_mismatch": EntrySpec("function", function="f"),
    "broadcast_mismatch": EntrySpec("function", function="f"),
    "interproc_matmul_mismatch": EntrySpec("function", function="use_tensor"),
    "autoencoder_mismatch": EntrySpec(
        "module", class_name="AE", input_shapes=((2, 784),)
    ),
    "cross_null_matmul": EntrySpec(
        "function", function="f", args=("True",), note="single-path cond=True smoke"
    ),
    "matmul_2d_correct": EntrySpec("function", function="f"),
    "linear_chain_correct": EntrySpec(
        "module", class_name="M", input_shapes=((2, 10),)
    ),
    "conv_chain_correct": EntrySpec(
        "module", class_name="M", input_shapes=((2, 3, 16, 16),)
    ),
    "null_guarded_tensor_correct": EntrySpec(
        "function", function="f", args=("True",), note="single-path cond=True smoke"
    ),
    "reshape_correct_flatten": EntrySpec("function", function="f"),
    "cat_correct_dim0": EntrySpec("function", function="f"),
    "broadcast_correct_scalar": EntrySpec("function", function="f"),
    "interproc_matmul_correct": EntrySpec("function", function="use_tensor"),
    "autoencoder_correct": EntrySpec(
        "module", class_name="AE", input_shapes=((2, 784),)
    ),
    "cross_null_matmul_guarded": EntrySpec(
        "function", function="f", args=("True",), note="single-path cond=True smoke"
    ),
}

TOOL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "tensorguard_unified": {
        "static_no_execution": True,
        "needs_concrete_inputs": False,
        "needs_annotations": False,
        "needs_code_upload": False,
        "live_or_frozen": "live_local",
        "domains": ["shape", "null", "cross-domain"],
    },
    "pytea": {
        "static_no_execution": True,
        "needs_concrete_inputs": False,
        "needs_annotations": False,
        "needs_code_upload": False,
        "live_or_frozen": "live_local_if_node_available",
        "domains": ["shape"],
    },
    "pyright": {
        "static_no_execution": True,
        "needs_concrete_inputs": False,
        "needs_annotations": False,
        "needs_code_upload": False,
        "live_or_frozen": "live_local_if_cli_available",
        "domains": ["python_types", "optional_none"],
    },
    "jaxtyping_runtime": {
        "static_no_execution": False,
        "needs_concrete_inputs": True,
        "needs_annotations": True,
        "needs_code_upload": False,
        "live_or_frozen": "live_local_if_package_available",
        "domains": ["runtime_shape_annotations"],
        "honesty_note": (
            "The benchmark cases are not authored with jaxtyping contracts, so "
            "this row mostly measures execution with a thin checked wrapper."
        ),
    },
    "torchtyping_runtime": {
        "static_no_execution": False,
        "needs_concrete_inputs": True,
        "needs_annotations": True,
        "needs_code_upload": False,
        "live_or_frozen": "live_local_if_package_available",
        "domains": ["runtime_shape_annotations"],
        "honesty_note": (
            "Unavailable in the locked capsule unless torchtyping is installed; "
            "scored NA rather than silently dropped."
        ),
    },
    "torch_export_guards": {
        "static_no_execution": False,
        "needs_concrete_inputs": True,
        "needs_annotations": False,
        "needs_code_upload": False,
        "live_or_frozen": "live_local",
        "domains": ["export_trace_guards"],
    },
    "torch_dynamo_guards": {
        "static_no_execution": False,
        "needs_concrete_inputs": True,
        "needs_annotations": False,
        "needs_code_upload": False,
        "live_or_frozen": "live_local_if_supported_by_torch",
        "domains": ["dynamo_guards"],
    },
    "runtime_forward_smoke": {
        "static_no_execution": False,
        "needs_concrete_inputs": True,
        "needs_annotations": False,
        "needs_code_upload": False,
        "live_or_frozen": "live_local",
        "domains": ["runtime_exceptions"],
    },
    "llm_gpt4_1_nano_frozen": {
        "static_no_execution": True,
        "needs_concrete_inputs": False,
        "needs_annotations": False,
        "needs_code_upload": True,
        "live_or_frozen": "frozen_prior_api_run",
        "domains": ["source_review"],
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _case_hash(cases: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "name": c["name"],
            "category": c["category"],
            "expect_bug": c["label"] == BUGGY,
            "source_sha256": hashlib.sha256(c["code"].encode("utf-8")).hexdigest(),
            "entry": _entry_spec_json(ENTRY_SPECS[c["name"]]),
        }
        for c in cases
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _entry_spec_json(spec: EntrySpec) -> Dict[str, Any]:
    return {
        "kind": spec.kind,
        "function": spec.function,
        "class_name": spec.class_name,
        "args": list(spec.args),
        "input_shapes": [list(s) for s in spec.input_shapes],
        "note": spec.note,
    }


def _package_version(pkg: str) -> Optional[str]:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(argv: Sequence[str], timeout: int = 20) -> Optional[str]:
    if not argv or shutil.which(argv[0]) is None:
        return None
    try:
        proc = subprocess.run(
            list(argv),
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return out[0].strip() if out else None


def _jaxtyping_typeguard_compatible() -> bool:
    global _JAXTYPING_COMPATIBLE
    if _JAXTYPING_COMPATIBLE is not None:
        return _JAXTYPING_COMPATIBLE
    if _package_version("jaxtyping") is None or _package_version("typeguard") is None:
        _JAXTYPING_COMPATIBLE = False
        return False
    source = """\
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from torch import Tensor
import torch

@jaxtyped(typechecker=typechecked)
def f(x: Float[Tensor, 'd0 d1']):
    return x

f(torch.randn(2, 3))
"""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=str(REPO)
    ) as fh:
        fh.write(source)
        path = fh.name
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, path],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            env=env,
        )
        _JAXTYPING_COMPATIBLE = proc.returncode == 0
        return _JAXTYPING_COMPATIBLE
    finally:
        os.unlink(path)


def _environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch": _package_version("torch"),
        "jaxtyping": _package_version("jaxtyping"),
        "typeguard": _package_version("typeguard"),
        "jaxtyping_typeguard_compatible": _jaxtyping_typeguard_compatible(),
        "torchtyping": _package_version("torchtyping"),
        "pyright": _command_version(["pyright", "--version"]),
        "node": _command_version(["node", "--version"]),
        "pytea_bin_sha256": _sha256(PYTEA_BIN) if PYTEA_BIN.exists() else None,
    }


def _load_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for bm in BENCHMARKS:
        if bm.name not in ENTRY_SPECS:
            raise KeyError(f"missing Step-252 entry spec for {bm.name}")
        seen.add(bm.name)
        out.append({
            "name": bm.name,
            "category": bm.category,
            "label": BUGGY if bm.expect_bug else CLEAN,
            "code": bm.code,
            "entry": _entry_spec_json(ENTRY_SPECS[bm.name]),
        })
    extra_specs = sorted(set(ENTRY_SPECS) - seen)
    if extra_specs:
        raise KeyError(f"entry specs without benchmark cases: {extra_specs}")
    return out


def _tensor_for_shape(shape: Sequence[int], torch):
    return torch.randn(*[int(d) for d in shape])


def _compile_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    exec(compile(case["code"], f"<step252:{case['name']}>", "exec"), ns)
    return ns


def _run_entry(case: Mapping[str, Any]) -> Tuple[str, str]:
    import torch

    spec = ENTRY_SPECS[case["name"]]
    torch.manual_seed(TORCH_SEED)
    ns = _compile_case(case)
    try:
        if spec.kind == "function":
            fn = ns[spec.function]
            args = [eval(arg, {"True": True, "False": False}) for arg in spec.args]
            with torch.no_grad():
                fn(*args)
            return CLEAN, "ran_clean"
        cls = ns[spec.class_name]
        model = cls()
        model.eval()
        inputs = tuple(_tensor_for_shape(s, torch) for s in spec.input_shapes)
        with torch.no_grad():
            model(*inputs)
        return CLEAN, "ran_clean"
    except Exception as exc:
        return BUGGY, f"raised:{type(exc).__name__}"


def _module_instance(case: Mapping[str, Any]):
    import torch

    spec = ENTRY_SPECS[case["name"]]
    if spec.kind != "module":
        raise ValueError("non-module case")
    torch.manual_seed(TORCH_SEED)
    ns = _compile_case(case)
    cls = ns[spec.class_name]
    model = cls()
    model.eval()
    inputs = tuple(_tensor_for_shape(s, torch) for s in spec.input_shapes)
    return model, inputs


def _temp_source(case: Mapping[str, Any], extra: str) -> str:
    return case["code"].rstrip() + "\n\n" + extra.strip() + "\n"


def _entry_harness_source(case: Mapping[str, Any]) -> str:
    spec = ENTRY_SPECS[case["name"]]
    lines = [
        "import torch as _tg_torch",
        f"_tg_torch.manual_seed({TORCH_SEED})",
    ]
    if spec.kind == "function":
        args = ", ".join(spec.args)
        lines.append(f"_tg_result = {spec.function}({args})")
    else:
        lines.append(f"_tg_model = {spec.class_name}()")
        lines.append("_tg_model.eval()")
        arg_names = []
        for i, shape in enumerate(spec.input_shapes):
            name = f"_tg_arg{i}"
            arg_names.append(name)
            dims = ", ".join(str(int(d)) for d in shape)
            lines.append(f"{name} = _tg_torch.randn({dims})")
        lines.append(f"_tg_result = _tg_model({', '.join(arg_names)})")
    return "\n".join(lines)


def _run_temp_python(source: str) -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=str(REPO)
    ) as fh:
        fh.write(source)
        path = fh.name
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, path],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            env=env,
        )
        if proc.returncode == 0:
            return True, "ran_clean"
        text = (proc.stderr or proc.stdout).strip().splitlines()
        last = text[-1] if text else f"exit_{proc.returncode}"
        return False, last[:160]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def _parse_pytea(out: str) -> Tuple[str, str]:
    fail_m = re.search(r"immediate failed path #:\s*(\d+)", out)
    invalid_m = re.search(r"Invalid paths[^:]*:\s*(\d+)", out)
    succ_m = re.search(r"potential success path #:\s*(\d+)", out)
    fail = int(fail_m.group(1)) if fail_m else 0
    invalid = int(invalid_m.group(1)) if invalid_m else 0
    succ = int(succ_m.group(1)) if succ_m else 0
    if fail > 0:
        return BUGGY, f"failed_path={fail}"
    if invalid > 0:
        return BUGGY, f"z3_invalid={invalid}"
    if succ > 0:
        return CLEAN, f"success_path={succ}"
    if "Frontend parse failed" in out:
        return NA, "frontend_parse_failed"
    return NA, "no_paths"


def predict_tensorguard(case: Mapping[str, Any]) -> Tuple[str, str]:
    result = analyze_unified(case["code"])
    bugs = len(result.bugs)
    return (BUGGY, f"bugs={bugs}") if bugs else (CLEAN, "bugs=0")


def predict_pytea(case: Mapping[str, Any]) -> Tuple[str, str]:
    if not PYTEA_BIN.exists() or shutil.which("node") is None:
        return NA, "pytea_unavailable"
    source = _temp_source(case, _entry_harness_source(case))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, str(PYTEA_BIN), path],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=PYTEA_TIMEOUT_S,
        )
        return _parse_pytea(proc.stdout + "\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        return NA, "timeout"
    finally:
        os.unlink(path)


def predict_pyright(case: Mapping[str, Any]) -> Tuple[str, str]:
    if shutil.which("pyright") is None:
        return NA, "pyright_unavailable"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=str(REPO)
    ) as fh:
        fh.write(case["code"])
        path = fh.name
    try:
        proc = subprocess.run(
            ["pyright", "--outputjson", path],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        payload = json.loads(proc.stdout or "{}")
        diags = payload.get("generalDiagnostics", [])
        errors = [d for d in diags if d.get("severity") == "error"]
        rules = sorted({d.get("rule", "unknown") for d in errors})
        if errors:
            return BUGGY, "errors=%d rules=%s" % (len(errors), ",".join(rules))
        return CLEAN, "errors=0"
    except Exception as exc:
        return NA, f"pyright_error:{type(exc).__name__}"
    finally:
        os.unlink(path)


def predict_jaxtyping(case: Mapping[str, Any]) -> Tuple[str, str]:
    if _package_version("jaxtyping") is None or _package_version("typeguard") is None:
        return NA, "jaxtyping_or_typeguard_unavailable"
    if not _jaxtyping_typeguard_compatible():
        return NA, "jaxtyping_typeguard_incompatible"
    spec = ENTRY_SPECS[case["name"]]
    wrapper: List[str] = [
        "from jaxtyping import Float, jaxtyped",
        "from typeguard import typechecked",
        "from torch import Tensor",
        "import torch as _tg_torch",
        f"_tg_torch.manual_seed({TORCH_SEED})",
    ]
    if spec.kind == "module" and len(spec.input_shapes) == 1:
        # jaxtyping shape annotations use axis identifiers, not bare numeric
        # literals.  The wrapper validates rank/dtype while the deterministic
        # tensor construction supplies the concrete shape for the underlying run.
        dims = " ".join(f"d{i}" for i, _ in enumerate(spec.input_shapes[0]))
        shape_args = ", ".join(str(d) for d in spec.input_shapes[0])
        wrapper += [
            "@jaxtyped(typechecker=typechecked)",
            f"def _tg_checked(x: Float[Tensor, '{dims}']) -> Tensor:",
            f"    return {spec.class_name}()(x)",
            f"_tg_checked(_tg_torch.randn({shape_args}))",
        ]
    elif spec.kind == "function":
        args = ", ".join(spec.args)
        wrapper += [
            "@jaxtyped(typechecker=typechecked)",
            "def _tg_checked():",
            f"    return {spec.function}({args})",
            "_tg_checked()",
        ]
    else:
        return NA, "unsupported_entry_shape"
    ok, detail = _run_temp_python(_temp_source(case, "\n".join(wrapper)))
    return (CLEAN, detail) if ok else (BUGGY, f"raised:{detail.split(':')[0]}")


def predict_torchtyping(case: Mapping[str, Any]) -> Tuple[str, str]:
    if _package_version("torchtyping") is None:
        return NA, "torchtyping_unavailable"
    spec = ENTRY_SPECS[case["name"]]
    wrapper: List[str] = [
        "from torchtyping import TensorType, patch_typeguard",
        "from typeguard import typechecked",
        "import torch as _tg_torch",
        "patch_typeguard()",
        f"_tg_torch.manual_seed({TORCH_SEED})",
    ]
    if spec.kind == "module" and len(spec.input_shapes) == 1:
        dims = ", ".join(str(d) for d in spec.input_shapes[0])
        shape_args = ", ".join(str(d) for d in spec.input_shapes[0])
        wrapper += [
            "@typechecked",
            f"def _tg_checked(x: TensorType[{dims}]):",
            f"    return {spec.class_name}()(x)",
            f"_tg_checked(_tg_torch.randn({shape_args}))",
        ]
    elif spec.kind == "function":
        args = ", ".join(spec.args)
        wrapper += [
            "@typechecked",
            "def _tg_checked():",
            f"    return {spec.function}({args})",
            "_tg_checked()",
        ]
    else:
        return NA, "unsupported_entry_shape"
    ok, detail = _run_temp_python(_temp_source(case, "\n".join(wrapper)))
    return (CLEAN, detail) if ok else (BUGGY, f"raised:{detail.split(':')[0]}")


def predict_torch_export(case: Mapping[str, Any]) -> Tuple[str, str]:
    if ENTRY_SPECS[case["name"]].kind != "module":
        return NA, "non_module_entrypoint"
    try:
        import torch

        logging.getLogger("torch").setLevel(logging.CRITICAL)
        model, inputs = _module_instance(case)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            torch.export.export(model, inputs)
        return CLEAN, "exported"
    except Exception as exc:
        return BUGGY, f"raised:{type(exc).__name__}"


def predict_torch_dynamo(case: Mapping[str, Any]) -> Tuple[str, str]:
    if ENTRY_SPECS[case["name"]].kind != "module":
        return NA, "non_module_entrypoint"
    try:
        import torch

        logging.getLogger("torch").setLevel(logging.CRITICAL)
        model, inputs = _module_instance(case)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            compiled(*inputs)
        return CLEAN, "compiled_eager"
    except RuntimeError as exc:
        msg = str(exc).splitlines()[0]
        if "not supported on Python" in msg:
            return NA, "torch_dynamo_python_unsupported"
        return BUGGY, "raised:RuntimeError"
    except Exception as exc:
        return BUGGY, f"raised:{type(exc).__name__}"


def predict_runtime(case: Mapping[str, Any]) -> Tuple[str, str]:
    return _run_entry(case)


def _load_llm_predictions(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[str, str]]:
    with LLM_OUTPUT.open(encoding="utf-8") as fh:
        llm = json.load(fh)
    rows = {row["name"]: row for row in llm["benchmarks"]}
    expected_names = [c["name"] for c in cases]
    if sorted(rows) != sorted(expected_names):
        raise ValueError("frozen LLM artifact does not match Step-252 cases")
    out = {}
    for case in cases:
        row = rows[case["name"]]
        if bool(row["expect_bug"]) != (case["label"] == BUGGY):
            raise ValueError(f"label mismatch in LLM artifact for {case['name']}")
        pred = BUGGY if row["llm_found_bug"] else CLEAN
        out[case["name"]] = (pred, f"frozen_label={row['llm_label']}")
    return out


PREDICTORS = {
    "tensorguard_unified": predict_tensorguard,
    "pytea": predict_pytea,
    "pyright": predict_pyright,
    "jaxtyping_runtime": predict_jaxtyping,
    "torchtyping_runtime": predict_torchtyping,
    "torch_export_guards": predict_torch_export,
    "torch_dynamo_guards": predict_torch_dynamo,
    "runtime_forward_smoke": predict_runtime,
}


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 4)


def confusion(rows: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
    tp = fp = tn = fn = na = 0
    ctp = cfp = ctn = cfn = 0
    for gt, pred in rows:
        gt_bug = gt == BUGGY
        if pred == NA:
            na += 1
            if gt_bug:
                fn += 1
            else:
                fp += 1
            continue
        pred_bug = pred == BUGGY
        if gt_bug and pred_bug:
            tp += 1
            ctp += 1
        elif gt_bug and not pred_bug:
            fn += 1
            cfn += 1
        elif not gt_bug and pred_bug:
            fp += 1
            cfp += 1
        else:
            tn += 1
            ctn += 1

    def prf(tp_: int, fp_: int, fn_: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        precision = tp_ / (tp_ + fp_) if (tp_ + fp_) else None
        recall = tp_ / (tp_ + fn_) if (tp_ + fn_) else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (tp_ + fp_ + fn_) else None
        return _round(precision), _round(recall), _round(f1)

    n = tp + fp + tn + fn
    precision, recall, f1 = prf(tp, fp, fn)
    cprecision, crecall, cf1 = prf(ctp, cfp, cfn)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "NA": na,
        "N": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": _round((n - na) / n) if n else None,
        "covered_only": {
            "TP": ctp,
            "FP": cfp,
            "TN": ctn,
            "FN": cfn,
            "precision": cprecision,
            "recall": crecall,
            "f1": cf1,
        },
    }


def measure() -> Dict[str, Any]:
    cases = _load_cases()
    llm_predictions = _load_llm_predictions(cases)
    per_case: List[Dict[str, Any]] = []
    aligned: Dict[str, List[Tuple[str, str]]] = {tool: [] for tool in TOOLS}

    for case in cases:
        row = {
            "name": case["name"],
            "category": case["category"],
            "label": case["label"],
            "entry": case["entry"],
            "predictions": {},
        }
        for tool in TOOLS:
            if tool == "llm_gpt4_1_nano_frozen":
                pred, detail = llm_predictions[case["name"]]
            else:
                pred, detail = PREDICTORS[tool](case)
            row["predictions"][tool] = {"pred": pred, "detail": detail}
            aligned[tool].append((case["label"], pred))
        per_case.append(row)

    module_case_names = {
        c["name"] for c in cases if ENTRY_SPECS[c["name"]].kind == "module"
    }
    shape_case_names = {
        c["name"] for c in cases if c["category"].startswith("shape_")
    } | {
        c["name"] for c in cases
        if c["category"] in {"interprocedural", "cross_domain"}
    }
    null_case_names = {
        c["name"] for c in cases if c["category"] in {"null", "cross_domain"}
    }

    def subset_conf(names: set[str], tool: str) -> Dict[str, Any]:
        rows = []
        for case, prediction in zip(cases, aligned[tool]):
            if case["name"] in names:
                rows.append(prediction)
        return confusion(rows)

    metrics = {}
    for tool in TOOLS:
        metrics[tool] = {
            "all": confusion(aligned[tool]),
            "module_subset": subset_conf(module_case_names, tool),
            "shape_or_cross_domain_subset": subset_conf(shape_case_names, tool),
            "null_or_cross_domain_subset": subset_conf(null_case_names, tool),
        }

    return {
        "schema_version": "tensorguard.head-to-head-step252.v1",
        "step": 252,
        "generated_by": "reproducibility/head_to_head_step252.py",
        "command": "python3 reproducibility/head_to_head_step252.py",
        "corpus": {
            "source": "experiments.run_llm_baseline.BENCHMARKS",
            "n_cases": len(cases),
            "n_buggy": sum(1 for c in cases if c["label"] == BUGGY),
            "n_clean": sum(1 for c in cases if c["label"] == CLEAN),
            "case_hash": _case_hash(cases),
            "torch_seed": TORCH_SEED,
            "entry_policy": (
                "Each case has an explicit deterministic entrypoint. Boolean "
                "branch cases use cond=True, exposing single-path runtime-smoke "
                "limitations rather than searching both branches."
            ),
        },
        "llm_artifact": {
            "path": str(LLM_OUTPUT.relative_to(REPO)),
            "sha256": _sha256(LLM_OUTPUT),
            "model": "gpt-4.1-nano",
            "frozen": True,
            "regeneration_command": (
                "OPENAI_API_KEY=... python3 experiments/run_llm_baseline.py"
            ),
            "not_regenerated_by_default": (
                "Requires an external API and code upload; this benchmark reuses "
                "the committed artifact after checking case/label alignment."
            ),
        },
        "tools": TOOLS,
        "tool_capabilities": TOOL_CAPABILITIES,
        "environment": _environment(),
        "na_policy": (
            "Headline metrics count NA as wrong: NA on a buggy case is FN and "
            "NA on a clean case is FP. covered_only excludes NA and reports "
            "coverage separately."
        ),
        "per_case": per_case,
        "metrics": metrics,
        "honesty_notes": [
            "Pyright is a Python type checker, not a tensor-shape verifier; on "
            "this corpus it can catch Optional/None bugs but not Linear/Conv/"
            "reshape/broadcast tensor-shape mismatches.",
            "jaxtyping/torchtyping require user-authored shape annotations. The "
            "source cases are unannotated, so these runtime rows mostly collapse "
            "to executing the same deterministic entrypoint under a thin checked "
            "wrapper.",
            "torch.export and TorchDynamo only support nn.Module entries here; "
            "function-only cases are scored NA and summarized by the module "
            "subset.",
        ],
    }


def _fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.3f}"


def render_markdown(data: Mapping[str, Any]) -> str:
    lines = [
        "# Step 252 same-case head-to-head benchmark",
        "",
        (
            "Broad comparison on the exact **{n}** frozen LLM-baseline cases "
            "({b} buggy / {c} clean). The committed GPT-4.1-nano artifact is "
            "hash-checked and reused; no external LLM call is made."
        ).format(
            n=data["corpus"]["n_cases"],
            b=data["corpus"]["n_buggy"],
            c=data["corpus"]["n_clean"],
        ),
        "",
        "## Headline metrics (NA counts as wrong)",
        "",
        "| tool | TP | FP | TN | FN | NA | precision | recall | F1 | coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tool in data["tools"]:
        c = data["metrics"][tool]["all"]
        lines.append(
            f"| `{tool}` | {c['TP']} | {c['FP']} | {c['TN']} | {c['FN']} | "
            f"{c['NA']} | {_fmt(c['precision'])} | {_fmt(c['recall'])} | "
            f"{_fmt(c['f1'])} | {_fmt(c['coverage'])} |"
        )

    lines += [
        "",
        "## Module subset for export-style tools",
        "",
        "| tool | TP | FP | TN | FN | NA | recall | coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tool in data["tools"]:
        c = data["metrics"][tool]["module_subset"]
        lines.append(
            f"| `{tool}` | {c['TP']} | {c['FP']} | {c['TN']} | {c['FN']} | "
            f"{c['NA']} | {_fmt(c['recall'])} | {_fmt(c['coverage'])} |"
        )

    lines += [
        "",
        "## Capability axes",
        "",
        "| tool | static/no exec | needs inputs | needs annotations | code upload/API | live/frozen |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for tool in data["tools"]:
        cap = data["tool_capabilities"][tool]
        lines.append(
            f"| `{tool}` | {cap['static_no_execution']} | "
            f"{cap['needs_concrete_inputs']} | {cap['needs_annotations']} | "
            f"{cap['needs_code_upload']} | {cap['live_or_frozen']} |"
        )

    lines += [
        "",
        "## Reading the comparison honestly",
        "",
    ]
    for note in data["honesty_notes"]:
        lines.append(f"- {note}")
    lines += [
        "",
        f"LLM artifact: `{data['llm_artifact']['path']}` "
        f"(sha256 `{data['llm_artifact']['sha256']}`).",
        "",
    ]
    return "\n".join(lines)


def _json_text(data: Mapping[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _structural_check(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    return (
        old.get("schema_version") == new.get("schema_version")
        and old.get("tools") == new.get("tools")
        and [r["name"] for r in old.get("per_case", [])]
        == [r["name"] for r in new.get("per_case", [])]
        and old.get("corpus", {}).get("case_hash")
        == new.get("corpus", {}).get("case_hash")
        and old.get("llm_artifact", {}).get("sha256")
        == new.get("llm_artifact", {}).get("sha256")
    )


def run(check: bool = False) -> int:
    data = measure()
    js = _json_text(data)
    md = render_markdown(data)
    if check:
        if not OUT_JSON.exists() or not OUT_MD.exists():
            print("MISSING: Step-252 artifacts; run without --check")
            return 1
        old = json.loads(OUT_JSON.read_text())
        if old.get("environment") != data.get("environment"):
            if _structural_check(old, data):
                print("head_to_head_step252: environment differs; structural check passed")
                return 0
            print("MISMATCH: Step-252 structural check failed under environment drift")
            return 1
        if OUT_JSON.read_text() != js or OUT_MD.read_text() != md:
            if _structural_check(old, data):
                print("head_to_head_step252: live tool metrics differ; structural check passed")
                return 0
            print("MISMATCH: head_to_head_step252 artifacts differ")
            return 1
        print("head_to_head_step252: byte-identical")
        return 0
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(check=args.check))


if __name__ == "__main__":
    main()
