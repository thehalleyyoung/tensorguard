"""Step 227 -- real-model deployment gallery with export gates.

The gallery is deliberately small enough for CI but representative of the
deployment surfaces users ask about: residual vision, ViT-style patch models,
Llama-style token blocks, diffusion U-Nets, recommenders, and speech encoders.
Each entry is a real ``nn.Module`` that executes under eager PyTorch and is gated
twice: once before export through the FX frontend and once after capture through
``torch.export`` when that backend is available.

The committed JSON/Markdown artifacts are deterministic architecture manifests;
``--gate`` runs the live TensorGuard checks against real model code.  README
claims intentionally avoid numeric tokens, so future quantitative claims about
this gallery should be registered in ``reproducibility/audit_numeric_claims.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

JSON_PATH = os.path.join(HERE, "deployment_gallery.json")
MD_PATH = os.path.join(HERE, "deployment_gallery.md")


class BackendUnavailable(RuntimeError):
    """Raised when an optional deployment backend cannot run in this process."""


@dataclass(frozen=True)
class GalleryModelSpec:
    name: str
    family: str
    input_shapes: Dict[str, Tuple[int, ...]]
    input_dtypes: Dict[str, str]
    expected_output_shape: Tuple[int, ...]
    description: str
    deployment_note: str
    operator_surface: Tuple[str, ...]


@dataclass(frozen=True)
class GateSpec:
    name: str
    phase: str
    backend: str
    gate: str
    required: bool
    description: str


MODEL_SPECS: Tuple[GalleryModelSpec, ...] = (
    GalleryModelSpec(
        name="resnet_residual_stage",
        family="ResNet",
        input_shapes={"x": (1, 3, 32, 32)},
        input_dtypes={},
        expected_output_shape=(1, 5),
        description="Residual Conv-BN-ReLU image stage with projection skip and classifier head.",
        deployment_note="Covers the canonical residual image deployment path.",
        operator_surface=(
            "AdaptiveAvgPool2d",
            "BatchNorm2d",
            "Conv2d",
            "Linear",
            "ResidualAdd",
            "flatten",
            "relu",
        ),
    ),
    GalleryModelSpec(
        name="vit_patch_mixer",
        family="ViT",
        input_shapes={"x": (1, 3, 32, 32)},
        input_dtypes={},
        expected_output_shape=(1, 3),
        description="Patch-embedding classifier with token-axis transpose, LayerNorm, GELU, and pooling.",
        deployment_note="Exercises patchification and token mixing without external torchvision weights.",
        operator_surface=(
            "Conv2d",
            "LayerNorm",
            "Linear",
            "flatten",
            "gelu",
            "mean",
            "transpose",
        ),
    ),
    GalleryModelSpec(
        name="llama_style_mlp_block",
        family="Llama-style block",
        input_shapes={"tokens": (2, 8)},
        input_dtypes={"tokens": "long"},
        expected_output_shape=(2, 16),
        description="Token embedding, normalization, SwiGLU-style gated MLP, and sequence pooling.",
        deployment_note="Uses integer token IDs for both eager execution and export capture.",
        operator_surface=(
            "Embedding",
            "LayerNorm",
            "Linear",
            "mean",
            "mul",
            "silu",
        ),
    ),
    GalleryModelSpec(
        name="diffusion_unet_skip",
        family="Diffusion U-Net",
        input_shapes={"x": (1, 4, 16, 16)},
        input_dtypes={},
        expected_output_shape=(1, 4, 16, 16),
        description="Tiny encoder/decoder U-Net with strided downsample, transpose-conv upsample, and skip concat.",
        deployment_note="Input size is divisible by the downsample factor so skip tensors realign exactly.",
        operator_surface=(
            "Conv2d",
            "ConvTranspose2d",
            "cat",
            "relu",
        ),
    ),
    GalleryModelSpec(
        name="recommender_two_tower",
        family="Recommender",
        input_shapes={"user_ids": (4, 3), "item_ids": (4, 3), "dense": (4, 3)},
        input_dtypes={"user_ids": "long", "item_ids": "long"},
        expected_output_shape=(4, 1),
        description="Two sparse embedding towers joined with dense features and an MLP scorer.",
        deployment_note="Export examples materialize long ID tensors so the embedding contract is real.",
        operator_surface=(
            "Embedding",
            "Linear",
            "cat",
            "mean",
            "relu",
        ),
    ),
    GalleryModelSpec(
        name="speech_conv_gru_encoder",
        family="Speech",
        input_shapes={"x": (2, 80, 32)},
        input_dtypes={},
        expected_output_shape=(2, 12),
        description="Log-mel style Conv1d subsampler feeding a batch-first GRU and classifier.",
        deployment_note="Represents speech encoders with time-axis subsampling and recurrent state contracts.",
        operator_surface=(
            "Conv1d",
            "GRU",
            "Linear",
            "mean",
            "relu",
            "transpose",
        ),
    ),
)

GATE_SPECS: Tuple[GateSpec, ...] = (
    GateSpec(
        name="pre_export_fx",
        phase="before_export",
        backend="fx",
        gate="src.fx_extractor.verify_module(backend='fx')",
        required=True,
        description="Verify the live nn.Module before handing it to an exporter.",
    ),
    GateSpec(
        name="post_export_torch_export",
        phase="after_export",
        backend="torch.export",
        gate="src.export_extractor.verify_module_export",
        required=False,
        description="Verify the graph captured by torch.export when the backend is available.",
    ),
)


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _prod(shape: Sequence[int]) -> int:
    return int(reduce(mul, shape, 1))


def _float_example(shape: Sequence[int]) -> Any:
    import torch

    return torch.zeros(*shape)


def _long_example(shape: Sequence[int], modulo: int) -> Any:
    import torch

    return (torch.arange(_prod(shape), dtype=torch.long) % modulo).reshape(*shape)


def _build_model(spec_name: str) -> Tuple[Any, Tuple[Any, ...]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if spec_name == "resnet_residual_stage":
        class ResNetResidualStage(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(8)
                self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(8)
                self.proj = nn.Conv2d(3, 8, 1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(8, 5)

            def forward(self, x: Any) -> Any:
                y = F.relu(self.bn1(self.conv1(x)))
                y = self.bn2(self.conv2(y))
                y = F.relu(y + self.proj(x))
                return self.fc(torch.flatten(self.pool(y), 1))

        return ResNetResidualStage().eval(), (_float_example((1, 3, 32, 32)),)

    if spec_name == "vit_patch_mixer":
        class ViTPatchMixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch = nn.Conv2d(3, 16, 4, stride=4)
                self.norm = nn.LayerNorm(16)
                self.mlp = nn.Linear(16, 16)
                self.head = nn.Linear(16, 3)

            def forward(self, x: Any) -> Any:
                x = self.patch(x).flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = F.gelu(self.mlp(x))
                return self.head(x.mean(dim=1))

        return ViTPatchMixer().eval(), (_float_example((1, 3, 32, 32)),)

    if spec_name == "llama_style_mlp_block":
        class LlamaStyleMLPBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(64, 16)
                self.norm = nn.LayerNorm(16)
                self.gate = nn.Linear(16, 32)
                self.up = nn.Linear(16, 32)
                self.down = nn.Linear(32, 16)

            def forward(self, tokens: Any) -> Any:
                x = self.norm(self.embed(tokens))
                y = F.silu(self.gate(x)) * self.up(x)
                return self.down(y).mean(dim=1)

        return LlamaStyleMLPBlock().eval(), (_long_example((2, 8), 64),)

    if spec_name == "diffusion_unet_skip":
        class DiffusionUNetSkip(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc = nn.Conv2d(4, 8, 3, padding=1)
                self.down = nn.Conv2d(8, 16, 3, stride=2, padding=1)
                self.mid = nn.Conv2d(16, 16, 3, padding=1)
                self.up = nn.ConvTranspose2d(16, 8, 2, stride=2)
                self.out = nn.Conv2d(16, 4, 1)

            def forward(self, x: Any) -> Any:
                skip = F.relu(self.enc(x))
                z = F.relu(self.down(skip))
                z = F.relu(self.mid(z))
                z = F.relu(self.up(z))
                return self.out(torch.cat([z, skip], dim=1))

        return DiffusionUNetSkip().eval(), (_float_example((1, 4, 16, 16)),)

    if spec_name == "recommender_two_tower":
        class RecommenderTwoTower(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.user = nn.Embedding(100, 8)
                self.item = nn.Embedding(200, 8)
                self.fc = nn.Sequential(
                    nn.Linear(19, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, user_ids: Any, item_ids: Any, dense: Any) -> Any:
                user = self.user(user_ids).mean(dim=1)
                item = self.item(item_ids).mean(dim=1)
                return self.fc(torch.cat([user, item, dense], dim=1))

        return RecommenderTwoTower().eval(), (
            _long_example((4, 3), 100),
            _long_example((4, 3), 200),
            _float_example((4, 3)),
        )

    if spec_name == "speech_conv_gru_encoder":
        class SpeechConvGRUEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(80, 32, 3, stride=2, padding=1)
                self.rnn = nn.GRU(32, 16, batch_first=True)
                self.fc = nn.Linear(16, 12)

            def forward(self, x: Any) -> Any:
                x = F.relu(self.conv(x))
                x = x.transpose(1, 2)
                y, _ = self.rnn(x)
                return self.fc(y.mean(dim=1))

        return SpeechConvGRUEncoder().eval(), (_float_example((2, 80, 32)),)

    raise ValueError(f"unknown deployment gallery model: {spec_name}")


def _param_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _normalise_result(result: Any) -> Tuple[bool, List[str]]:
    if result is None:
        return False, ["TensorGuard abstained without a verification result"]

    errors_attr = getattr(result, "errors", None)
    if callable(errors_attr):
        raw_errors = errors_attr()
    else:
        raw_errors = errors_attr or []
    errors = [getattr(err, "message", str(err)) for err in raw_errors]

    safe_attr = getattr(result, "safe", None)
    if isinstance(safe_attr, bool):
        return safe_attr and not errors, errors
    if isinstance(safe_attr, str):
        return safe_attr.upper() == "SAFE" and not errors, errors

    verdict = getattr(result, "verdict", None)
    if verdict is not None:
        verdict_text = str(verdict).upper()
        bugs = getattr(result, "bugs", None) or []
        bug_messages = [getattr(bug, "message", str(bug)) for bug in bugs]
        return verdict_text == "SAFE" and not bug_messages, errors + bug_messages

    return False, [f"unrecognized TensorGuard result type: {type(result).__name__}"]


def _shape_list_map(input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, List[int]]:
    return {
        name: list(shape) for name, shape in sorted(input_shapes.items())
    }


def _model_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        model, _ = _build_model(spec.name)
        rows.append({
            "model": spec.name,
            "family": spec.family,
            "description": spec.description,
            "deployment_note": spec.deployment_note,
            "input_shapes": _shape_list_map(spec.input_shapes),
            "input_dtypes": dict(sorted(spec.input_dtypes.items())),
            "expected_output_shape": list(spec.expected_output_shape),
            "parameter_count": _param_count(model),
            "operator_surface": list(spec.operator_surface),
        })
    return rows


def gate_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        for gate in GATE_SPECS:
            rows.append({
                "model": spec.name,
                "family": spec.family,
                "phase": gate.phase,
                "backend": gate.backend,
                "gate": gate.gate,
                "required": gate.required,
                "description": gate.description,
            })
    rows.sort(key=lambda row: (
        str(row["model"]),
        str(row["phase"]),
        str(row["backend"]),
    ))
    return rows


def manifest() -> Dict[str, object]:
    families = sorted({spec.family for spec in MODEL_SPECS})
    return {
        "meta": {
            "generated_by": "evaluation/deployment_gallery.py",
            "command": "PYTHONPATH=. python3 evaluation/deployment_gallery.py",
            "gate_command": "PYTHONPATH=. python3 evaluation/deployment_gallery.py --gate",
            "note": (
                "Deterministic real-model deployment gallery. The live gate executes "
                "each nn.Module, verifies the pre-export FX graph, and verifies the "
                "post-export torch.export graph when that backend is available."
            ),
        },
        "families": families,
        "models": _model_rows(),
        "gate_rows": gate_rows(),
        "reproducible_artifacts": [
            "evaluation/deployment_gallery.json",
            "evaluation/deployment_gallery.md",
        ],
    }


def render_markdown(man: Dict[str, object]) -> str:
    lines = [
        "# Real-model deployment gallery",
        "",
        (
            "Each gallery model is a real `nn.Module` that runs under eager PyTorch, "
            "then passes a pre-export FX gate and a post-export `torch.export` gate "
            "when that backend is available."
        ),
        "",
        "## Model gallery",
        "",
        "| Model | Family | Inputs | Output | Operator surface |",
        "|-------|--------|--------|--------|------------------|",
    ]
    for row in man["models"]:
        inputs = ", ".join(
            f"`{name}:{tuple(shape)}`"
            for name, shape in row["input_shapes"].items()
        )
        operators = ", ".join(f"`{op}`" for op in row["operator_surface"])
        lines.append(
            "| `{model}` | {family} | {inputs} | `{output}` | {operators} |".format(
                model=row["model"],
                family=row["family"],
                inputs=inputs,
                output=tuple(row["expected_output_shape"]),
                operators=operators,
            )
        )
    lines.extend(["", "## Export gate matrix", ""])
    lines.append("| Model | Phase | Backend | Required | Gate |")
    lines.append("|-------|-------|---------|----------|------|")
    for row in man["gate_rows"]:
        lines.append(
            "| `{model}` | {phase} | `{backend}` | {required} | `{gate}` |".format(
                model=row["model"],
                phase=row["phase"],
                backend=row["backend"],
                required="yes" if row["required"] else "if available",
                gate=row["gate"],
            )
        )
    lines.extend([
        "",
        "## Rebuild",
        "",
        "```bash",
        "make deployment-gallery",
        "make deployment-gallery-gate",
        "```",
        "",
    ])
    return "\n".join(lines)


def _expected_shape(spec: GalleryModelSpec, output: Any) -> Tuple[int, ...]:
    actual = tuple(int(dim) for dim in getattr(output, "shape", ()))
    if actual != spec.expected_output_shape:
        raise RuntimeError(
            f"{spec.name} eager output shape {actual} != expected {spec.expected_output_shape}"
        )
    return actual


def _run_fx(spec: GalleryModelSpec, model: Any) -> Tuple[bool, List[str]]:
    from src.fx_extractor import verify_module

    return _normalise_result(
        verify_module(
            model,
            input_shapes=spec.input_shapes,
            backend="fx",
            input_dtypes=spec.input_dtypes or None,
        )
    )


def _run_export(
    spec: GalleryModelSpec,
    model: Any,
    examples: Tuple[Any, ...],
) -> Tuple[bool, List[str]]:
    from src.export_extractor import HAS_EXPORT, verify_module_export

    if not HAS_EXPORT:
        raise BackendUnavailable("torch.export is unavailable")
    return _normalise_result(
        verify_module_export(
            model,
            input_shapes=spec.input_shapes,
            example_inputs=examples,
        )
    )


def _measure_gate(
    spec: GalleryModelSpec,
    gate: GateSpec,
    model: Any,
    examples: Tuple[Any, ...],
    output_shape: Tuple[int, ...],
) -> Dict[str, object]:
    runner: Callable[[], Tuple[bool, List[str]]]
    if gate.name == "pre_export_fx":
        runner = lambda: _run_fx(spec, model)
    elif gate.name == "post_export_torch_export":
        runner = lambda: _run_export(spec, model, examples)
    else:
        raise ValueError(f"unknown deployment gallery gate: {gate.name}")

    row = {
        "model": spec.name,
        "family": spec.family,
        "phase": gate.phase,
        "backend": gate.backend,
        "gate": gate.gate,
        "required": gate.required,
        "eager_output_shape": list(output_shape),
    }
    try:
        safe, errors = runner()
    except BackendUnavailable as exc:
        return {
            **row,
            "status": "skipped",
            "skip_reason": str(exc),
            "safe": True,
            "errors": [],
        }
    except Exception as exc:
        return {
            **row,
            "status": "failed",
            "safe": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    return {
        **row,
        "status": "passed" if safe else "failed",
        "safe": safe,
        "errors": errors,
    }


def measure() -> List[Dict[str, object]]:
    try:
        import torch
    except Exception as exc:
        return [
            {
                "model": spec.name,
                "family": spec.family,
                "phase": gate.phase,
                "backend": gate.backend,
                "gate": gate.gate,
                "required": gate.required,
                "status": "skipped",
                "skip_reason": f"PyTorch unavailable: {exc}",
                "safe": True,
                "errors": [],
            }
            for spec in MODEL_SPECS
            for gate in GATE_SPECS
        ]

    rows: List[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        model, examples = _build_model(spec.name)
        try:
            with torch.no_grad():
                output_shape = _expected_shape(spec, model(*examples))
        except Exception as exc:
            for gate in GATE_SPECS:
                rows.append({
                    "model": spec.name,
                    "family": spec.family,
                    "phase": gate.phase,
                    "backend": gate.backend,
                    "gate": gate.gate,
                    "required": gate.required,
                    "status": "failed",
                    "safe": False,
                    "errors": [f"eager forward failed: {type(exc).__name__}: {exc}"],
                })
            continue
        for gate in GATE_SPECS:
            rows.append(_measure_gate(spec, gate, model, examples, output_shape))
    return rows


def gate() -> int:
    rows = measure()
    failures = [row for row in rows if row["status"] == "failed"]
    skipped = [row for row in rows if row["status"] == "skipped"]
    passed = [row for row in rows if row["status"] == "passed"]

    for row in rows:
        if row["status"] == "skipped":
            print(
                "  [skip] {model:26s} {phase:<13s} {backend:12s} {reason}".format(
                    model=str(row["model"]),
                    phase=str(row["phase"]),
                    backend=str(row["backend"]),
                    reason=str(row["skip_reason"]),
                )
            )
            continue
        flag = "ok" if row["status"] == "passed" else "FAIL"
        print(
            "  [{flag}] {model:26s} {phase:<13s} {backend:12s} output={shape}".format(
                flag=flag,
                model=str(row["model"]),
                phase=str(row["phase"]),
                backend=str(row["backend"]),
                shape=tuple(row.get("eager_output_shape", ())),
            )
        )

    if failures:
        print("DEPLOYMENT GALLERY GATE FAILED: %d row(s)" % len(failures))
        for row in failures:
            details = "; ".join(str(err) for err in row.get("errors", [])[:3])
            print(
                "  - {model} {phase} {backend}: {details}".format(
                    model=row["model"],
                    phase=row["phase"],
                    backend=row["backend"],
                    details=details or "unsafe",
                )
            )
        return 1
    print(
        "deployment gallery gate PASS: %d checked, %d skipped optional backend row(s)"
        % (len(passed), len(skipped))
    )
    return 0


def run(check: bool = False, write: bool = True) -> int:
    man = manifest()
    text = _dumps(man)
    md = render_markdown(man)

    if check:
        ok = True
        if not os.path.exists(JSON_PATH) or open(JSON_PATH).read() != text:
            print("deployment_gallery.json is stale; run `make deployment-gallery`")
            ok = False
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("deployment_gallery.md is stale; run `make deployment-gallery`")
            ok = False
        if ok:
            print("deployment gallery manifest up to date")
        return 0 if ok else 1

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(md)
    print(
        "deployment gallery manifest written: %d models, %d gate rows"
        % (len(man["models"]), len(man["gate_rows"]))
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="assert committed gallery artifacts are fresh")
    parser.add_argument("--gate", action="store_true",
                        help="run the live deployment gallery gates")
    args = parser.parse_args(argv)
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
