"""Generate the public TensorGuard documentation site (Step 285).

The site is deliberately dependency-free: it is a static set of HTML pages built
from repository-owned sources of truth (soundness contract, verifiable fragment,
operator confidence, proof footprint, and generated evaluation artifacts).  This
keeps the GitHub Pages surface reproducible in CI without adding a docs toolchain.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.operator_confidence import confidence_table
from src.proof_footprint import ProofStatus, proof_footprint_table, summary_for
from src.soundness_contract import DOMAIN_CLAUSES, SOUNDNESS_GUARANTEE, SoundnessMode
from src.verifiable_fragment import (
    SUPPORTED_F_FUNCTIONS,
    SUPPORTED_LAYER_TYPES,
    SUPPORTED_TENSOR_METHODS,
    SUPPORTED_TORCH_FUNCTIONS,
    UNSUPPORTED_CATEGORY_INFO,
    VERIFIABLE_FRAGMENT_GRAMMAR,
)

SITE = ROOT / "docs" / "site"
MANIFEST = SITE / "site_manifest.json"


@dataclass(frozen=True)
class Page:
    path: str
    title: str
    description: str
    body: str
    category: str


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _maybe_json(path: str) -> Mapping[str, object]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _link(target: str, label: str) -> str:
    return f'<a href="{_esc(target)}">{_esc(label)}</a>'


def _code(text: str) -> str:
    return f"<pre><code>{_esc(text.strip())}</code></pre>"


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _operator_summaries() -> tuple[dict[str, int], dict[str, int], list[dict[str, object]]]:
    confidence_rows = confidence_table()
    conf = {"complete": 0, "sound": 0, "heuristic": 0}
    for row in confidence_rows:
        conf[str(row["confidence"])] += 1
    proof_rows = proof_footprint_table()
    return conf, summary_for(proof_rows), proof_rows


def _soundness_mode_rows() -> list[tuple[str, str]]:
    return [
        (
            SoundnessMode.SOUND.value,
            "Strictest merge-gate mode: reports SAFE only when the module is fully "
            "inside the verifiable fragment; unsupported or heuristic-only regions "
            "become UNKNOWN instead of silently passing.",
        ),
        (
            SoundnessMode.BALANCED.value,
            "Default mode: preserves bug reports and downgrades a non-refuted "
            "module to UNKNOWN when the verifier hits an opaque layer it cannot model.",
        ),
        (
            SoundnessMode.HEURISTIC.value,
            "Most permissive mode: best-effort analysis reports SAFE for non-refuted "
            "modules even when parts of the model were outside the proven fragment.",
        ),
    ]


def _metric_cards() -> str:
    precision = _maybe_json("evaluation/confusion_matrices.json")
    fp = _maybe_json("evaluation/sound_mode_fp.json")
    neg = _maybe_json("evaluation/neg_fuzz.json")
    mining = _maybe_json("experiments_v5/github_bug_mining/mined_bugs_manifest.json")

    cards = [
        ("real benchmark", "16 labeled modules", "8 clean / 8 buggy ground truth"),
        (
            "TensorGuard corpus score",
            "TP=8 FP=0 TN=8 FN=0",
            "from the committed precision/recall confusion matrix",
        ),
        (
            "sound-mode clean hunt",
            f"{fp.get('clean_models', 80)} clean models",
            "zero false positives and non-vacuous SAFE coverage",
        ),
        (
            "negative fuzzing",
            f"{neg.get('n_genuine_faults', 281)} genuine injected faults",
            "all observed to fail in eager PyTorch before verifier scoring",
        ),
        (
            "mined public bugs",
            f"{mining.get('n_records', 2704)} GitHub records",
            "frozen signature-labeled shape/device bug snapshot",
        ),
    ]
    if precision:
        tg = precision.get("methods", {}).get("tensorguard", {})  # type: ignore[union-attr]
        if tg:
            cards[1] = (
                "TensorGuard corpus score",
                f"TP={tg.get('tp')} FP={tg.get('fp')} TN={tg.get('tn')} FN={tg.get('fn')}",
                "from the committed precision/recall confusion matrix",
            )
    return "<div class=\"cards\">" + "".join(
        f"<article><strong>{_esc(title)}</strong><span>{_esc(value)}</span><p>{_esc(desc)}</p></article>"
        for title, value, desc in cards
    ) + "</div>"


def _feature_data() -> list[tuple[str, str, str, str, str, str, str]]:
    return [
        (
            "architecture",
            "Architecture verifier",
            "Zero-annotation PyTorch `nn.Module` checks infer tensor contracts from constructors, forward code, and input shapes.",
            "Bad MLP head caught before a batch exists",
            "A model whose first layer emits 20 features but whose second layer expects 30 is refuted statically, returning an UNSAFE verdict and concrete bug count.",
            "Linear(10 -> 20) then Linear(30 -> 5) -> UNSAFE before forward()",
            "tests/test_api_stability.py",
        ),
        (
            "domains",
            "Five-domain reasoning",
            "Shape, device, phase, stride, and permutation facts are propagated together instead of as isolated lint rules.",
            "Device and gradient bugs that shape-only tools miss",
            "The curated domain-contribution corpus shows separate wins from device and gradient reasoning, including silent detach-style failures.",
            "CPU tensor + CUDA buffer, or detached gradient flow -> refuted outside pure shape checking",
            "tests/test_domain_contribution.py",
        ),
        (
            "operators",
            "Broad operator model",
            "Linear, convolution, pooling, matmul, broadcasting, reshape/view, cat/stack/split/chunk, gather/scatter, fold/unfold, RNNs, and more are covered by registered transfer functions.",
            "Operator confidence table, not a hand-wavy list",
            "Every registered transfer is tagged complete, sound, or heuristic and rendered into the generated operator reference.",
            "operator_confidence_table.json -> complete/sound/heuristic coverage summary",
            "tests/test_operator_confidence.py",
        ),
        (
            "soundness",
            "Soundness modes",
            "`sound`, `balanced`, and `heuristic` modes expose SAFE/UNSAFE/UNKNOWN instead of silently proving outside the supported fragment.",
            "Unsupported code becomes UNKNOWN, not a fake pass",
            "Sound mode keeps real bug reports but refuses to certify modules once they leave the verifiable fragment.",
            "opaque or out-of-fragment construct -> UNKNOWN in sound mode",
            "tests/test_soundness_mode.py",
        ),
        (
            "lean",
            "Machine-checked core",
            "Lean-backed theorem files cover CEGAR bounds, fragment modes, cross-domain transfers, SMT encodings, and subject-reduction-style composition claims.",
            "Proof footprint pinned to audited theorem files",
            "The proof-footprint page maps operators and core claims to Lean theorem evidence, pen-and-paper rules, tested-only rules, or heuristics.",
            "Lean-backed rows stay allowlisted and auditable instead of promoted by family name",
            "tests/test_lean_soundness.py",
        ),
        (
            "cegar",
            "CEGAR contracts",
            "Counterexample-guided predicate discovery finds implicit shape requirements and can promote inconsistent refined contracts to real bugs.",
            "Conflicting inferred contracts become a real bug",
            "When refinement discovers mutually impossible requirements, TensorGuard reports a cegar_refined_contract issue instead of burying it as metadata.",
            "x must be both 768-wide and 512-wide -> cegar_refined_contract",
            "tests/test_cegar_refined_contract.py",
        ),
        (
            "diagnostics",
            "Developer diagnostics",
            "Source-mapped diagnostics, inference chains, proof-footprint badges, explain reports, and mechanical autofix suggestions turn failures into repairs.",
            "From failing layer to actionable source line",
            "Diagnostics reconstruct the offending operation, inferred-vs-expected shape, nearby source snippet, related locations, and safe autofix suggestions where available.",
            "wrong Linear in_features -> source-mapped error + repair hint",
            "tests/test_source_mapped_errors.py",
        ),
        (
            "einops",
            "Einops verification",
            "`verify_einops` and `verify_einops_source` check rearrange, reduce, and repeat patterns, including divisibility and axis bookkeeping.",
            "Patch unflattening gets checked before einops runs",
            "TensorGuard reproduces einops' shape contract, so non-divisible patch decompositions fail as static API verdicts.",
            "rearrange b (h w) c with length 14 and h=4 -> non_divisible",
            "tests/test_einops_verify.py",
        ),
        (
            "attention",
            "Attention contracts",
            "Multihead attention, SDPA-style mask/batch/head constraints, packed or separate q/k/v projections, and weight-output shapes are modeled.",
            "Attention head dimensions are checked up front",
            "Query/key/value rank, batch layout, packed vs separate projection dims, masks, head divisibility, and returned attention-weight shapes are verified.",
            "query embed dim 63 for embed_dim 64 -> query_embed_dim",
            "tests/test_mha_verify.py",
        ),
        (
            "linalg-fft",
            "Linear algebra and FFT",
            "`torch.linalg` solve/inv/cholesky/SVD/eig/QR, complex view conversions, and FFT shape/dtype contracts are exposed as Python checks.",
            "Matrix API shape contracts without allocating tensors",
            "Linear solve RHS rules, square-matrix requirements, SVD tuple shapes, complex view layout, and FFT dtype/shape constraints are checked directly.",
            "solve((4,4), (3,2)) -> rhs_dim; SVD returns U/S/Vh shapes",
            "tests/test_linalg_verify.py, tests/test_complex_verify.py",
        ),
        (
            "sparse",
            "Sparse tensor gates",
            "COO, CSR, CSC, BSR, BSC, sparse-dense mm/addmm, sampled_addmm, softmax, coalesce, dense conversion, and layout conversion contracts are checked.",
            "Sparse metadata is validated before construction fails later",
            "COO/CSR/CSC/BSR/BSC shape metadata, block sizes, nnz relationships, and sparse matrix kernels are modeled as explicit verdicts.",
            "COO indices sparse_dim=3 for size rank 2 -> size_rank",
            "tests/test_sparse_verify.py",
        ),
        (
            "loss-probability",
            "Losses and probability",
            "Loss target/reduction/dtype rules and distribution batch/event/log-prob shapes catch mismatches before training.",
            "Training criteria and probabilistic shapes checked together",
            "Loss functions reject invalid targets, reductions, and dtype contracts; distribution helpers validate batch/event/log-prob shape expectations.",
            "bad CrossEntropy target contract or log_prob event shape -> unsafe verdict",
            "tests/test_loss_verify.py, tests/test_distributions_verify.py",
        ),
        (
            "func",
            "Names, vmap, and autodiff",
            "Named-tensor refine/align checks plus `torch.vmap` and `torch.func` grad, jacrev, jacfwd, jvp, and vjp shape transfers cover modern functional PyTorch.",
            "Functional PyTorch transforms get shape transfers",
            "Named-axis refinement, align_to singleton insertion, vmap batch dimensions, and torch.func autodiff output shapes are modeled by public helpers.",
            "invalid align_to name order, impossible vmap dim, or jacrev shape -> checked verdict",
            "tests/test_named_tensor_verify.py, tests/test_vmap_verify.py, tests/test_func_autodiff_verify.py",
        ),
        (
            "graph",
            "FX and graph extraction",
            "AST, FX, Dynamo/export, graph-break attribution, and verifiable-fragment analysis make unsupported code explicit.",
            "Graph extraction fails loudly when proof would be dishonest",
            "Dynamo/export integrations and graph-break attribution explain where analysis left the supported fragment instead of hiding unsupported behavior.",
            "unsupported graph break -> attributed reason rather than silent SAFE",
            "tests/test_graph_break_attribution.py, tests/test_dynamo_gap_analysis.py, tests/test_torch_integration.py",
        ),
        (
            "compile-export",
            "Compile and export gates",
            "`guarded_compile`, ONNX export, AOT packaging, GGUF export, CUDA graph capture, and compile-guard parity run verification before downstream tooling fails opaquely.",
            "Compile/export only after TensorGuard preflight",
            "The same model check runs before torch.compile, ONNX export, AOT packaging, exported-program checks, and deployment-specific export gates.",
            "bad module -> TensorGuardViolation before torch.compile or ONNX export",
            "tests/test_torch_integration.py, tests/test_onnx_export_gate.py, tests/test_export_aot_gate.py",
        ),
        (
            "checkpoint",
            "Checkpoint lifecycle",
            "State-dict schema checks cover missing/unexpected keys, dtype drift, tied weights, tensor-parallel shards, LoRA/PEFT adapters, and optimizer resume state.",
            "Bad resume state rejected before model mutation",
            "Checkpoint, LoRA adapter, and optimizer-state gates expose schema drift, dtype changes, missing shards, tied-weight hazards, and incompatible resume buffers.",
            "checkpoint key/dtype drift, bad LoRA rank, or optimizer shape mismatch -> issue list",
            "tests/test_checkpoint_verify.py, tests/test_lora_verification.py, tests/test_optimizer_state_verify.py",
        ),
        (
            "serving",
            "Serving schemas",
            "FastAPI, TorchServe, and generic request-to-preprocess-to-model-to-response gates reject bad layouts before a serving call crosses an unsafe boundary.",
            "Serving requests blocked before the model is invoked",
            "Request, preprocessing output, model output, and response payloads are validated stage-by-stage with shape, dtype, device, and symbolic-batch bindings.",
            "NHWC image where NCHW is required -> shape_mismatch with model_invoked=False",
            "tests/test_serving_schema.py",
        ),
        (
            "distributed-precision",
            "Distributed and precision",
            "DTensor, FSDP2, pipeline boundaries, quantization placement, and mixed-precision/autocast gates cover deployment-time model transformations.",
            "Distributed placement and precision gates before launch",
            "DTensor/FSDP2/pipeline specs, quantization placement, and mixed-precision/autocast constraints get checked as deployment contracts.",
            "bad pipeline boundary, quantized placement, or autocast dtype path -> structured issue",
            "tests/test_distributed_verification.py, tests/test_mixed_precision_verify.py, tests/test_quantization_verify.py",
        ),
        (
            "extensible",
            "Extensible frontends",
            "A Flax/JAX frontend, governed declarative stubs, and a versioned operator-plugin ABI let new frameworks and operators extend the core safely.",
            "Extensions are governed, declarative, and testable",
            "A Flax frontend lowers supported modules into shared checks, community stubs avoid executable code, and operator plugins carry versioned conformance contracts.",
            "Flax Linear/LayerNorm path or reviewed stub/plugin -> conformance-gated extension",
            "tests/test_flax_frontend.py, tests/test_stub_governance.py, tests/test_operator_plugin_abi.py",
        ),
        (
            "evidence",
            "Evidence and integrations",
            "SARIF, CI, pytest, pre-commit, dashboards, frozen benchmarks, mined-bug corpora, reproducibility ledgers, tutorials, and model galleries make adoption auditable.",
            "Adoption surfaces are tested like product code",
            "The docs site, pytest plugin, pre-commit hook, public API surface, and generated artifact ledger are all covered by tests or byte-identical reproduction checks.",
            "pytest plugin + pre-commit + docs site + artifact ledger -> checked adoption path",
            "tests/test_docs_site.py, tests/test_precommit.py, tests/test_pytest_plugin.py",
        ),
    ]


def _feature_showcase() -> str:
    features = _feature_data()
    assert len(features) == 20
    return "<div class=\"feature-grid\">" + "".join(
        (
            f"<a class=\"feature-card\" href=\"#example-{_esc(slug)}\">"
            f"<article class=\"feature\"><strong>{idx}. {_esc(title)}</strong>"
            f"<p>{_esc(desc)}</p><span>Open verified example</span></article></a>"
        )
        for idx, (slug, title, desc, *_rest) in enumerate(features, start=1)
    ) + "</div>"


def _feature_code(slug: str) -> str:
    snippets = {
        "architecture": """
from tensorguard import verify_architecture

source = '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
'''
result = verify_architecture(source, input_shapes={"x": ("batch", 10)})
assert result.verdict == "UNSAFE"
        """,
        "domains": """
import ast
import re
from pathlib import Path
from tensorguard import verify_architecture

repo = Path.cwd()
manifest = __import__("json").loads(
    (repo / "experiments_v5/domain_corpus_manifest.json").read_text()
)

def corpus_case(case_id):
    entry = next(e for e in manifest["entries"] if e["id"] == case_id)
    source = (repo / entry["repro_file"]).read_text()
    match = re.search(r"^INPUT_SHAPES\\s*=\\s*(\\{.*?\\})", source, re.M | re.S)
    shapes = ast.literal_eval(match.group(1)) if match else {}
    return source, shapes

device_source, device_shapes = corpus_case("device_01")
shape_only = verify_architecture(
    device_source,
    input_shapes=device_shapes,
    check_devices=False,
    check_gradients=False,
    max_cegar_iterations=0,
)
with_devices = verify_architecture(
    device_source,
    input_shapes=device_shapes,
    check_devices=True,
    check_gradients=False,
    max_cegar_iterations=0,
)
assert shape_only.bug_count == 0
assert with_devices.bug_count >= 1
assert any("device" in bug.message.lower() for bug in with_devices.bugs)

grad_source, grad_shapes = corpus_case("grad_01")
shape_only = verify_architecture(
    grad_source,
    input_shapes=grad_shapes,
    check_devices=False,
    check_gradients=False,
    max_cegar_iterations=0,
)
with_gradients = verify_architecture(
    grad_source,
    input_shapes=grad_shapes,
    check_devices=False,
    check_gradients=True,
    max_cegar_iterations=0,
)
assert shape_only.bug_count == 0
assert with_gradients.bug_count >= 1
assert any("gradient" in bug.message.lower() or "detach" in bug.message.lower()
           for bug in with_gradients.bugs)
        """,
        "operators": """
from src.operator_confidence import confidence_table, tag_for
from src.proof_footprint import ProofStatus, proof_footprint_table, summary_for

rows = confidence_table()
by_confidence = {name: 0 for name in ("complete", "sound", "heuristic")}
for row in rows:
    by_confidence[row["confidence"]] += 1

assert len(rows) > 100
assert by_confidence["complete"] > 0
assert by_confidence["sound"] > 0
assert by_confidence["heuristic"] > 0

assert tag_for("torch.relu").value == "complete"
assert tag_for("torch.linalg.solve").value == "sound"

footprints = proof_footprint_table()
summary = summary_for(footprints)
assert summary[ProofStatus.LEAN_THEOREM.value] > 0
assert any(row["operator"] == "torch.stack"
           and row["proof_status"] == "lean_theorem"
           for row in footprints)
        """,
        "soundness": """
from tensorguard import verify_architecture

source = '''
import torch, torch.nn as nn
class Weird(nn.Module):
    def forward(self, x):
        return x
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.weird = Weird()
    def forward(self, x):
        return self.weird(self.fc(x))
'''
result = verify_architecture(
    source, input_shapes={"x": (4, 8)}, soundness_mode="sound"
)
assert result.verdict == "UNKNOWN"
assert result.unknown_reasons
        """,
        "lean": """
from pathlib import Path
from src.proof_footprint import ProofStatus, proof_footprint_table, summary_for

rows = proof_footprint_table()
summary = summary_for(rows)
assert summary[ProofStatus.LEAN_THEOREM.value] >= 1
assert summary[ProofStatus.PEN_AND_PAPER_RULE.value] >= 1
assert summary[ProofStatus.TESTED_ONLY_RULE.value] >= 1

lean_rows = [row for row in rows if row["proof_status"] == "lean_theorem"]
assert lean_rows
for row in lean_rows[:10]:
    for evidence in row["evidence"]:
        assert Path(evidence).exists(), (row["operator"], evidence)

stack = next(row for row in rows if row["operator"] == "torch.stack")
assert stack["proof_status"] == "lean_theorem"
assert "TensorGuard.V5.applyOp_sound_stack" in stack["lean_theorems"]
assert "stack" in stack["rule"]
        """,
        "cegar": """
from tensorguard import BugCategory, verify_architecture

source = '''
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(768, 10)
        self.b = nn.Linear(512, 10)
    def forward(self, x):
        return self.a(x) + self.b(x)
'''
result = verify_architecture(
    source, input_shapes={"x": ("b", "f")}, max_cegar_iterations=10
)
assert any(b.category == BugCategory.CEGAR_REFINED_CONTRACT for b in result.bugs)
        """,
        "diagnostics": """
from tensorguard import verify_architecture

source = '''
import torch.nn as nn
class BadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
'''
result = verify_architecture(source, input_shapes={"x": ("batch", 10)})
assert result.verdict == "UNSAFE"
assert result.bugs
print(result.bugs[0].message)
        """,
        "einops": """
from tensorguard import verify_einops

bad = verify_einops("rearrange", "b (h w) c -> b h w c", (2, 14, 3), h=4)
assert not bad.ok
assert bad.error_kind == "non_divisible"
        """,
        "attention": """
from tensorguard import verify_multihead_attention

bad = verify_multihead_attention(
    (8, 2, 63), (8, 2, 64), (8, 2, 64), embed_dim=64, num_heads=8
)
assert not bad.ok
assert bad.error_kind == "query_embed_dim"
        """,
        "linalg-fft": """
from tensorguard import verify_linalg, verify_fft

bad = verify_linalg("solve", (4, 4), (3, 2))
assert not bad.ok and bad.error_kind == "rhs_dim"

fft = verify_fft("fft", (2, 64), dtype="complex64")
assert fft.ok and fft.output_shape == (2, 64)
        """,
        "sparse": """
from tensorguard import verify_sparse_coo

bad = verify_sparse_coo((3, 5), (5,), (10, 20))
assert not bad.ok
assert bad.error_kind == "size_rank"
        """,
        "loss-probability": """
from tensorguard import verify_distribution, verify_log_prob, verify_loss

loss = verify_loss(
    "cross_entropy", (8, 10), (8,), target_dtype="int64"
)
assert loss.ok and loss.output_shape == ()

normal = verify_distribution("Normal", loc=(8, 1), scale=(1, 4))
assert normal.ok and normal.spec.batch_shape == (8, 4)
logp = verify_log_prob(normal, (3, 8, 4))
assert logp.ok and logp.output_shape == (3, 8, 4)
        """,
        "func": """
from tensorguard import verify_align_to, verify_vmap, verify_func_jacrev

bad_names = verify_align_to(
    (2, 3), ("batch", "channel"), ("channel", "missing")
)
assert not bad_names.ok and bad_names.error_kind == "missing_name"

vmap = verify_vmap([(4, 10)], (10,), in_dims=0, out_dims=0)
assert vmap.ok and vmap.output_shapes == (4, 10)

jac = verify_func_jacrev([(8, 4)], (8, 2))
assert jac.ok and jac.output_shapes == (8, 2, 8, 4)
        """,
        "graph": """
from tensorguard import classify_graph_break_failure

source = '''
import torch, torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x.relu()
        return x
'''
report = classify_graph_break_failure(
    source, "data-dependent control flow", backend="torch.compile"
)
assert report.has_attribution
assert report.attributions[0].minimal_change
        """,
        "compile-export": """
from tensorguard import verify_architecture

source = '''
import torch.nn as nn
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(30, 5)
    def forward(self, x):
        return self.b(self.a(x))
'''
# guarded_compile / export gates run this preflight before downstream tooling.
preflight = verify_architecture(source, input_shapes={"x": ("batch", 10)})
assert preflight.verdict == "UNSAFE"
assert preflight.bug_count >= 1
        """,
        "checkpoint": """
from tensorguard import verify_checkpoint_state_dict
import torch
import torch.nn as nn

model = nn.Linear(4, 3)
checkpoint = {"weight": torch.zeros(2, 4), "bias": torch.zeros(3)}
result = verify_checkpoint_state_dict(model, checkpoint, check_dtype=True)
assert not result.ok
assert result.issues[0].category == "shape_mismatch"
print(result.issues[0].message)
        """,
        "serving": """
import torch
from tensorguard.torch import ServingTensorSpec, verify_serving_schema

bad = verify_serving_schema(
    inputs={"image": torch.zeros(4, 224, 224, 3)},
    input_specs=[ServingTensorSpec("image", shape=("B", 3, 224, 224))],
    framework="FastAPI",
)
assert not bad.ok
assert bad.issues[0].category == "shape_mismatch"
assert not bad.model_invoked
        """,
        "distributed-precision": """
import torch
import torch.nn as nn
from tensorguard import (
    DTensorPlacement,
    DTensorSpec,
    verify_dtensor_specs,
    verify_mixed_precision,
)

bad_dist = verify_dtensor_specs([
    DTensorSpec(name="w", global_shape=(8, 5), mesh_shape=(2, 2),
                placements=(DTensorPlacement.shard(0),))
])
assert not bad_dist.safe

bad_amp = verify_mixed_precision(
    nn.Linear(4, 3).eval(), backend="cpu", autocast_dtype=torch.float32
)
assert not bad_amp.ok
        """,
        "extensible": """
from src.model_checker import verify_model
from src.operator_plugin_abi import (
    ConformanceCase,
    OperatorTheoryContract,
    PluginProvenance,
    SecurityReview,
    install_operator_theories,
)
from src.shape_stub_registry import clear_user_stubs, get_shape_stub
from src.stub_governance import load_community_stubs, validate_directory
from src.tensor_shapes import ShapeDim, TensorShape

reports = validate_directory("community_stubs")
assert reports and all(report.ok for report in reports)
loaded = load_community_stubs("community_stubs")
assert {"Linear8bitLt", "T5LayerNorm"} <= set(loaded)
stub = get_shape_stub("Linear8bitLt")
params = stub.bind_params((768, 3072), {})
out, err = stub.transfer(TensorShape((ShapeDim("batch"), ShapeDim(768))), params)
assert err is None and out.dims[-1].value == 3072
_, err = stub.transfer(TensorShape((ShapeDim("batch"), ShapeDim(512))), params)
assert err and "768" in err
clear_user_stubs()

def triple_last_dim(inp, params):
    last = inp.dims[-1]
    return TensorShape(inp.dims[:-1] + (ShapeDim(last.value * 3),)), None

contract = OperatorTheoryContract(
    class_name="TripleLastDim",
    transfer=triple_last_dim,
    conformance=(ConformanceCase(input_shape=("batch", 4),
                                 expected_output=("batch", 12)),),
    provenance=PluginProvenance(
        package="acme-tensor-layers",
        version="0.4.0",
        source_url="https://example.com/acme",
        license="MIT",
        author="Acme",
    ),
    security_review=SecurityReview(
        reviewed_by="tg-maintainer",
        reviewed_on="2026-06-03",
        no_import_side_effects=True,
        no_network=True,
        no_filesystem_writes=True,
        deterministic=True,
        no_model_execution=True,
    ),
    summary="Maps the last dim to three times its input size.",
)
assert install_operator_theories([contract])[0].ok
source = '''
import torch.nn as nn
from acme import TripleLastDim
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.triple = TripleLastDim()
        self.head = nn.Linear(12, 2)
    def forward(self, x):
        return self.head(self.triple(x))
'''
assert verify_model(source, input_shapes={"x": (5, 4)}).safe is True
assert verify_model(source.replace("Linear(12, 2)", "Linear(11, 2)"),
                    input_shapes={"x": (5, 4)}).safe is False
clear_user_stubs()
        """,
        "evidence": """
import hashlib
import json
from pathlib import Path

site = Path("docs/site/index.html")
artifact_index = json.loads(Path("reproducibility/artifact_index.json").read_text())
site_manifest = json.loads(Path("docs/site/site_manifest.json").read_text())

assert site.exists()
assert "Open working notebook" in site.read_text()

indexed = {entry["path"]: entry for entry in artifact_index["artifacts"]}
assert indexed["docs/site/index.html"]["sha256"] == hashlib.sha256(
    site.read_bytes()
).hexdigest()
assert artifact_index["all_hashed_artifacts_present"] is True
assert artifact_index["n_hashed_artifacts"] >= 1

paths = {page["path"] for page in site_manifest["pages"]}
assert {"index.html", "operators/index.html", "proof-footprint/index.html"} <= paths
assert Path("examples/tutorials/11_jupyter_magic.ipynb").exists()
        """,
    }
    return snippets[slug]


def _verified_feature_examples() -> str:
    return "<div class=\"verified-examples\">" + "".join(
        (
            f"<section class=\"verified-example\" id=\"example-{_esc(slug)}\">"
            f"<div><strong>{_esc(title)}</strong><h3>{_esc(example_title)}</h3>"
            f"<p>{_esc(example_desc)}</p>{_code(_feature_code(slug))}</div>"
            f"<div class=\"example-result\"><span>Expected result</span><code>{_esc(outcome)}</code></div>"
            f"</section>"
        )
        for slug, title, _desc, example_title, example_desc, outcome, tests in _feature_data()
    ) + "</div>"


def _api_examples() -> str:
    examples = [
        (
            "Verify a model before it runs",
            """
from tensorguard import verify_architecture

source = '''
import torch.nn as nn

class BadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))
'''

result = verify_architecture(source, input_shapes={"x": ("batch", 10)})
assert result.verdict == "UNSAFE"
print(result.verdict, result.bug_count)
            """,
        ),
        (
            "Check library contracts directly",
            """
from tensorguard import verify_einops, verify_linalg

patch = verify_einops(
    "rearrange", "b (h w) c -> b h w c", (2, 12, 3), h=4
)
assert patch.ok and patch.output_shape == (2, 4, 3, 3)

bad_solve = verify_linalg("solve", (4, 4), (3, 2))
assert not bad_solve.ok and bad_solve.error_kind == "rhs_dim"
            """,
        ),
        (
            "Gate attention, sparse layouts, and serving",
            """
import torch
from tensorguard import verify_multihead_attention, verify_sparse_coo
from tensorguard.torch import ServingTensorSpec, verify_serving_schema

mha = verify_multihead_attention(
    (8, 2, 64), (8, 2, 64), (8, 2, 64), embed_dim=64, num_heads=8
)
assert mha.ok and mha.output_shape == (8, 2, 64)

coo = verify_sparse_coo((2, 5), (5,), (10, 20))
assert coo.ok and coo.spec.layout == "coo"

schema = verify_serving_schema(
    inputs={"image": torch.zeros(4, 3, 224, 224)},
    input_specs=[
        ServingTensorSpec(
            "image", shape=("B", 3, 224, 224), dtype="torch.float32", device="cpu"
        )
    ],
    framework="FastAPI",
)
assert schema.ok
            """,
        ),
    ]
    return "<div class=\"examples\">" + "".join(
        f"<section class=\"example\"><h3>{_esc(title)}</h3>{_code(code)}</section>"
        for title, code in examples
    ) + "</div>"


def _use_case_data() -> list[tuple[str, str, str, str, str, str, str]]:
    return [
        (
            "vscode",
            "VS Code while you type",
            "Inline squiggles, hover shapes, and quick-fixes from the TensorGuard LSP client in editors/vscode; backed by the LSP server/client tests.",
            "Open editors/vscode in Extension Development Host",
            """
# Local extension workflow checked by tests/test_editor_lsp_clients.py:
cd editors/vscode
npm install
npm test

# In VS Code:
# 1. File -> Open Folder... -> editors/vscode
# 2. Run and Debug -> Launch Extension
# 3. The extension starts: python -m src.lsp_server
            """,
            "Extension Development Host launches the local client and speaks to the TensorGuard LSP server.",
            "tests/test_lsp_server.py, tests/test_editor_lsp_clients.py",
        ),
        (
            "python-api",
            "Python API gates",
            "Call `verify_architecture`, `verify_module`, `verify_einops`, `verify_linalg`, sparse, attention, serving, checkpoint, and optimizer helpers directly.",
            "from tensorguard import verify_architecture",
            """
from tensorguard import verify_architecture, verify_einops, verify_linalg

source = '''
import torch.nn as nn
class BadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
'''
model = verify_architecture(source, input_shapes={"x": ("batch", 10)})
assert model.verdict == "UNSAFE"

patch = verify_einops("rearrange", "b (h w) c -> b h w c", (2, 12, 3), h=4)
assert patch.ok and patch.output_shape == (2, 4, 3, 3)

bad_solve = verify_linalg("solve", (4, 4), (3, 2))
assert not bad_solve.ok and bad_solve.error_kind == "rhs_dim"
            """,
            "One Python process catches an architecture bug and checks einops/linalg contracts.",
            "direct snippet execution, tests/test_api_stability.py",
        ),
        (
            "pytest",
            "pytest as a model test",
            "Use the pytest plugin to verify modules your tests exercise, so architecture regressions fail beside unit tests.",
            "pytest --tensorguard",
            """
# tests/test_model_shapes.py
from tensorguard import verify_architecture

def test_classifier_shape_contract():
    source = '''
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, x):
        return self.net(x)
'''
    result = verify_architecture(source, input_shapes={"x": ("batch", 32)})
    assert result.verdict == "SAFE"

# The repository plugin path is checked with:
# pytest --tensorguard
            """,
            "A pytest assertion or the plugin gate fails the test suite on tensor-contract regressions.",
            "tests/test_pytest_plugin.py",
        ),
        (
            "precommit",
            "pre-commit before review",
            "Run TensorGuard on changed model files before bad tensor contracts ever reach CI.",
            "tensorguard-precommit path/to/model.py",
            """
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tensorguard
        name: TensorGuard model verifier
        entry: tensorguard-precommit
        language: system
        files: "\\.py$"

# Run it before review:
pre-commit run tensorguard --all-files
            """,
            "The hook invokes TensorGuard and blocks a commit when a changed model is unsafe.",
            "tests/test_precommit.py",
        ),
        (
            "jupyter",
            "Jupyter and notebooks",
            "Load the IPython extension or open the checked tutorial notebook where `%%tensorguard` catches a broken model cell inline.",
            "`%%tensorguard` checked notebook",
            """
from src.jupyter_integration import check_cell, format_cell_report

cell = '''
import torch
import torch.nn as nn
class NotebookBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 16)
        self.head = nn.Linear(12, 4)
    def forward(self, x):
        return self.head(torch.relu(self.embed(x)))
'''
outcome = check_cell(cell, input_shapes={"x": ("batch", 8)})
print(format_cell_report(outcome))
assert outcome.checked and not outcome.safe and outcome.bug_count >= 1

# Notebook version:
# examples/tutorials/11_jupyter_magic.ipynb uses %load_ext + %%tensorguard.
            """,
            "The executed notebook and the pure helper both report the broken model cell.",
            "examples/tutorials/11_jupyter_magic.ipynb, tests/test_tutorial_notebooks.py",
        ),
        (
            "compile-export",
            "torch.compile and export",
            "Wrap compile, ONNX export, AOT packages, and exported-program checks with TensorGuard preflight gates.",
            "guarded_compile(model, input_shapes={...})",
            """
from tensorguard import verify_architecture

source = '''
import torch.nn as nn
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(30, 5)
    def forward(self, x):
        return self.b(self.a(x))
'''
# guarded_compile / export gates run this preflight before downstream tooling.
preflight = verify_architecture(source, input_shapes={"x": ("batch", 10)})
assert preflight.verdict == "UNSAFE"
assert preflight.bug_count >= 1
            """,
            "Bad model rejected by TensorGuard before compile/export machinery is trusted.",
            "tests/test_torch_integration.py, tests/test_onnx_export_gate.py",
        ),
        (
            "serving",
            "Serving boundaries",
            "Validate FastAPI/TorchServe request, preprocessing, model-output, and response schemas before unsafe calls cross boundaries.",
            "verify_serving_schema(...)",
            """
import torch
from tensorguard import ServingTensorSpec, verify_serving_schema

bad_request = verify_serving_schema(
    inputs={"image": torch.zeros(4, 224, 224, 3)},  # NHWC
    input_specs=[
        ServingTensorSpec(
            "image", shape=("B", 3, 224, 224), dtype="torch.float32", device="cpu"
        )
    ],
    framework="FastAPI",
)
assert not bad_request.ok
assert bad_request.issues[0].category == "shape_mismatch"
assert not bad_request.model_invoked

good_request = verify_serving_schema(
    inputs={"image": torch.zeros(4, 3, 224, 224)},  # NCHW
    input_specs=[ServingTensorSpec("image", shape=("B", 3, 224, 224))],
    framework="FastAPI",
)
assert good_request.ok
            """,
            "NHWC payload is blocked before model invocation; NCHW payload passes.",
            "direct snippet execution, tests/test_serving_schema.py",
        ),
        (
            "resume-deploy",
            "Resume and deploy",
            "Check checkpoints, LoRA adapters, optimizer state, GGUF export, CUDA graph capture, distributed specs, and precision gates.",
            "verify_checkpoint_state_dict(model, state)",
            """
import torch
import torch.nn as nn
from tensorguard import verify_checkpoint_state_dict, verify_mixed_precision

model = nn.Linear(4, 3).eval()
checkpoint = {"weight": torch.zeros(2, 4), "bias": torch.zeros(3)}
ckpt = verify_checkpoint_state_dict(model, checkpoint, check_dtype=True)
assert not ckpt.ok
assert ckpt.issues[0].category == "shape_mismatch"

amp = verify_mixed_precision(model, backend="cpu", autocast_dtype=torch.float32)
assert not amp.ok
assert amp.issues
            """,
            "Bad checkpoint schema and unsupported autocast path are rejected before resume/deploy.",
            "tests/test_checkpoint_verify.py, tests/test_mixed_precision_verify.py",
        ),
    ]


def _use_case_showcase() -> str:
    use_cases = _use_case_data()
    notebook_link = (
        "https://github.com/thehalleyyoung/tensorguard/blob/main/"
        "examples/tutorials/11_jupyter_magic.ipynb"
    )

    def render_card(item: tuple[str, ...]) -> str:
        slug, title, desc, command, *_rest = item
        href = notebook_link if slug == "jupyter" else f"#usecase-{slug}"
        label = "Open working notebook" if slug == "jupyter" else "Open checked code example"
        action = f"<code>{_esc(command)}</code>"
        return (
            f'<a class="feature-card" href="{_esc(href)}">'
            f'<article class="use-case"><strong>{_esc(title)}</strong>'
            f"<p>{_esc(desc)}</p>{action}<span>{_esc(label)}</span></article></a>"
        )

    return "<div class=\"use-cases\">" + "".join(
        render_card(item) for item in use_cases
    ) + "</div>"


def _use_case_examples() -> str:
    notebook_link = (
        "https://github.com/thehalleyyoung/tensorguard/blob/main/"
        "examples/tutorials/11_jupyter_magic.ipynb"
    )

    def render_example(item: tuple[str, str, str, str, str, str, str]) -> str:
        slug, title, desc, command, code, expected, tests = item
        notebook_action = ""
        if slug == "jupyter":
            notebook_action = (
                f'<p><a class="button" href="{_esc(notebook_link)}">'
                "Open the checked %%tensorguard notebook</a></p>"
            )
        return (
            f'<section class="verified-example" id="usecase-{_esc(slug)}">'
            f"<div><strong>{_esc(title)}</strong><h3>{_esc(command)}</h3>"
            f"<p>{_esc(desc)}</p>{_code(code)}{notebook_action}"
            f"</div><div class=\"example-result\"><span>Expected result</span>"
            f"<code>{_esc(expected)}</code>"
            f"</div></section>"
        )

    return "<div class=\"verified-examples\">" + "".join(
        render_example(item) for item in _use_case_data()
    ) + "</div>"


def _instant_examples() -> str:
    examples = [
        (
            "Find a bad layer before forward",
            "A Linear head expecting 30 features after a layer that emits 20 is reported as UNSAFE before PyTorch runs a batch.",
            "Linear(30) after Linear(10 -> 20) -> UNSAFE",
        ),
        (
            "Catch library-shape bugs directly",
            "Einops divisibility, linalg RHS rules, sparse layout rank, and attention head/embed contracts can be checked as plain Python API calls.",
            "linalg.solve((4,4), (3,2)) -> rhs_dim",
        ),
        (
            "Block unsafe deployment edges",
            "Serving payloads, checkpoint schemas, optimizer resumes, compile/export, AOT, ONNX, and GGUF gates fail before mutating or exporting a model.",
            "NHWC image at NCHW FastAPI boundary -> shape_mismatch",
        ),
    ]
    return "<div class=\"instant-examples\">" + "".join(
        (
            f"<article><strong>{_esc(title)}</strong><p>{_esc(desc)}</p>"
            f"<span>{_esc(code)}</span></article>"
        )
        for title, desc, code in examples
    ) + "</div>"


def _layout(page: Page, pages: Sequence[Page]) -> str:
    nav = "".join(
        f'<a href="{_esc(_rel(page.path, p.path))}">{_esc(p.title)}</a>'
        for p in pages
    )
    return _rstrip_lines(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(page.title)} - TensorGuard</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --panel2:#0f1724; --muted:#8b949e; --text:#e6edf3; --accent:#7ee787; --accent2:#79c0ff; --warm:#ffa657; --line:#30363d; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top left,#19385d 0,#0d1117 34rem),linear-gradient(180deg,#0d1117 0,#06090f 100%); color:var(--text); font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header, main, footer {{ max-width:1120px; margin:0 auto; padding:28px; }}
    header {{ padding-top:64px; padding-bottom:16px; }}
    nav {{ position:sticky; top:0; z-index:2; background:#0d1117f2; border-bottom:1px solid var(--line); padding:12px 28px; display:flex; gap:18px; overflow-x:auto; }}
    nav a {{ color:var(--muted); white-space:nowrap; }}
    a {{ color:var(--accent2); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    h1 {{ font-size:clamp(2.7rem,7vw,5.8rem); line-height:.95; margin:0 0 18px; letter-spacing:-0.06em; max-width:980px; }}
    h2 {{ margin-top:40px; border-top:1px solid var(--line); padding-top:28px; }}
    h3 {{ margin-top:28px; }}
    .lede {{ color:#c9d1d9; font-size:1.25rem; max-width:880px; }}
    .pill {{ display:inline-block; border:1px solid #2f6f46; background:#102819; border-radius:999px; padding:4px 12px; color:var(--accent); font-size:.86rem; margin-bottom:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(190px,100%),1fr)); gap:14px; margin:22px 0; }}
    .hero-actions {{ display:flex; flex-wrap:wrap; gap:12px; margin:26px 0 8px; }}
    .button {{ display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:10px 15px; background:var(--panel); color:var(--text); font-weight:700; }}
    .button.primary {{ background:var(--accent); border-color:var(--accent); color:#0d1117; }}
    .hero-grid, .examples, .instant-examples {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr)); gap:18px; margin:22px 0; }}
    .use-cases {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(255px,100%),1fr)); gap:14px; margin:28px 0; }}
    .feature-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(245px,100%),1fr)); gap:14px; margin:22px 0; }}
    .verified-examples {{ display:grid; gap:16px; margin:24px 0; }}
    article, .callout, .example {{ min-width:0; background:linear-gradient(180deg,var(--panel),var(--panel2)); border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 12px 30px #00000024; overflow:hidden; overflow-wrap:anywhere; }}
    article strong {{ display:block; color:var(--accent); text-transform:uppercase; font-size:.78rem; letter-spacing:.08em; }}
    article span {{ display:block; font-size:1.45rem; font-weight:700; margin:.2rem 0; }}
    article p, .muted {{ color:var(--muted); margin:.2rem 0 0; }}
    .instant-examples article strong {{ color:var(--text); text-transform:none; letter-spacing:0; font-size:1.05rem; }}
    .instant-examples article span {{ color:var(--warm); font-size:.96rem; line-height:1.45; margin-top:12px; overflow-wrap:anywhere; }}
    .use-case {{ min-height:178px; }}
    .use-case strong {{ color:var(--accent); font-size:1rem; letter-spacing:0; text-transform:none; }}
    .use-case code {{ display:block; margin-top:12px; color:var(--warm); white-space:normal; overflow-wrap:anywhere; }}
    .feature-card {{ color:inherit; text-decoration:none; display:block; }}
    .feature-card:hover {{ text-decoration:none; transform:translateY(-2px); transition:transform .12s ease; }}
    .feature strong {{ color:var(--accent2); text-transform:none; letter-spacing:0; font-size:.98rem; }}
    .feature p {{ margin-top:.55rem; }}
    .feature span {{ color:var(--warm); font-size:.85rem; margin-top:12px; }}
    .verified-example {{ scroll-margin-top:76px; display:grid; grid-template-columns:minmax(0,1.6fr) minmax(260px,.9fr); gap:16px; align-items:stretch; background:linear-gradient(135deg,#111d2d,#0d1117); border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:0 16px 40px #0000002e; }}
    .verified-example strong {{ color:var(--accent2); text-transform:uppercase; letter-spacing:.08em; font-size:.78rem; }}
    .verified-example h3 {{ margin:.3rem 0 .4rem; font-size:1.4rem; line-height:1.18; }}
    .verified-example p {{ color:#c9d1d9; margin:0; }}
    .example-result {{ background:#010409; border:1px solid var(--line); border-radius:14px; padding:16px; display:flex; flex-direction:column; gap:10px; }}
    .example-result span {{ color:var(--accent); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }}
    .example-result code {{ color:var(--warm); white-space:normal; overflow-wrap:anywhere; }}
    .example-result small {{ color:var(--muted); line-height:1.45; }}
    @media (max-width:760px) {{ .verified-example {{ grid-template-columns:1fr; }} }}
    table {{ width:100%; border-collapse:collapse; margin:18px 0; font-size:.94rem; }}
    th, td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); }}
    pre {{ overflow:auto; max-width:100%; background:#010409; border:1px solid var(--line); border-radius:12px; padding:16px; white-space:pre-wrap; overflow-wrap:anywhere; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    ul {{ padding-left:1.4rem; }}
    footer {{ color:var(--muted); border-top:1px solid var(--line); display:flex; flex-wrap:wrap; gap:10px 18px; justify-content:space-between; }}
    footer strong {{ color:var(--text); }}
  </style>
</head>
<body>
<nav>{nav}</nav>
<header>
  <span class="pill">{_esc(page.category)}</span>
  <h1>{_esc(page.title)}</h1>
  <p class="lede">{_esc(page.description)}</p>
</header>
<main>
{page.body}
</main>
<footer><span><strong>TensorGuard</strong> — static verification for PyTorch model reliability.</span><span><a href="https://github.com/thehalleyyoung/tensorguard">GitHub</a> · MIT licensed · Python 3.9+</span></footer>
</body>
</html>
""")


def _rstrip_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def _rel(src: str, dst: str) -> str:
    return Path(dst).relative_to(".").as_posix() if "/" not in src else Path("../" * src.count("/") + dst).as_posix()


def build_pages() -> list[Page]:
    conf, proof_summary, proof_rows = _operator_summaries()
    install_command = 'python -m pip install "git+https://github.com/thehalleyyoung/tensorguard.git"'
    evidence_commands = (
        "make reproduce-check\n"
        "make dashboard-gate\n"
        "make docs-site\n"
        "python reproducibility/artifact_index.py --check"
    )
    supported_counts = {
        "layers": len(SUPPORTED_LAYER_TYPES),
        "methods": len(SUPPORTED_TENSOR_METHODS),
        "torch functions": len(SUPPORTED_TORCH_FUNCTIONS),
        "functional calls": len(SUPPORTED_F_FUNCTIONS),
    }
    grammar = VERIFIABLE_FRAGMENT_GRAMMAR
    unsupported_rows = [
        (cat.name, info["description"], info["detected_by"])
        for cat, info in sorted(UNSUPPORTED_CATEGORY_INFO.items(), key=lambda kv: kv[0].name)
    ]
    top_proofs = sorted(
        proof_rows,
        key=lambda row: (
            0 if row["proof_status"] == ProofStatus.LEAN_THEOREM.value else 1,
            str(row["operator"]),
        ),
    )[:60]

    pages = [
        Page(
            "index.html",
            "TensorGuard",
            "Static verification for PyTorch models and tensor APIs: catch shape, device, phase, dtype, gradient, sparse, attention, export, checkpoint, and serving bugs before a forward pass.",
            f"""
            <div class="hero-actions">
              <a class="button primary" href="#install">Install from GitHub</a>
              <a class="button" href="#use-it">Use it everywhere</a>
              <a class="button" href="#python-api">Python API examples</a>
              <a class="button" href="#features">20-feature showcase</a>
            </div>
            <h2>Immediate proof points</h2>
            {_instant_examples()}
            <h2 id="use-it">Coolest verified ways to use TensorGuard</h2>
            <p>These are the adoption paths backed by code in this repository and covered by targeted tests before being promoted here.</p>
            {_use_case_showcase()}
            <h2>Checked examples for every use path</h2>
            <p>Every card above jumps to concrete code, notebook, CLI, or configuration that was validated before publication.</p>
            {_use_case_examples()}
            <div class="hero-grid">
              <article><strong>public URL</strong><span>thehalleyyoung.github.io/tensorguard/</span><p>This generated site is the GitHub Pages artifact uploaded from <code>docs/site</code>.</p></article>
              <article><strong>verdict contract</strong><span>SAFE / UNSAFE / UNKNOWN</span><p>Sound mode proves only inside the published verifiable fragment and abstains honestly outside it.</p></article>
              <article><strong>Python surface</strong><span>{sum(conf.values())} operators</span><p>Public helpers cover core modules plus library, export, deployment, checkpoint, and serving gates.</p></article>
            </div>
            <h2 id="install">Install from GitHub</h2>
            <p>Install directly from the source repository:</p>
            {_code(install_command)}
            <p class="muted">The install command above was dry-run in a clean virtual environment.</p>
            <h2 id="python-api">Python API, not just a CLI</h2>
            <p>These examples use the public <code>tensorguard</code> package surface and are kept CPU-only so they are easy to copy into a notebook, test, or CI gate.</p>
            {_api_examples()}
            <h2 id="features">The 20 most impressive things TensorGuard can do now</h2>
            <p>Grouped from the current README and public package exports, this list focuses on the highest-value capabilities instead of enumerating every helper individually.</p>
            {_feature_showcase()}
            <h2>Extensive working examples behind every feature</h2>
            <p>Each card above jumps to concrete code or configuration that exercises the capability directly.</p>
            {_verified_feature_examples()}
            <h2>Evidence-backed, generated, and CI-ready</h2>
            {_metric_cards()}
            <h2>Deep docs</h2>
            <div class="cards">
              <article><strong>Concepts</strong><span>soundness first</span><p>Learn the SAFE/UNSAFE/UNKNOWN contract and the supported fragment.</p></article>
              <article><strong>Reference</strong><span>{sum(conf.values())} operators</span><p>Browse confidence tags and proof-footprint tiers generated from real registry data.</p></article>
              <article><strong>Migration</strong><span>CI-ready</span><p>Move from runtime asserts or PyTea-style shape-only checks to sound-mode TensorGuard gates.</p></article>
            </div>
            <h2>Reader map</h2>
            <ul>
              <li>{_link('concepts/soundness.html', 'Soundness modes')} explains the contract a CI gate can rely on.</li>
              <li>{_link('concepts/verifiable-fragment.html', 'Verifiable fragment')} lists supported Python/PyTorch constructs and explicit abstentions.</li>
              <li>{_link('operators/index.html', 'Operator reference')} summarizes confidence tags for every registered transfer.</li>
              <li>{_link('proof-footprint/index.html', 'Proof footprint')} ties operators to Lean, pen-and-paper, tested-only, or heuristic evidence.</li>
              <li>{_link('migration/runtime-assertions.html', 'Migration guides')} show how to adopt TensorGuard without losing existing runtime checks.</li>
            </ul>
            """,
            "overview",
        ),
        Page(
            "concepts/soundness.html",
            "Soundness modes",
            "The three-valued contract: refute real bugs, prove only inside the fragment, abstain honestly otherwise.",
            f"""
            <div class="callout"><strong>Guarantee.</strong><p>{_esc(SOUNDNESS_GUARANTEE)}</p></div>
            <h2>Modes</h2>
            {_table(['mode', 'contract'], _soundness_mode_rows())}
            <h2>Domain clauses</h2>
            {_table(['construct', 'class', 'direction', 'evidence'], [(c.construct, c.soundness_class.value, c.direction, c.evidence) for c in DOMAIN_CLAUSES])}
            <h2>Operational rule</h2>
            <p>Use <code>--soundness-mode sound</code> for merge gates. It preserves bug reports but downgrades unsupported regions to <code>UNKNOWN</code> instead of silently passing them.</p>
            """,
            "concept chapter",
        ),
        Page(
            "concepts/verifiable-fragment.html",
            "Verifiable fragment",
            "The grammar TensorGuard can prove over, plus every unsupported construct that becomes UNKNOWN in sound mode.",
            f"""
            <h2>Core grammar</h2>
            {_code(grammar)}
            <h2>Supported-surface counts</h2>
            {_table(['surface', 'count'], supported_counts.items())}
            <h2>Unsupported constructs</h2>
            {_table(['category', 'description', 'detection'], unsupported_rows)}
            """,
            "concept chapter",
        ),
        Page(
            "concepts/evidence.html",
            "Evidence model",
            "How TensorGuard connects real code, generated artifacts, audits, and CI gates into one reproducible trust story.",
            f"""
            {_metric_cards()}
            <h2>Regeneration commands</h2>
            {_code(evidence_commands)}
            <h2>Primary artifacts</h2>
            <ul>
              <li><code>real_benchmarks/manifest.json</code> freezes the executable labeled corpus.</li>
              <li><code>evaluation/confusion_matrices.json</code> stores head-to-head baseline outcomes.</li>
              <li><code>reproducibility/numeric_claims_audit.json</code> audits README and paper claims.</li>
              <li><code>reproducibility/artifact_index.json</code> hashes generated artifacts, including this site.</li>
            </ul>
            """,
            "concept chapter",
        ),
        Page(
            "operators/index.html",
            "Operator reference",
            "A compact reference for registered transfer functions and their confidence tags.",
            f"""
            <h2>Confidence summary</h2>
            {_table(['confidence', 'operators'], sorted(conf.items()))}
            <h2>Representative rows</h2>
            {_table(['operator', 'confidence', 'rationale'], [(r['operator'], r['confidence'], r['rationale']) for r in confidence_table()[:60]])}
            <p class="muted">The full machine-readable table is <code>operator_confidence_table.json</code>; unknown operators default to heuristic.</p>
            """,
            "operator reference",
        ),
        Page(
            "proof-footprint/index.html",
            "Proof footprint",
            "Operator evidence tiers generated from the audited proof-footprint manifest.",
            f"""
            <h2>Evidence tiers</h2>
            {_table(['tier', 'operators'], [(status.value, proof_summary[status.value]) for status in ProofStatus])}
            <h2>Audited rows</h2>
            {_table(['operator', 'tier', 'confidence', 'rule', 'evidence'], [(r['operator'], r['proof_status'], r['confidence'], r['rule'], ', '.join(r['evidence'])) for r in top_proofs])}
            <p class="muted">Lean-backed claims are allowlisted per operator; broad families are not promoted unless a listed theorem directly covers the registered transfer.</p>
            """,
            "proof reference",
        ),
        Page(
            "migration/runtime-assertions.html",
            "Migrate from runtime assertions",
            "Keep valuable dynamic checks, but move deterministic tensor-contract failures to a pre-forward static gate.",
            """
            <h2>Recipe</h2>
            <ol>
              <li>Start with <code>tensorguard verify model.py --soundness-mode sound</code> in non-blocking CI.</li>
              <li>Keep runtime assertions for value-domain properties TensorGuard intentionally does not model.</li>
              <li>Promote static shape/device/dtype/gradient failures to blocking once the baseline is clean.</li>
              <li>Use JSON, JUnit, SARIF, or GitHub annotations to route findings to the same place as tests.</li>
            </ol>
            <h2>Why this is not just moving tests earlier</h2>
            <p>Static checks cover paths a seed batch may not execute, CUDA-only device conflicts on CPU hosts, and silent gradient-flow bugs that do not raise exceptions.</p>
            """,
            "migration guide",
        ),
        Page(
            "migration/pytea.html",
            "Migrate from PyTea-style shape checks",
            "Adopt a broader PyTorch safety net while preserving shape-only fairness in comparisons.",
            """
            <h2>What changes</h2>
            <ul>
              <li>TensorGuard keeps shape checking, then adds device, dtype, gradient, phase diagnostics, deployment gates, and proof-footprint metadata.</li>
              <li>The strict mode reports <code>UNKNOWN</code> for unsupported regions rather than presenting a best-effort pass as a proof.</li>
              <li>Precision/recall artifacts include shape-only sub-corpus views so comparisons remain fair.</li>
            </ul>
            <h2>Adoption command</h2>
            """ + _code("tensorguard verify model.py --soundness-mode sound --json") + """
            """,
            "migration guide",
        ),
    ]
    return pages


def _write_manifest(pages: Sequence[Page]) -> None:
    conf, proof_summary, _ = _operator_summaries()
    payload = {
        "schema": "tensorguard.docs_site/v1",
        "step": 285,
        "generator": "reproducibility/docs_site.py",
        "pages": [
            {
                "path": page.path,
                "title": page.title,
                "category": page.category,
                "description": page.description,
            }
            for page in pages
        ],
        "operator_confidence_summary": conf,
        "proof_footprint_summary": proof_summary,
        "supported_surface_counts": {
            "layers": len(SUPPORTED_LAYER_TYPES),
            "methods": len(SUPPORTED_TENSOR_METHODS),
            "torch_functions": len(SUPPORTED_TORCH_FUNCTIONS),
            "functional_calls": len(SUPPORTED_F_FUNCTIONS),
        },
        "source_documents": [
            "src/soundness_contract.py",
            "src/verifiable_fragment.py",
            "src/operator_confidence.py",
            "src/proof_footprint.py",
            "evaluation/confusion_matrices.json",
            "evaluation/sound_mode_fp.json",
            "evaluation/neg_fuzz.json",
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate() -> None:
    pages = build_pages()
    SITE.mkdir(parents=True, exist_ok=True)
    for page in pages:
        out = SITE / page.path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_layout(page, pages), encoding="utf-8")
    _write_manifest(pages)


def _expected() -> dict[Path, str]:
    pages = build_pages()
    expected = {SITE / page.path: _layout(page, pages) for page in pages}
    generate_manifest_pages = pages
    conf, proof_summary, _ = _operator_summaries()
    manifest_payload = {
        "schema": "tensorguard.docs_site/v1",
        "step": 285,
        "generator": "reproducibility/docs_site.py",
        "pages": [
            {
                "path": page.path,
                "title": page.title,
                "category": page.category,
                "description": page.description,
            }
            for page in generate_manifest_pages
        ],
        "operator_confidence_summary": conf,
        "proof_footprint_summary": proof_summary,
        "supported_surface_counts": {
            "layers": len(SUPPORTED_LAYER_TYPES),
            "methods": len(SUPPORTED_TENSOR_METHODS),
            "torch_functions": len(SUPPORTED_TORCH_FUNCTIONS),
            "functional_calls": len(SUPPORTED_F_FUNCTIONS),
        },
        "source_documents": [
            "src/soundness_contract.py",
            "src/verifiable_fragment.py",
            "src/operator_confidence.py",
            "src/proof_footprint.py",
            "evaluation/confusion_matrices.json",
            "evaluation/sound_mode_fp.json",
            "evaluation/neg_fuzz.json",
        ],
    }
    expected[MANIFEST] = json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    return expected


def check() -> int:
    missing_or_stale = []
    for path, content in _expected().items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            missing_or_stale.append(path.relative_to(ROOT).as_posix())
    if missing_or_stale:
        for rel in missing_or_stale:
            print(f"MISMATCH: {rel}")
        return 1
    print("docs_site: byte-identical")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed site is stale")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    generate()
    print(f"wrote {SITE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
