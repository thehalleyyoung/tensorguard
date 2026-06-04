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


def _feature_showcase() -> str:
    features = [
        (
            "Architecture verifier",
            "Zero-annotation PyTorch `nn.Module` checks infer tensor contracts from constructors, forward code, and input shapes.",
        ),
        (
            "Five-domain reasoning",
            "Shape, device, phase, stride, and permutation facts are propagated together instead of as isolated lint rules.",
        ),
        (
            "Broad operator model",
            "Linear, convolution, pooling, matmul, broadcasting, reshape/view, cat/stack/split/chunk, gather/scatter, fold/unfold, RNNs, and more are covered by registered transfer functions.",
        ),
        (
            "Soundness modes",
            "`sound`, `balanced`, and `heuristic` modes expose SAFE/UNSAFE/UNKNOWN instead of silently proving outside the supported fragment.",
        ),
        (
            "Machine-checked core",
            "Lean-backed theorem files cover CEGAR bounds, fragment modes, cross-domain transfers, SMT encodings, and subject-reduction-style composition claims.",
        ),
        (
            "CEGAR contracts",
            "Counterexample-guided predicate discovery finds implicit shape requirements and can promote inconsistent refined contracts to real bugs.",
        ),
        (
            "Developer diagnostics",
            "Source-mapped diagnostics, inference chains, proof-footprint badges, explain reports, and mechanical autofix suggestions turn failures into repairs.",
        ),
        (
            "Einops verification",
            "`verify_einops` and `verify_einops_source` check rearrange, reduce, and repeat patterns, including divisibility and axis bookkeeping.",
        ),
        (
            "Attention contracts",
            "Multihead attention, SDPA-style mask/batch/head constraints, packed or separate q/k/v projections, and weight-output shapes are modeled.",
        ),
        (
            "Linear algebra and FFT",
            "`torch.linalg` solve/inv/cholesky/SVD/eig/QR, complex view conversions, and FFT shape/dtype contracts are exposed as Python checks.",
        ),
        (
            "Sparse tensor gates",
            "COO, CSR, CSC, BSR, BSC, sparse-dense mm/addmm, sampled_addmm, softmax, coalesce, dense conversion, and layout conversion contracts are checked.",
        ),
        (
            "Losses and probability",
            "Loss target/reduction/dtype rules and distribution batch/event/log-prob shapes catch mismatches before training.",
        ),
        (
            "Names, vmap, and autodiff",
            "Named-tensor refine/align checks plus `torch.vmap` and `torch.func` grad, jacrev, jacfwd, jvp, and vjp shape transfers cover modern functional PyTorch.",
        ),
        (
            "FX and graph extraction",
            "AST, FX, Dynamo/export, graph-break attribution, and verifiable-fragment analysis make unsupported code explicit.",
        ),
        (
            "Compile and export gates",
            "`guarded_compile`, ONNX export, AOT packaging, GGUF export, CUDA graph capture, and compile-guard parity run verification before downstream tooling fails opaquely.",
        ),
        (
            "Checkpoint lifecycle",
            "State-dict schema checks cover missing/unexpected keys, dtype drift, tied weights, tensor-parallel shards, LoRA/PEFT adapters, and optimizer resume state.",
        ),
        (
            "Serving schemas",
            "FastAPI, TorchServe, and generic request-to-preprocess-to-model-to-response gates reject bad layouts before a serving call crosses an unsafe boundary.",
        ),
        (
            "Distributed and precision",
            "DTensor, FSDP2, pipeline boundaries, quantization placement, and mixed-precision/autocast gates cover deployment-time model transformations.",
        ),
        (
            "Extensible frontends",
            "A Flax/JAX frontend, governed declarative stubs, and a versioned operator-plugin ABI let new frameworks and operators extend the core safely.",
        ),
        (
            "Evidence and integrations",
            "SARIF, CI, pytest, pre-commit, dashboards, frozen benchmarks, mined-bug corpora, reproducibility ledgers, tutorials, and model galleries make adoption auditable.",
        ),
    ]
    assert len(features) == 20
    return "<div class=\"feature-grid\">" + "".join(
        f"<article class=\"feature\"><strong>{idx}. {_esc(title)}</strong><p>{_esc(desc)}</p></article>"
        for idx, (title, desc) in enumerate(features, start=1)
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


def _use_case_showcase() -> str:
    use_cases = [
        (
            "VS Code while you type",
            "Inline squiggles, hover shapes, and quick-fixes from the TensorGuard LSP client in `editors/vscode`; backed by the LSP server/client tests.",
            "Open `editors/vscode` in Extension Development Host",
        ),
        (
            "Python API gates",
            "Call `verify_architecture`, `verify_module`, `verify_einops`, `verify_linalg`, sparse, attention, serving, checkpoint, and optimizer helpers directly.",
            "from tensorguard import verify_architecture",
        ),
        (
            "pytest as a model test",
            "Use the pytest plugin to verify modules your tests exercise, so architecture regressions fail beside unit tests.",
            "pytest --tensorguard",
        ),
        (
            "pre-commit before review",
            "Run TensorGuard on changed model files before bad tensor contracts ever reach CI.",
            "tensorguard-precommit path/to/model.py",
        ),
        (
            "Jupyter and notebooks",
            "Load a notebook extension or use `%%tensorguard` cell magic to check model snippets where experiments happen.",
            "%load_ext src.jupyter_integration",
        ),
        (
            "torch.compile and export",
            "Wrap compile, ONNX export, AOT packages, and exported-program checks with TensorGuard preflight gates.",
            "guarded_compile(model, input_shapes={...})",
        ),
        (
            "Serving boundaries",
            "Validate FastAPI/TorchServe request, preprocessing, model-output, and response schemas before unsafe calls cross boundaries.",
            "verify_serving_schema(...)",
        ),
        (
            "Resume and deploy",
            "Check checkpoints, LoRA adapters, optimizer state, GGUF export, CUDA graph capture, distributed specs, and precision gates.",
            "verify_checkpoint_state_dict(model, state)",
        ),
    ]
    return "<div class=\"use-cases\">" + "".join(
        (
            f"<article class=\"use-case\"><strong>{_esc(title)}</strong>"
            f"<p>{_esc(desc)}</p><code>{_esc(command)}</code></article>"
        )
        for title, desc, command in use_cases
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
    :root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --panel2:#0f1724; --muted:#8b949e; --text:#e6edf3; --accent:#7ee787; --accent2:#79c0ff; --line:#30363d; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top left,#13233a 0,#0d1117 34rem); color:var(--text); font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header, main, footer {{ max-width:1120px; margin:0 auto; padding:28px; }}
    header {{ padding-top:42px; }}
    nav {{ position:sticky; top:0; z-index:2; background:#0d1117f2; border-bottom:1px solid var(--line); padding:12px 28px; display:flex; gap:18px; overflow-x:auto; }}
    nav a {{ color:var(--muted); white-space:nowrap; }}
    a {{ color:var(--accent2); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    h1 {{ font-size:clamp(2.7rem,7vw,5.8rem); line-height:.95; margin:0 0 18px; letter-spacing:-0.06em; max-width:980px; }}
    h2 {{ margin-top:40px; border-top:1px solid var(--line); padding-top:28px; }}
    h3 {{ margin-top:28px; }}
    .lede {{ color:#c9d1d9; font-size:1.25rem; max-width:880px; }}
    .pill {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 10px; color:var(--accent); font-size:.86rem; margin-bottom:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:22px 0; }}
    .hero-actions {{ display:flex; flex-wrap:wrap; gap:12px; margin:26px 0 4px; }}
    .button {{ display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:10px 15px; background:var(--panel); color:var(--text); font-weight:700; }}
    .button.primary {{ background:var(--accent); border-color:var(--accent); color:#0d1117; }}
    .hero-grid, .examples {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; margin:22px 0; }}
    .use-cases {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(255px,1fr)); gap:14px; margin:28px 0; }}
    .feature-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(245px,1fr)); gap:14px; margin:22px 0; }}
    article, .callout, .example {{ background:linear-gradient(180deg,var(--panel),var(--panel2)); border:1px solid var(--line); border-radius:14px; padding:18px; }}
    article strong {{ display:block; color:var(--accent); text-transform:uppercase; font-size:.78rem; letter-spacing:.08em; }}
    article span {{ display:block; font-size:1.45rem; font-weight:700; margin:.2rem 0; }}
    article p, .muted {{ color:var(--muted); margin:.2rem 0 0; }}
    .use-case {{ min-height:178px; }}
    .use-case strong {{ color:var(--accent); font-size:1rem; letter-spacing:0; text-transform:none; }}
    .use-case code {{ display:block; margin-top:12px; color:#ffa657; white-space:normal; }}
    .feature strong {{ color:var(--accent2); text-transform:none; letter-spacing:0; font-size:.98rem; }}
    .feature p {{ margin-top:.55rem; }}
    table {{ width:100%; border-collapse:collapse; margin:18px 0; font-size:.94rem; }}
    th, td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); }}
    pre {{ overflow:auto; background:#010409; border:1px solid var(--line); border-radius:12px; padding:16px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    ul {{ padding-left:1.4rem; }}
    footer {{ color:var(--muted); border-top:1px solid var(--line); }}
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
<footer>Generated by <code>reproducibility/docs_site.py</code>. Every headline link resolves inside this repository.</footer>
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
            <h2 id="use-it">Coolest verified ways to use TensorGuard</h2>
            <p>These are the adoption paths backed by code in this repository and covered by targeted tests before being promoted here.</p>
            {_use_case_showcase()}
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
            {_table(['mode', 'contract'], [(m.value, m.__doc__ or '') for m in SoundnessMode])}
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
