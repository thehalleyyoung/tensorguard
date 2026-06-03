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
    :root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --muted:#8b949e; --text:#e6edf3; --accent:#7ee787; --line:#30363d; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header, main, footer {{ max-width:1120px; margin:0 auto; padding:28px; }}
    header {{ padding-top:42px; }}
    nav {{ position:sticky; top:0; z-index:2; background:#0d1117f2; border-bottom:1px solid var(--line); padding:12px 28px; display:flex; gap:18px; overflow-x:auto; }}
    nav a {{ color:var(--muted); white-space:nowrap; }}
    a {{ color:#79c0ff; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    h1 {{ font-size:3rem; line-height:1.05; margin:0 0 12px; letter-spacing:-0.04em; }}
    h2 {{ margin-top:40px; border-top:1px solid var(--line); padding-top:28px; }}
    h3 {{ margin-top:28px; }}
    .lede {{ color:var(--muted); font-size:1.16rem; max-width:850px; }}
    .pill {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 10px; color:var(--accent); font-size:.86rem; margin-bottom:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:22px 0; }}
    article, .callout {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
    article strong {{ display:block; color:var(--accent); text-transform:uppercase; font-size:.78rem; letter-spacing:.08em; }}
    article span {{ display:block; font-size:1.45rem; font-weight:700; margin:.2rem 0; }}
    article p, .muted {{ color:var(--muted); margin:.2rem 0 0; }}
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
            "TensorGuard documentation",
            "A generated, repository-backed guide to the verifier, its evidence, and its adoption path.",
            f"""
            {_metric_cards()}
            <h2>Start here</h2>
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
            {_code('make reproduce-check\\nmake dashboard-gate\\nmake docs-site\\npython reproducibility/artifact_index.py --check')}
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
