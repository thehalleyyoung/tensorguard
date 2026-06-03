"""Interactive results dashboard, regenerated from deterministic artifacts (Step 115).

A reviewer should be able to *see* the evidence without running anything. This
generator reads a curated set of the committed, byte-deterministic reproducibility
artifacts and emits a single self-contained static site --
``docs/dashboard/index.html`` plus its ``docs/dashboard/data.json`` data bundle --
with no external dependencies, no CDN and no server: open the HTML file and the
headline numbers, per-artifact metrics and raw values are all there, with
client-side category tabs and a live text filter.

Because the site is built purely from the committed artifacts, it is itself
byte-deterministic and is verified by ``--check`` (and by ``reproduce_all.py``).
Each curated artifact contributes a card with a one-line headline and a small set
of extracted metrics; the underlying JSON stays the single source of truth.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPRO = REPO / "reproducibility"
OUT_DIR = REPO / "docs" / "dashboard"
OUT_HTML = OUT_DIR / "index.html"
OUT_JSON = OUT_DIR / "data.json"


def _load(name: str) -> dict:
    return json.loads((REPRO / name).read_text())


def _pct(x) -> str:
    return f"{round(100.0 * float(x), 2)} percent" if x is not None else "n/a"


# ---------------------------------------------------------------------------
# Per-artifact extractors: each returns (headline, [ {label, value}, ... ]).
# Values are plain strings/ints/bools so the bundle is stable across machines.
# ---------------------------------------------------------------------------


def _x_differential():
    d = _load("differential_dispatcher.json")
    headline = (
        f"{d['n_modules']} random modules vs the live torch dispatcher: "
        f"{d['n_soundness_violations']} soundness violations, "
        f"{d['n_false_alarms']} false alarms"
    )
    metrics = [
        {"label": "random modules", "value": str(d["n_modules"])},
        {"label": "families", "value": str(len(d["families"]))},
        {"label": "decided verdicts", "value": str(d["n_decided"])},
        {"label": "soundness violations", "value": str(d["n_soundness_violations"])},
        {"label": "false alarms", "value": str(d["n_false_alarms"])},
        {
            "label": "perfect decided agreement",
            "value": str(d["decided_agreement_perfect"]),
        },
    ]
    return headline, metrics


def _x_hypothesis():
    d = _load("hypothesis_module_ast.json")
    s = d["soundness"]
    sh = d["shrinking_demo"]
    headline = (
        f"{d['n_generated']} structured module ASTs: "
        f"{s['n_soundness_violations']} soundness violations, "
        f"{s['n_false_alarms']} false alarms; shrinks "
        f"{sh['start_n_layers']} layers to {sh['minimal_n_layers']}"
    )
    metrics = [
        {"label": "structured ASTs", "value": str(d["n_generated"])},
        {"label": "decided", "value": str(s["n_decided"])},
        {"label": "soundness violations", "value": str(s["n_soundness_violations"])},
        {"label": "false alarms", "value": str(s["n_false_alarms"])},
        {
            "label": "shrink layers (start to minimal)",
            "value": f"{sh['start_n_layers']} to {sh['minimal_n_layers']}",
        },
        {
            "label": "real verifier catches minimal",
            "value": str(sh["real_verifier_catches_minimal"]),
        },
    ]
    return headline, metrics


def _x_mutation():
    d = _load("mutation_clean_models.json")
    headline = (
        f"{d['n_genuine_bug_mutants']} genuine-bug mutants from "
        f"{d['n_clean_parents']} clean parents: sound-mode kill rate "
        f"{_pct(d['sound_mode_kill_rate_point'])}"
    )
    metrics = [
        {"label": "clean parents", "value": str(d["n_clean_parents"])},
        {"label": "genuine-bug mutants", "value": str(d["n_genuine_bug_mutants"])},
        {"label": "mutation operators", "value": str(d["n_operators"])},
        {
            "label": "sound-mode kill rate",
            "value": _pct(d["sound_mode_kill_rate_point"]),
        },
        {
            "label": "zero surviving false-safe",
            "value": str(d["sound_mode_zero_false_safe"]),
        },
    ]
    return headline, metrics


def _x_fp_stress():
    d = _load("fp_stress_eval.json")
    headline = (
        f"{d['n_models']} clean models across {d['n_families']} families: "
        f"zero sound-mode false alarms "
        f"({d['zero_false_alarms_sound_mode']})"
    )
    metrics = [
        {"label": "clean models", "value": str(d["n_models"])},
        {"label": "families", "value": str(d["n_families"])},
        {
            "label": "zero false alarms (sound)",
            "value": str(d["zero_false_alarms_sound_mode"]),
        },
        {
            "label": "zero false alarms (all modes)",
            "value": str(d["zero_false_alarms_all_modes"]),
        },
    ]
    return headline, metrics


def _x_blind():
    d = _load("blind_split_eval.json")
    headline = (
        "held-out blind split with pre-registered hypotheses: "
        f"all modes confirm pre-registration ({d['all_modes_confirm_preregistration']})"
    )
    metrics = [
        {
            "label": "manifest matches registration",
            "value": str(d["manifest_matches_registration"]),
        },
        {
            "label": "all modes confirm pre-registration",
            "value": str(d["all_modes_confirm_preregistration"]),
        },
        {
            "label": "registered manifest sha256",
            "value": str(d["registered_manifest_sha256"])[:16] + "...",
        },
    ]
    return headline, metrics


def _x_protocol():
    d = _load("evaluation_protocol.json")
    ro = d["reproduction_order"]
    headline = (
        "pre-specified protocol freezes splits, tuning rules, metric formulas, "
        f"and {len(d['analysis_scripts'])} analysis-script hashes"
    )
    metrics = [
        {"label": "protocol version", "value": str(d["protocol_version"])},
        {
            "label": "registered splits",
            "value": str(len(d["splits"])),
        },
        {
            "label": "dev/blind disjoint",
            "value": str(d["split_disjointness"]["development_vs_blind_disjoint"]),
        },
        {
            "label": "blind hash matches registration",
            "value": str(d["blind_preregistration"]["hash_matches_document"]),
        },
        {
            "label": "all scripts present",
            "value": str(d["all_analysis_scripts_present"]),
        },
        {
            "label": "protocol precedes governed scoring",
            "value": str(ro["protocol_precedes_governed_scoring"]),
        },
    ]
    return headline, metrics


def _x_time_to_detect():
    d = _load("time_to_detect.json")
    s = d["static"]
    dy = d["dynamic"]
    headline = (
        f"{d['n_buggy_modules']} bugs: static catches all at depth 0 "
        f"(input-free); dynamic needs a median of {dy['detect_depth_median']} "
        f"ops, up to {dy['detect_depth_max']}"
    )
    metrics = [
        {"label": "buggy modules", "value": str(d["n_buggy_modules"])},
        {"label": "static detect depth", "value": str(s["detect_depth"])},
        {
            "label": "static caught (UNSAFE)",
            "value": str(s["n_caught_unsafe"]),
        },
        {
            "label": "dynamic median detect depth",
            "value": str(dy["detect_depth_median"]),
        },
        {"label": "dynamic max detect depth", "value": str(dy["detect_depth_max"])},
        {
            "label": "bugs needing a successful prefix",
            "value": str(dy["n_requires_successful_prefix"]),
        },
    ]
    return headline, metrics


def _x_domain_ablation():
    d = _load("domain_ablation.json")
    headline = (
        f"{d['n_cases']} labeled bugs: every verification domain is necessary "
        "(leave-one-out recall drops to zero on its own bugs), domains are "
        "orthogonal, phase is diagnostic-only"
    )
    metrics = [
        {"label": "labeled bugs", "value": str(d["n_cases"])},
        {
            "label": "all verification domains at full recall",
            "value": str(d["all_verification_domains_full_recall"]),
        },
        {
            "label": "every domain necessary (LODO)",
            "value": str(d["every_domain_necessary"]),
        },
        {"label": "domains orthogonal", "value": str(d["domains_orthogonal"])},
        {
            "label": "phase diagnostic-only",
            "value": str(d["phase_diagnostic"]["is_diagnostic_only"]),
        },
        {
            "label": "toggle/report cross-check agrees",
            "value": str(d["toggle_report_crosscheck"]["agree"]),
        },
    ]
    return headline, metrics


def _x_cross_version():
    d = _load("cross_version_stability.json")
    headline = (
        "verdict stability across torch 2.1 to 2.9: "
        f"stable ({d['verdict_stable_across_torch_2_1_to_2_9']}), "
        "static (no torch execution)"
    )
    metrics = [
        {"label": "cases", "value": str(d["n_cases"])},
        {
            "label": "stable across torch 2.1 to 2.9",
            "value": str(d["verdict_stable_across_torch_2_1_to_2_9"]),
        },
        {
            "label": "static, no torch execution",
            "value": str(d["verifier_is_static_no_torch_execution"]),
        },
        {
            "label": "baseline verdict sha256",
            "value": str(d["baseline_verdict_sha256"])[:16] + "...",
        },
    ]
    return headline, metrics


# (id, title, category, extractor, source_artifact)
REGISTRY = [
    (
        "differential",
        "Differential testing vs the live torch dispatcher",
        "Soundness at scale",
        _x_differential,
        "differential_dispatcher.json",
    ),
    (
        "hypothesis",
        "Property-based full-module-AST testing with shrinking",
        "Soundness at scale",
        _x_hypothesis,
        "hypothesis_module_ast.json",
    ),
    (
        "mutation",
        "Mutation testing of clean models",
        "Soundness at scale",
        _x_mutation,
        "mutation_clean_models.json",
    ),
    (
        "fp_stress",
        "Clean-model false-positive stress test",
        "Precision",
        _x_fp_stress,
        "fp_stress_eval.json",
    ),
    (
        "time_to_detect",
        "Time-to-detect: static vs first failing forward",
        "Methodology",
        _x_time_to_detect,
        "time_to_detect.json",
    ),
    (
        "blind",
        "Held-out blind split + pre-registration",
        "Methodology",
        _x_blind,
        "blind_split_eval.json",
    ),
    (
        "evaluation_protocol",
        "Pre-specified evaluation protocol",
        "Methodology",
        _x_protocol,
        "evaluation_protocol.json",
    ),
    (
        "domain_ablation",
        "Per-domain ablation (leave-one-domain-out recall)",
        "Ablations",
        _x_domain_ablation,
        "domain_ablation.json",
    ),
    (
        "cross_version",
        "Cross-version verdict stability (torch 2.1 to 2.9)",
        "Robustness",
        _x_cross_version,
        "cross_version_stability.json",
    ),
]


def measure() -> dict:
    cards = []
    for cid, title, category, extractor, source in REGISTRY:
        headline, metrics = extractor()
        cards.append(
            {
                "id": cid,
                "title": title,
                "category": category,
                "headline": headline,
                "metrics": metrics,
                "source_artifact": source,
            }
        )
    categories = sorted({c["category"] for c in cards})
    return {
        "schema": 1,
        "title": "TensorGuard — evidence dashboard",
        "subtitle": (
            "Regenerated from committed, byte-deterministic reproducibility "
            "artifacts. No server, no network."
        ),
        "categories": categories,
        "cards": cards,
        "n_cards": len(cards),
    }


# ---------------------------------------------------------------------------
# Static HTML renderer (self-contained: inline CSS + vanilla JS).
# ---------------------------------------------------------------------------


def render_html(data: dict) -> str:
    bundle = json.dumps(data, indent=2, sort_keys=True)
    bundle_for_script = bundle.replace("</", "<\\/")
    title = html.escape(data["title"])
    subtitle = html.escape(data["subtitle"])
    # The page logic is intentionally tiny and dependency-free.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --fg:#e6edf3; --mut:#8b949e; --card:#161b22;
           --accent:#2f81f7; --ok:#3fb950; --line:#30363d; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
          font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:28px 24px 12px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0; font-size:24px; }}
  .sub {{ color:var(--mut); margin-top:6px; max-width:760px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center;
               padding:16px 24px; }}
  .tab {{ background:var(--card); color:var(--fg); border:1px solid var(--line);
          border-radius:999px; padding:6px 14px; cursor:pointer; font-size:13px; }}
  .tab.active {{ border-color:var(--accent); color:var(--accent); }}
  #q {{ margin-left:auto; background:var(--card); color:var(--fg);
        border:1px solid var(--line); border-radius:8px; padding:8px 12px;
        min-width:220px; }}
  main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
          gap:16px; padding:8px 24px 40px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
           padding:16px; }}
  .card h3 {{ margin:0 0 6px; font-size:16px; }}
  .cat {{ display:inline-block; font-size:11px; color:var(--mut);
          border:1px solid var(--line); border-radius:6px; padding:1px 7px; }}
  .headline {{ margin:10px 0 12px; color:var(--ok); font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  td {{ padding:4px 0; border-top:1px solid var(--line); vertical-align:top; }}
  td.k {{ color:var(--mut); padding-right:10px; }}
  td.v {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .src {{ margin-top:10px; font-size:11px; color:var(--mut); }}
  footer {{ color:var(--mut); padding:0 24px 32px; font-size:12px; }}
  code {{ background:#0b0f14; padding:1px 5px; border-radius:5px; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</header>
<div class="controls" id="tabs"></div>
<main id="grid"></main>
<footer>
  Every card is derived from a committed artifact under
  <code>reproducibility/</code>; rebuild with
  <code>python reproducibility/build_dashboard.py</code>.
</footer>
<script id="data" type="application/json">{bundle_for_script}</script>
<script>
  const DATA = JSON.parse(document.getElementById('data').textContent);
  let activeCat = 'All';
  let query = '';
  const cats = ['All'].concat(DATA.categories);
  const tabsEl = document.getElementById('tabs');
  cats.forEach(c => {{
    const b = document.createElement('button');
    b.className = 'tab' + (c === activeCat ? ' active' : '');
    b.textContent = c;
    b.onclick = () => {{ activeCat = c; renderTabs(); render(); }};
    b.dataset.cat = c;
    tabsEl.appendChild(b);
  }});
  const q = document.createElement('input');
  q.id = 'q'; q.placeholder = 'filter...';
  q.oninput = () => {{ query = q.value.toLowerCase(); render(); }};
  tabsEl.appendChild(q);
  function renderTabs() {{
    [...tabsEl.querySelectorAll('.tab')].forEach(b =>
      b.classList.toggle('active', b.dataset.cat === activeCat));
  }}
  function render() {{
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    DATA.cards.filter(c =>
      (activeCat === 'All' || c.category === activeCat) &&
      (query === '' || JSON.stringify(c).toLowerCase().includes(query))
    ).forEach(c => {{
      const rows = c.metrics.map(m =>
        '<tr><td class="k">' + esc(m.label) + '</td><td class="v">' +
        esc(m.value) + '</td></tr>').join('');
      const el = document.createElement('div');
      el.className = 'card';
      el.innerHTML =
        '<span class="cat">' + esc(c.category) + '</span>' +
        '<h3>' + esc(c.title) + '</h3>' +
        '<div class="headline">' + esc(c.headline) + '</div>' +
        '<table>' + rows + '</table>' +
        '<div class="src">source: <code>reproducibility/' +
        esc(c.source_artifact) + '</code></div>';
      grid.appendChild(el);
    }});
  }}
  function esc(s) {{
    return String(s).replace(/[&<>"]/g, ch =>
      ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}})[ch]);
  }}
  render();
</script>
</body>
</html>
"""


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    htmltext = render_html(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_HTML.exists() or OUT_HTML.read_text() != htmltext:
            print(f"MISMATCH: {OUT_HTML}")
            ok = False
        if ok:
            print("dashboard: byte-identical")
        return 0 if ok else 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(js)
    OUT_HTML.write_text(htmltext)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
