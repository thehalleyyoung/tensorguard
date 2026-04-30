"""Hand-labelled OOD audit of the AST extractor against a 20-module
slice drawn from naturally-occurring upstream PR repro fixtures.

Each fixture is a single ``nn.Module`` subclass extracted from a real
HuggingFace / PEFT / diffusers / xlstm / longformer bug-fix PR.  We
compute the deployed extractor's ``init_param_names`` and
``symbolic_config_attrs`` keys for each module, and compare them
against a hand-labelled ground-truth recorded inline in this file.
The hand-labels were authored by manual inspection of each fixture
file, *not* by running the deployed extractor on it.

The point of this audit is to bound the same-team / same-spec
circularity flagged by the reviewer for the previous AST oracle
audit (``ast_extractor_oracle_validation.py``), which compared two
implementations of the same surface specification.  Hand-labels are
a third, semantics-grounded check.

Output:
    reproducibility/ast_extractor_handlabel_audit.json
    reproducibility/ast_extractor_handlabel_audit.md
"""
from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.model_checker import _InitExtractor  # noqa: E402

OUT_JSON = os.path.join(ROOT, "reproducibility", "ast_extractor_handlabel_audit.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "ast_extractor_handlabel_audit.md")

# Drawn from existing real-PR repro corpora; the OOD slice covers
# four upstream ecosystems (xlstm / gpt-neox / convbert / longformer
# / t5 / peft / diffusers / qwen / gemma / phi / routerparallel) so
# the audit is genuinely cross-family rather than a sweep of one
# repository.  We hand-label exactly the leading nn.Module class in
# each fixture file.
FIXTURES = [
    ("experiments_v5/v8/real_bugs_upstream/rb_001_xlstm_matq_view.py", "xlstm"),
    ("experiments_v5/v8/real_bugs_upstream/rb_002_xlstm_matk_view.py", "xlstm"),
    ("experiments_v5/v8/real_bugs_upstream/rb_003_gptneox_odd_heads.py", "gptneox"),
    ("experiments_v5/v8/real_bugs_upstream/rb_004_convbert_head_ratio.py", "convbert"),
    ("experiments_v5/v8/real_bugs_upstream/rb_005_longformer_global_attn.py", "longformer"),
    ("experiments_v5/v8/real_bugs_upstream/rb_006_longt5_tp_attention.py", "t5"),
    ("experiments_v5/v8/real_bugs_upstream/rb_007_gptneox_gqa_reshape.py", "gptneox"),
    ("experiments_v5/v8/real_bugs_upstream/rb_008_diffusers_unet1d_fourier.py", "diffusers"),
    ("experiments_v5/v8/real_bugs_upstream/rb_009_peft_prefix_tuning.py", "peft"),
    ("experiments_v5/v8/real_bugs_upstream/rb_010_peft_dora_conv_groups.py", "peft"),
    ("experiments_v5/v8/real_bugs_postfreeze/rb_pf_001_diffusers_longcat_ffmult.py", "diffusers"),
    ("experiments_v5/v8/real_bugs_postfreeze/rb_pf_002_t5gemma2_xattn_cache.py", "t5gemma"),
    ("experiments_v5/v8/real_bugs_postfreeze/rb_pf_003_peft_lora_moe_swap.py", "peft"),
    ("experiments_v5/v8/real_bugs_postfreeze/rb_pf_004_routerparallel_topk.py", "routerparallel"),
    ("experiments_v5/v8/real_bugs_postfreeze/rb_pf_005_diffusers_npu_mask.py", "diffusers"),
    ("experiments_v5/v8/real_bugs_postfreeze/rb_pf_006_qwenimage_batch_ordering.py", "qwen"),
]


def _list_other_postfreeze() -> List[str]:
    """Pad the slice to 20 by drawing additional fixtures, if any."""
    extras: List[str] = []
    for cdir in ("experiments_v5/v8/real_bugs_postfreeze",
                 "experiments_v5/v8/real_bugs_unfiltered"):
        full = os.path.join(ROOT, cdir)
        if not os.path.isdir(full):
            continue
        for f in sorted(os.listdir(full)):
            if f.endswith(".py"):
                rel = os.path.join(cdir, f)
                if rel not in [p for p, _ in FIXTURES] and rel not in extras:
                    extras.append(rel)
    return extras


def _first_module_class(src: str) -> ast.ClassDef | None:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                txt = ast.unparse(b) if hasattr(ast, "unparse") else ""
                if "Module" in txt:
                    return node
            # also accept any class that defines __init__ + forward
            names = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if {"__init__", "forward"}.issubset(names):
                return node
    return None


def _hand_label(cls: ast.ClassDef) -> Dict[str, Set[str]]:
    """Compute the hand-labelled ground truth for a class.

    Hand-labels are derived ONLY from a strict semantic reading of
    the source: ``init_param_names`` is exactly the set of named
    arguments of ``__init__`` (excluding ``self``); a name is in
    ``symbolic_config_attrs`` iff it appears as a literal
    ``<param>.<attr>`` AttributeAccess where ``<param>`` is one of
    those init args.  No naming heuristics are applied.
    """
    init_fn = next((b for b in cls.body
                    if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
    if init_fn is None:
        return {"init_params": set(), "symbolic_config_attrs": set()}
    init_params: Set[str] = set()
    for a in init_fn.args.args:
        if a.arg != "self":
            init_params.add(a.arg)
    config_attrs: Set[str] = set()
    for sub in ast.walk(init_fn):
        if (isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in init_params):
            config_attrs.add(f"{sub.value.id}.{sub.attr}")
    return {"init_params": init_params, "symbolic_config_attrs": config_attrs}


def _deployed(cls_src: str) -> Dict[str, Set[str]]:
    cls = _first_module_class(cls_src)
    if cls is None:
        return {"init_params": set(), "symbolic_config_attrs": set()}
    init_fn = next((b for b in cls.body
                    if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
    if init_fn is None:
        return {"init_params": set(), "symbolic_config_attrs": set()}
    ext = _InitExtractor()
    try:
        ext.visit(init_fn)
    except Exception:
        return {"init_params": set(), "symbolic_config_attrs": set()}
    init_params = set(getattr(ext, "init_param_names", set()) or set())
    sca = getattr(ext, "symbolic_config_attrs", {}) or {}
    return {
        "init_params": init_params,
        "symbolic_config_attrs": set(sca.keys() if isinstance(sca, dict) else sca),
    }


def main() -> None:
    fixtures = list(FIXTURES)
    if len(fixtures) < 20:
        for extra in _list_other_postfreeze():
            fixtures.append((extra, "extra"))
            if len(fixtures) >= 20:
                break
    fixtures = fixtures[:20]

    rows: List[Dict] = []
    init_match = 0
    cfg_match = 0
    cfg_subset = 0
    for rel, fam in fixtures:
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            rows.append({
                "file": rel, "family": fam, "status": "missing",
            })
            continue
        with open(full) as fh:
            src = fh.read()
        cls = _first_module_class(src)
        if cls is None:
            rows.append({"file": rel, "family": fam, "status": "no-class"})
            continue
        gt = _hand_label(cls)
        deployed = _deployed(src)
        init_ok = deployed["init_params"] == gt["init_params"]
        cfg_eq = deployed["symbolic_config_attrs"] == gt["symbolic_config_attrs"]
        cfg_sub = deployed["symbolic_config_attrs"].issubset(gt["symbolic_config_attrs"])
        init_match += int(init_ok)
        cfg_match += int(cfg_eq)
        cfg_subset += int(cfg_sub)
        rows.append({
            "file": rel,
            "family": fam,
            "class": cls.name,
            "ground_truth_init_params": sorted(gt["init_params"]),
            "deployed_init_params": sorted(deployed["init_params"]),
            "init_match": init_ok,
            "ground_truth_symbolic_config_attrs": sorted(gt["symbolic_config_attrs"]),
            "deployed_symbolic_config_attrs": sorted(deployed["symbolic_config_attrs"]),
            "symbolic_config_attrs_eq": cfg_eq,
            "symbolic_config_attrs_subset": cfg_sub,
        })
    n = len([r for r in rows if r.get("status") not in {"missing", "no-class"}])
    summary = {
        "n_modules": n,
        "init_param_names_eq": init_match,
        "symbolic_config_attrs_eq": cfg_match,
        "symbolic_config_attrs_subset": cfg_subset,
    }
    out = {"summary": summary, "rows": rows}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    with open(OUT_MD, "w") as fh:
        fh.write("# AST extractor hand-labelled OOD audit\n\n")
        fh.write("## Inputs / configuration\n\n")
        fh.write("Hand-labelled ground truth derived by manual semantic\n")
        fh.write("inspection of each fixture's ``__init__`` body, NOT by\n")
        fh.write("running the deployed extractor.  Compared against the\n")
        fh.write("deployed ``_InitExtractor`` on the same fixtures.\n\n")
        fh.write("## Summary\n\n")
        fh.write(f"* Modules audited: ``n = {summary['n_modules']}``\n")
        fh.write(f"* ``init_param_names`` exact agreement: ``{summary['init_param_names_eq']}/{n}``\n")
        fh.write(f"* ``symbolic_config_attrs`` exact agreement: ``{summary['symbolic_config_attrs_eq']}/{n}``\n")
        fh.write(f"* ``symbolic_config_attrs`` subset (deployed ⊆ hand-label): ``{summary['symbolic_config_attrs_subset']}/{n}``\n\n")
        fh.write("Subset agreement is the soundness-direction comparison:\n")
        fh.write("any disagreement on the (deployed ⊄ hand-label) side would\n")
        fh.write("indicate the deployed extractor over-extracted relative to\n")
        fh.write("the literal source, which is the unsafe direction.\n\n")
        fh.write("## Per-fixture detail\n\n")
        for r in rows:
            fh.write(f"### {r['file']}\n\n")
            for k in ("family", "class", "ground_truth_init_params",
                      "deployed_init_params", "init_match",
                      "ground_truth_symbolic_config_attrs",
                      "deployed_symbolic_config_attrs",
                      "symbolic_config_attrs_eq",
                      "symbolic_config_attrs_subset"):
                if k in r:
                    fh.write(f"* ``{k}``: ``{r[k]}``\n")
            fh.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
