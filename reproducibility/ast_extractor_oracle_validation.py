"""Round-3 W5: AST extractor cross-validation against an independent oracle.

The reviewer notes that the soundness of all 128 CV verdicts depends on
correct synthesis of assume_M from class source by the AST extractor
(_InitExtractor in src/model_checker.py), which is in the TCB
(not Lean-audited). This script cross-validates that extractor against
an *independent* re-implementation built from the standard-library
``ast`` module: a simple, deliberately minimal oracle that uses no
state shared with the deployed extractor.

Oracle definition (per __init__):
    * INIT_PARAMS_oracle  = {arg.arg for arg in args.args if arg.arg != 'self'}
    * SCALAR_ATTRS_oracle = {tgt.attr  | "self.<attr> = <Constant int|float>"}
    * CONFIG_REFS_oracle  = {(arg, attr) | <arg>.<attr> appears in __init__
                              AND _is_config_param_name(arg)}
        (using the same _is_config_param_name predicate, since the
         "config-likeness" naming convention is part of the protocol;
         we are testing the structural extraction, not the naming
         heuristic itself.)

Comparison metric per fixture:
    * config_attrs_agreement: oracle CONFIG_REFS == extractor.symbolic_config_attrs.keys()
        (modulo reassigned-attr exclusions; the oracle is a strict
         superset because the extractor soundly drops reassignments;
         we report the symmetric and the conservative-only deltas.)
    * scalar_attrs_agreement: oracle SCALAR_ATTRS ⊇ extractor.scalar_attrs.keys()
    * init_params_agreement: oracle INIT_PARAMS ⊇ extractor.init_param_names

Corpora:
    1. The 113 config-attribute fixtures
       (reproducibility/config_attribute_113_fixtures/).
    2. The 60-bug historical corpus (experiments_v5/v8/bug_corpus/).
    3. The 10-bug upstream-faithful real corpus
       (experiments_v5/v8/real_bugs_upstream/).

Output:
    reproducibility/ast_extractor_oracle_validation.json
    reproducibility/ast_extractor_oracle_validation.md
"""
from __future__ import annotations

import ast
import json
import os
import sys
import traceback
from typing import Dict, List, Set, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.model_checker import _InitExtractor, _is_config_param_name  # noqa: E402

OUT_JSON = os.path.join(ROOT, "reproducibility", "ast_extractor_oracle_validation.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "ast_extractor_oracle_validation.md")

CORPORA = [
    ("113-fixture-corpus", os.path.join(ROOT, "reproducibility", "config_attribute_113_fixtures")),
    ("10-real-public-bugs", os.path.join(ROOT, "experiments_v5", "v8", "real_bugs_upstream")),
    ("15-real-public-postfreeze", os.path.join(ROOT, "experiments_v5", "v8", "real_bugs_postfreeze")),
    ("15-real-public-unfiltered", os.path.join(ROOT, "experiments_v5", "v8", "real_bugs_unfiltered")),
]


def oracle_extract(init_fn: ast.FunctionDef) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    """Independent re-implementation of init-time AST extraction."""
    init_params: Set[str] = set()
    for a in init_fn.args.args:
        if a.arg != "self":
            init_params.add(a.arg)
    scalar_attrs: Set[str] = set()
    config_refs: Set[Tuple[str, str]] = set()
    config_param_names = {p for p in init_params if _is_config_param_name(p)}
    for sub in ast.walk(init_fn):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and isinstance(sub.value, ast.Constant)
                        and isinstance(sub.value.value, (int, float))
                        and not isinstance(sub.value.value, bool)):
                    scalar_attrs.add(tgt.attr)
        if (isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in config_param_names):
            config_refs.add((sub.value.id, sub.attr))
    return init_params, scalar_attrs, config_refs


def find_init(tree: ast.Module) -> List[Tuple[str, ast.FunctionDef]]:
    out: List[Tuple[str, ast.FunctionDef]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for it in node.body:
                if isinstance(it, ast.FunctionDef) and it.name == "__init__":
                    out.append((node.name, it))
    return out


def compare_one(src_path: str) -> Dict:
    try:
        src = open(src_path).read()
    except Exception as e:
        return {"file": os.path.basename(src_path), "status": "read_error", "error": str(e)}
    try:
        tree = ast.parse(src)
    except Exception as e:
        return {"file": os.path.basename(src_path), "status": "parse_error", "error": str(e)}
    pairs = find_init(tree)
    if not pairs:
        return {"file": os.path.basename(src_path), "status": "no_init", "n_classes": 0}
    rows = []
    for cls_name, init_fn in pairs:
        try:
            ext = _InitExtractor()
            ext.extract(init_fn)
            ip_e = set(ext.init_param_names)
            sa_e = set(ext.scalar_attrs.keys())
            ca_e = set(ext.symbolic_config_attrs.keys())
        except Exception as e:
            rows.append({"class": cls_name, "status": "extractor_error",
                         "error": repr(e)[:200]})
            continue
        ip_o, sa_o, ca_o = oracle_extract(init_fn)
        rows.append({
            "class": cls_name,
            "init_params_oracle": sorted(ip_o),
            "init_params_extractor": sorted(ip_e),
            "init_params_extractor_subset_of_oracle": ip_e.issubset(ip_o),
            "init_params_oracle_minus_extractor": sorted(ip_o - ip_e),
            "scalar_attrs_oracle": sorted(sa_o),
            "scalar_attrs_extractor": sorted(sa_e),
            "scalar_attrs_extractor_subset_of_oracle": sa_e.issubset(sa_o),
            "config_attrs_oracle": sorted(map(list, ca_o)),
            "config_attrs_extractor": sorted(map(list, ca_e)),
            "config_attrs_extractor_subset_of_oracle": ca_e.issubset(ca_o),
            "config_attrs_match_exact": ca_e == ca_o,
        })
    return {"file": os.path.basename(src_path), "status": "ok", "rows": rows}


def main() -> None:
    summary = {
        "_question": (
            "Round-3 W5: cross-validate the AST extractor (_InitExtractor) "
            "against an independent simple-AST oracle on every available "
            "in-repo corpus, to bound the synthesis-error rate that could "
            "silently inflate Contract-Violation verdicts."
        ),
        "corpora": [],
    }
    overall = {
        "n_classes": 0,
        "config_attrs_extractor_is_subset_of_oracle": 0,
        "config_attrs_match_exact": 0,
        "scalar_attrs_extractor_is_subset_of_oracle": 0,
        "init_params_extractor_is_subset_of_oracle": 0,
        "extractor_errors": 0,
    }
    for corpus_name, corpus_dir in CORPORA:
        if not os.path.isdir(corpus_dir):
            summary["corpora"].append({
                "name": corpus_name, "dir": corpus_dir, "status": "missing"
            })
            continue
        files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".py"))
        files = [os.path.join(corpus_dir, f) for f in files]
        per = []
        c_n = 0
        c_ca_subset = 0
        c_ca_match = 0
        c_sa_subset = 0
        c_ip_subset = 0
        c_err = 0
        for fp in files:
            r = compare_one(fp)
            per.append(r)
            for row in r.get("rows", []):
                if row.get("status") == "extractor_error":
                    c_err += 1
                    continue
                c_n += 1
                if row["config_attrs_extractor_subset_of_oracle"]:
                    c_ca_subset += 1
                if row["config_attrs_match_exact"]:
                    c_ca_match += 1
                if row["scalar_attrs_extractor_subset_of_oracle"]:
                    c_sa_subset += 1
                if row["init_params_extractor_subset_of_oracle"]:
                    c_ip_subset += 1
        summary["corpora"].append({
            "name": corpus_name,
            "dir": os.path.relpath(corpus_dir, ROOT),
            "n_files": len(files),
            "n_classes": c_n,
            "extractor_errors": c_err,
            "config_attrs_extractor_is_subset_of_oracle": c_ca_subset,
            "config_attrs_match_exact": c_ca_match,
            "scalar_attrs_extractor_is_subset_of_oracle": c_sa_subset,
            "init_params_extractor_is_subset_of_oracle": c_ip_subset,
            "per_file": per,
        })
        overall["n_classes"] += c_n
        overall["config_attrs_extractor_is_subset_of_oracle"] += c_ca_subset
        overall["config_attrs_match_exact"] += c_ca_match
        overall["scalar_attrs_extractor_is_subset_of_oracle"] += c_sa_subset
        overall["init_params_extractor_is_subset_of_oracle"] += c_ip_subset
        overall["extractor_errors"] += c_err

    summary["overall"] = overall
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    n = max(1, overall["n_classes"])
    md = []
    md.append("# AST extractor cross-validation against an independent simple-AST oracle\n")
    md.append("Round-3 W5: validates that `_InitExtractor` in "
              "`src/model_checker.py` (the component synthesising "
              "`assume_M` from class source, in the TCB) does not "
              "over-extract relative to a deliberately minimal "
              "second AST traversal whose only dependencies are "
              "Python's standard `ast` module and the shared "
              "config-name predicate `_is_config_param_name`.\n")
    md.append("## Headline\n")
    md.append(f"- Total classes scanned: **{overall['n_classes']}**")
    md.append(f"- Extractor errors: **{overall['extractor_errors']}**")
    md.append(f"- `symbolic_config_attrs` ⊆ oracle config refs: "
              f"**{overall['config_attrs_extractor_is_subset_of_oracle']}/{n}** "
              f"({100*overall['config_attrs_extractor_is_subset_of_oracle']/n:.1f}%)")
    md.append(f"- `symbolic_config_attrs` exactly equals oracle config refs: "
              f"**{overall['config_attrs_match_exact']}/{n}** "
              f"({100*overall['config_attrs_match_exact']/n:.1f}%)")
    md.append(f"- `scalar_attrs` ⊆ oracle scalar-attr writes: "
              f"**{overall['scalar_attrs_extractor_is_subset_of_oracle']}/{n}** "
              f"({100*overall['scalar_attrs_extractor_is_subset_of_oracle']/n:.1f}%)")
    md.append(f"- `init_param_names` ⊆ oracle init-param set: "
              f"**{overall['init_params_extractor_is_subset_of_oracle']}/{n}** "
              f"({100*overall['init_params_extractor_is_subset_of_oracle']/n:.1f}%)\n")
    md.append("## Reading guide\n")
    md.append("The deployed extractor must never over-extract: every "
              "`(config_param, attr)` it treats as a sym-attr must "
              "appear as a literal `<config>.<attr>` AST node in `__init__`. "
              "The oracle enumerates exactly those nodes, so `extractor ⊆ "
              "oracle` is the soundness direction. The extractor may legitimately "
              "*under*-extract (e.g. drop reassigned attributes, restrict to "
              "constructor-bound scalars), so `oracle = extractor` is not "
              "required for soundness; it is reported as an informational "
              "metric.\n")
    md.append("Any class where `extractor ⊄ oracle` is a candidate "
              "synthesis error and would falsify W5 of round 3; "
              "see `per_file` in the JSON for the exact deltas.\n")
    md.append("## Per-corpus\n")
    md.append("| corpus | files | classes | err | ⊆-config | =-config | ⊆-scalar | ⊆-init |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in summary["corpora"]:
        if c.get("status") == "missing":
            md.append(f"| {c['name']} | (missing) | -- | -- | -- | -- | -- | -- |")
            continue
        md.append(
            f"| {c['name']} | {c['n_files']} | {c['n_classes']} | "
            f"{c['extractor_errors']} | "
            f"{c['config_attrs_extractor_is_subset_of_oracle']} | "
            f"{c['config_attrs_match_exact']} | "
            f"{c['scalar_attrs_extractor_is_subset_of_oracle']} | "
            f"{c['init_params_extractor_is_subset_of_oracle']} |"
        )
    md.append("\n## Reproduce\n")
    md.append("    PYTHONPATH=. python3 reproducibility/ast_extractor_oracle_validation.py\n")
    md.append("## Paper claim cited\n")
    md.append("Eval §4.1, §4.4: the AST extractor that synthesises "
              "`assume_M` is in the TCB; this artefact bounds its "
              "soundness-direction agreement against an independent "
              "minimal oracle on the in-repo corpora.\n")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
