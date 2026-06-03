"""Stage-wise ablations for TensorGuard's verification stack (Step 253).

This harness complements the earlier per-domain, reduced-product, and CEGAR
ablations by isolating every major stage reviewers ask about:

* graph extraction from real ``nn.Module`` source;
* each abstract domain (shape, dtype, device, gradient, phase);
* every registered cross-domain reduction rule;
* CEGAR refinement depth;
* third-party shape stubs; and
* proof-backed versus heuristic operator rules in sound mode.

Every row exercises live TensorGuard code paths.  The artifact records only
counts, booleans, verdicts, and finite labels, so regeneration is
byte-identical across machines.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.domain_ablation_corpus import (  # noqa: E402
    DIAGNOSTIC_DOMAINS,
    TAG_TO_DOMAIN,
    VERIFICATION_DOMAINS,
    build_corpus as build_domain_corpus,
)
from reproducibility import cegar_depth_ablation  # noqa: E402
from src.api import verify_architecture  # noqa: E402
from src.confidence_tags import ConfidenceTag  # noqa: E402
from src.domains.intervals import Bound, Interval, IntervalValue  # noqa: E402
from src.domains.nullity import NullityValue  # noqa: E402
from src.domains.product import (  # noqa: E402
    NumericToNullityReduction,
    NullityToTypeTagReduction,
    ProductValue,
    ReducedProductDomain,
    Reduction,
    ReductionEngine,
    TruthinessReduction,
    TypeTagToNullityReduction,
    TypeTagToNumericReduction,
)
from src.domains.typetags import TypeTagSet, TypeTagValue  # noqa: E402
from src.model_checker import (  # noqa: E402
    LayerKind,
    extract_computation_graph,
    verify_model,
)
from src.proof_footprint import (  # noqa: E402
    ProofStatus,
    footprint_for,
    proof_footprint_table,
    summary_for,
)
from src.shape_stub_registry import (  # noqa: E402
    clear_user_stubs,
    register_last_dim_linear,
)

OUT_JSON = REPO / "reproducibility" / "stagewise_ablation.json"
OUT_MD = REPO / "reproducibility" / "stagewise_ablation.md"

_TAG_RE = re.compile(r"\[([A-Z][A-Z-]+)\]")
_DOMAIN_SEED = 20240605


EXTRACTION_SRC = textwrap.dedent(
    """
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 16)
            self.b = nn.Linear(7, 4)

        def forward(self, x):
            return self.b(self.a(x))
    """
)

STUB_SRC = textwrap.dedent(
    """
    import torch.nn as nn
    from thirdparty import FancyBlock

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = FancyBlock(8, 16)
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            return self.head(self.block(x))
    """
)

PROOF_BACKED_SRC = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def forward(self, x):
            return torch.relu(x)
    """
)

HEURISTIC_SRC = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def forward(self, x):
            return torch.unique(x)
    """
)


def _bug_tags(bugs: Sequence[Any]) -> List[str]:
    tags: List[str] = []
    for bug in bugs:
        msg = getattr(bug, "message", "")
        match = _TAG_RE.match(msg)
        if match:
            tags.append(match.group(1))
    return sorted(tags)


def _caught_after_report_ablation(bugs: Sequence[Any], ablated_domain: str) -> bool:
    for bug in bugs:
        msg = getattr(bug, "message", "")
        match = _TAG_RE.match(msg)
        domain = TAG_TO_DOMAIN.get(match.group(1)) if match else None
        if domain != ablated_domain:
            return True
    return False


def _verify_domain_case(source: str, input_shape: Sequence[int], **kwargs: Any):
    return verify_architecture(
        source,
        input_shapes={"x": tuple(input_shape)},
        soundness_mode="sound",
        max_cegar_iterations=0,
        infer_inputs=False,
        **kwargs,
    )


def measure_extraction() -> Dict[str, object]:
    graph = extract_computation_graph(EXTRACTION_SRC)
    result = verify_architecture(
        EXTRACTION_SRC,
        input_shapes={"x": (2, 8)},
        soundness_mode="sound",
        max_cegar_iterations=0,
        infer_inputs=False,
    )
    tags = _bug_tags(result.bugs)
    return {
        "case_id": "two_linear_shape_mismatch",
        "without_extraction_analyzable_modules": 0,
        "with_extraction_layers": len(graph.layers),
        "with_extraction_steps": len(graph.steps),
        "with_extraction_inputs": sorted(graph.input_names),
        "with_extraction_outputs": sorted(graph.output_names),
        "caught_after_extraction": result.verdict == "UNSAFE",
        "bug_tags": tags,
    }


def measure_domains() -> Dict[str, object]:
    logging.disable(logging.CRITICAL)
    try:
        cases = build_domain_corpus(seed=_DOMAIN_SEED, n_per_domain=1)
        by_domain = {case.domain: case for case in cases}
        per_domain: Dict[str, dict] = {}

        for domain in list(VERIFICATION_DOMAINS) + list(DIAGNOSTIC_DOMAINS):
            case = by_domain[domain]
            full = _verify_domain_case(case.source, case.input_shape)
            full_caught = bool(full.bugs)
            ablation_kind = "report_filter"

            if domain == "device":
                ablated = _verify_domain_case(
                    case.source, case.input_shape, check_devices=False
                )
                ablated_caught = bool(ablated.bugs)
                ablation_kind = "runtime_flag"
            elif domain == "gradient":
                ablated = _verify_domain_case(
                    case.source, case.input_shape, check_gradients=False
                )
                ablated_caught = bool(ablated.bugs)
                ablation_kind = "runtime_flag"
            elif domain == "phase":
                ablated = _verify_domain_case(
                    case.source, case.input_shape, check_phases=False
                )
                ablated_caught = bool(ablated.bugs)
                ablation_kind = "diagnostic_flag"
            else:
                ablated_caught = _caught_after_report_ablation(full.bugs, domain)

            per_domain[domain] = {
                "case_id": case.case_id,
                "ablation_kind": ablation_kind,
                "full_caught": full_caught,
                "ablated_caught": ablated_caught,
                "delta": int(full_caught) - int(ablated_caught),
                "full_bug_tags": _bug_tags(full.bugs),
                "full_verdict": full.verdict,
            }

        return {
            "seed": _DOMAIN_SEED,
            "n_cases": len(per_domain),
            "verification_domains": list(VERIFICATION_DOMAINS),
            "diagnostic_domains": list(DIAGNOSTIC_DOMAINS),
            "per_domain": per_domain,
            "each_verification_domain_load_bearing": all(
                per_domain[d]["full_caught"] and not per_domain[d]["ablated_caught"]
                for d in VERIFICATION_DOMAINS
            ),
            "phase_is_diagnostic_only": (
                not per_domain["phase"]["full_caught"]
                and not per_domain["phase"]["ablated_caught"]
            ),
        }
    finally:
        logging.disable(logging.NOTSET)


def _pv(
    *,
    interval: Interval | None = None,
    tags: TypeTagSet | None = None,
    nullity: NullityValue | None = None,
) -> ProductValue:
    return ProductValue(
        interval=IntervalValue(interval or Interval.top()),
        type_tag=TypeTagValue(tags or TypeTagSet.top()),
        nullity=nullity or NullityValue.maybe_null(),
    )


def _bound_repr(bound: Bound) -> str | int:
    if bound.is_neg_inf:
        return "-inf"
    if bound.is_pos_inf:
        return "inf"
    return int(bound.value)


def _interval_repr(interval: Interval) -> str | List[str | int]:
    if interval.is_bottom:
        return "BOTTOM"
    if interval.is_top:
        return "TOP"
    return [_bound_repr(interval.lo), _bound_repr(interval.hi)]


def _tags_repr(tag_set: TypeTagSet) -> str | List[str]:
    if tag_set.is_top:
        return "TOP"
    if tag_set.is_bottom:
        return "BOTTOM"
    return sorted(tag_set.tags)


def _snapshot(value: ProductValue) -> Dict[str, object]:
    return {
        "interval": _interval_repr(value.interval.interval),
        "type_tags": _tags_repr(value.type_tag.tag_set),
        "nullity": value.nullity.kind.name,
        "is_bottom": value.is_bottom(),
    }


def _changed_components(before: ProductValue, after: ProductValue) -> List[str]:
    changed = []
    if before.interval != after.interval:
        changed.append("interval")
    if before.type_tag != after.type_tag:
        changed.append("type_tag")
    if before.nullity != after.nullity:
        changed.append("nullity")
    return changed


_REDUCTION_CASES: Mapping[str, Callable[[], ProductValue]] = {
    "TypeTagToNullityReduction": lambda: _pv(
        tags=TypeTagSet.singleton("int"),
        nullity=NullityValue.maybe_null(),
    ),
    "NullityToTypeTagReduction": lambda: _pv(
        tags=TypeTagSet.from_names("int", "NoneType"),
        nullity=NullityValue.definitely_not_null(),
    ),
    "TypeTagToNumericReduction": lambda: _pv(
        interval=Interval.singleton(3),
        tags=TypeTagSet.singleton("str"),
        nullity=NullityValue.definitely_not_null(),
    ),
    "NumericToNullityReduction": lambda: _pv(
        interval=Interval.positive(),
        tags=TypeTagSet.top(),
        nullity=NullityValue.maybe_null(),
    ),
    "TruthinessReduction": lambda: _pv(
        interval=Interval.from_bounds(0, 5),
        tags=TypeTagSet.from_names("int", "NoneType"),
        nullity=NullityValue.maybe_null(),
    ),
}


def _registered_reduction_classes() -> List[type[Reduction]]:
    return sorted(Reduction.__subclasses__(), key=lambda cls: cls.__name__)


def measure_reductions() -> Dict[str, object]:
    domain = ReducedProductDomain()
    default_engine = ReductionEngine()
    default_names = sorted(r.name() for r in default_engine.reductions)
    per_reduction: Dict[str, dict] = {}

    for cls in _registered_reduction_classes():
        if cls.__name__ not in _REDUCTION_CASES:
            raise AssertionError(f"no stage-wise witness for {cls.__name__}")
        reduction = cls()
        before = _REDUCTION_CASES[cls.__name__]()
        after = reduction.apply(before)
        engine_after = default_engine.reduce(before) if reduction.name() in default_names else None
        row = {
            "reduction_name": reduction.name(),
            "default_engine_member": reduction.name() in default_names,
            "before": _snapshot(before),
            "after": _snapshot(after),
            "changed_components": _changed_components(before, after),
            "changed": before != after,
            "after_refines_before": domain.leq(after, before),
        }
        if engine_after is not None:
            row["default_engine_after"] = _snapshot(engine_after)
            row["default_engine_refines_before"] = domain.leq(engine_after, before)
        per_reduction[cls.__name__] = row

    return {
        "registered_class_names": [cls.__name__ for cls in _registered_reduction_classes()],
        "default_engine_reductions": default_names,
        "per_reduction": per_reduction,
        "all_registered_reductions_exercised": all(
            row["changed"] for row in per_reduction.values()
        ),
        "all_reductions_are_refinements": all(
            row["after_refines_before"] for row in per_reduction.values()
        ),
    }


def measure_cegar() -> Dict[str, object]:
    data = cegar_depth_ablation.measure()
    knee = data["precision_knee_depth"]
    per_depth = [
        {
            "depth": row["depth"],
            "bugs_detected": row["bugs_detected"],
            "refined_contract_diagnoses": row["refined_contract_diagnoses"],
            "clean_false_alarms": row["clean_false_alarms"],
            "total_cegar_iterations": row["total_cegar_iterations"],
        }
        for row in data["per_depth"]
    ]
    return {
        "n_conflict_cases": data["n_conflict_cases"],
        "n_clean_cases": data["n_clean_cases"],
        "precision_knee_depth": knee,
        "work_saturation_depth": data["work_saturation_depth"],
        "refined_diagnoses_at_depth_0": data["refined_diagnoses_at_depth_0"],
        "refined_diagnoses_at_knee": data["refined_diagnoses_at_knee"],
        "recall_is_depth_invariant_full": data["recall_is_depth_invariant_full"],
        "zero_false_alarms_all_depths": data["zero_false_alarms_all_depths"],
        "precision_rises_then_plateaus": data["precision_rises_then_plateaus"],
        "work_saturates_at_convergence": data["work_saturates_at_convergence"],
        "per_depth": per_depth,
    }


def measure_stubs() -> Dict[str, object]:
    bad_head = STUB_SRC.replace("nn.Linear(16, 4)", "nn.Linear(99, 4)")
    bad_input = STUB_SRC

    clear_user_stubs()
    try:
        graph_without = extract_computation_graph(STUB_SRC)
        without_layer = graph_without.layers.get("block")
        without_kind = (
            without_layer.kind.name if without_layer is not None else "(missing)"
        )
        good_without = verify_model(STUB_SRC, input_shapes={"x": (2, 8)})
        bad_head_without = verify_model(bad_head, input_shapes={"x": (2, 8)})
        bad_input_without = verify_model(bad_input, input_shapes={"x": (2, 5)})

        register_last_dim_linear(
            "FancyBlock",
            in_arg="in_features",
            out_arg="out_features",
            arg_names=("in_features", "out_features"),
        )
        graph_with = extract_computation_graph(STUB_SRC)
        with_layer = graph_with.layers.get("block")
        with_kind = with_layer.kind.name if with_layer is not None else "(missing)"
        good_with = verify_model(STUB_SRC, input_shapes={"x": (2, 8)})
        bad_head_with = verify_model(bad_head, input_shapes={"x": (2, 8)})
        bad_input_with = verify_model(bad_input, input_shapes={"x": (2, 5)})

        return {
            "without_stub_layer_kind": without_kind,
            "with_stub_layer_kind": with_kind,
            "stub_params": dict(with_layer.params or {}) if with_layer else {},
            "valid_model_safe_without_stub": bool(good_without.safe),
            "valid_model_safe_with_stub": bool(good_with.safe),
            "bad_head_caught_without_stub": not bool(bad_head_without.safe),
            "bad_head_caught_with_stub": not bool(bad_head_with.safe),
            "bad_input_caught_without_stub": not bool(bad_input_without.safe),
            "bad_input_caught_with_stub": not bool(bad_input_with.safe),
            "stub_stage_load_bearing": (
                with_kind == LayerKind.STUB.name
                and without_kind != LayerKind.STUB.name
                and not bool(good_without.safe)
                and bool(good_with.safe)
                and not bool(bad_input_with.safe)
            ),
        }
    finally:
        clear_user_stubs()


def measure_proof_rules() -> Dict[str, object]:
    proof_case = verify_architecture(
        PROOF_BACKED_SRC,
        input_shapes={"x": (2, 3)},
        soundness_mode="sound",
        max_cegar_iterations=0,
        infer_inputs=False,
    )
    heuristic_case = verify_architecture(
        HEURISTIC_SRC,
        input_shapes={"x": (2, 3)},
        soundness_mode="sound",
        max_cegar_iterations=0,
        infer_inputs=False,
    )
    rows = proof_footprint_table()
    heuristic_rows = [
        row for row in rows
        if row["confidence"] == ConfidenceTag.HEURISTIC.value
    ]
    return {
        "proof_backed_case": {
            "operator": "torch.relu",
            "proof_status": footprint_for("torch.relu")["proof_status"],
            "confidence": footprint_for("torch.relu")["confidence"],
            "verdict": proof_case.verdict,
            "abstained": proof_case.abstained,
            "unknown_reasons": sorted(proof_case.unknown_reasons),
        },
        "heuristic_case": {
            "operator": "torch.unique",
            "proof_status": footprint_for("torch.unique")["proof_status"],
            "confidence": footprint_for("torch.unique")["confidence"],
            "verdict": heuristic_case.verdict,
            "abstained": heuristic_case.abstained,
            "unknown_reasons": sorted(heuristic_case.unknown_reasons),
        },
        "proof_backed_safe_without_abstention": (
            proof_case.verdict == "SAFE" and not proof_case.abstained
        ),
        "heuristic_abstains_in_sound_mode": (
            heuristic_case.verdict == "UNKNOWN"
            and any("heuristic-tagged operator" in r for r in heuristic_case.unknown_reasons)
        ),
        "heuristic_rows": sorted(str(row["operator"]) for row in heuristic_rows),
        "heuristic_rows_all_heuristic_footprints": all(
            row["proof_status"] == ProofStatus.HEURISTIC.value
            for row in heuristic_rows
        ),
        "proof_footprint_summary": summary_for(rows),
    }


def measure() -> Dict[str, object]:
    data = {
        "schema": "tensorguard.stagewise_ablation/v1",
        "step": 253,
        "stage_order": [
            "extraction",
            "abstract_domains",
            "cross_domain_reductions",
            "cegar",
            "stubs",
            "proof_rules",
        ],
        "extraction": measure_extraction(),
        "abstract_domains": measure_domains(),
        "cross_domain_reductions": measure_reductions(),
        "cegar": measure_cegar(),
        "stubs": measure_stubs(),
        "proof_rules": measure_proof_rules(),
    }
    data["headline_claims"] = {
        "extraction_is_live": data["extraction"]["caught_after_extraction"],
        "every_verification_domain_is_load_bearing": data["abstract_domains"][
            "each_verification_domain_load_bearing"
        ],
        "every_registered_reduction_is_exercised": data["cross_domain_reductions"][
            "all_registered_reductions_exercised"
        ],
        "cegar_refinement_has_a_diagnosis_knee": data["cegar"][
            "precision_rises_then_plateaus"
        ],
        "stubs_are_load_bearing": data["stubs"]["stub_stage_load_bearing"],
        "sound_mode_separates_proof_backed_from_heuristic": (
            data["proof_rules"]["proof_backed_safe_without_abstention"]
            and data["proof_rules"]["heuristic_abstains_in_sound_mode"]
        ),
    }
    return data


def render_markdown(data: Dict[str, object]) -> str:
    extraction = data["extraction"]  # type: ignore[index]
    domains = data["abstract_domains"]  # type: ignore[index]
    reductions = data["cross_domain_reductions"]  # type: ignore[index]
    cegar = data["cegar"]  # type: ignore[index]
    stubs = data["stubs"]  # type: ignore[index]
    proof = data["proof_rules"]  # type: ignore[index]

    lines = [
        "# Stage-wise ablation stack (Step 253)",
        "",
        "This artifact isolates TensorGuard's major verification stages with live "
        "code paths: extraction, abstract domains, cross-domain reductions, CEGAR, "
        "third-party stubs, and proof-backed versus heuristic rule policy.",
        "",
        "## Headline",
        "",
        "| stage | live check | outcome |",
        "| --- | --- | --- |",
        (
            f"| extraction | graph has {extraction['with_extraction_layers']} layers "
            f"/ {extraction['with_extraction_steps']} steps and catches the seeded "
            f"shape bug | {extraction['caught_after_extraction']} |"
        ),
        (
            "| abstract domains | each verification domain loses its own detection "
            f"when ablated; phase stays diagnostic-only | "
            f"{domains['each_verification_domain_load_bearing']} / "
            f"{domains['phase_is_diagnostic_only']} |"
        ),
        (
            "| cross-domain reductions | every registered reduction has a witness "
            f"and refines its input | {reductions['all_registered_reductions_exercised']} / "
            f"{reductions['all_reductions_are_refinements']} |"
        ),
        (
            "| CEGAR | refined-contract diagnoses rise from "
            f"{cegar['refined_diagnoses_at_depth_0']} to "
            f"{cegar['refined_diagnoses_at_knee']} at depth "
            f"{cegar['precision_knee_depth']} | "
            f"{cegar['precision_rises_then_plateaus']} |"
        ),
        (
            "| stubs | registering `FancyBlock` turns an opaque clean model into a "
            f"SAFE `{stubs['with_stub_layer_kind']}` model and catches bad contracts | "
            f"{stubs['stub_stage_load_bearing']} |"
        ),
        (
            "| proof rules | `torch.relu` is SAFE in sound mode while heuristic "
            f"`torch.unique` abstains with a heuristic reason | "
            f"{proof['proof_backed_safe_without_abstention']} / "
            f"{proof['heuristic_abstains_in_sound_mode']} |"
        ),
        "",
        "## Per-domain ablation",
        "",
        "| domain | mode | full caught | ablated caught | tags |",
        "| --- | --- | --- | --- | --- |",
    ]
    for domain, row in domains["per_domain"].items():  # type: ignore[index]
        tags = ", ".join(row["full_bug_tags"]) or "-"
        lines.append(
            f"| {domain} | {row['ablation_kind']} | {row['full_caught']} "
            f"| {row['ablated_caught']} | {tags} |"
        )

    lines += [
        "",
        "## Per-reduction witnesses",
        "",
        "| reduction class | rule | changed components | default engine? |",
        "| --- | --- | --- | --- |",
    ]
    for cls_name, row in reductions["per_reduction"].items():  # type: ignore[index]
        changed = ", ".join(row["changed_components"]) or "-"
        lines.append(
            f"| {cls_name} | {row['reduction_name']} | {changed} "
            f"| {row['default_engine_member']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("stagewise_ablation: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
