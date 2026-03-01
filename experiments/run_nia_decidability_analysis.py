"""NIA decidability analysis for relational shape constraints.

Creates benchmark models with relational constraints of varying nonlinearity,
classifies each as QF_LIA-reducible or genuine QF_NIA, tests Z3's actual
behaviour (SAT/UNSAT/UNKNOWN/timeout), and reports statistics.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decidability import (
    RelationalConstraintClass,
    RelationalConstraintClassifier,
    analyze_nia_fragment,
)

# ── Benchmark suite ──────────────────────────────────────────────────────────

BENCHMARKS = [
    # ── QF_LIA-reducible (at most one symbolic factor) ───────────────────
    {
        "name": "mha_fixed_heads",
        "description": "embed_dim = 8 * head_dim (heads=8 concrete)",
        "constraints": {"embed_dim": "8 * head_dim"},
        "concrete_dims": {"8": 8},  # 8 is a literal, not a dim name
    },
    {
        "name": "mha_fixed_head_dim",
        "description": "embed_dim = heads * 64 (head_dim=64 concrete)",
        "constraints": {"embed_dim": "heads * 64"},
        "concrete_dims": {},
    },
    {
        "name": "linear_sum",
        "description": "total = a + b (pure addition, linear)",
        "constraints": {"total": "a + b"},
        "concrete_dims": {},
    },
    {
        "name": "linear_difference",
        "description": "out_dim = in_dim - padding (subtraction, linear)",
        "constraints": {"out_dim": "in_dim - padding"},
        "concrete_dims": {},
    },
    {
        "name": "concrete_assignment",
        "description": "embed_dim = 512 (constant)",
        "constraints": {"embed_dim": 512},
        "concrete_dims": {},
    },
    {
        "name": "scaled_single_var",
        "description": "ffn_dim = 4 * embed_dim (one symbolic variable)",
        "constraints": {"ffn_dim": "4 * embed_dim"},
        "concrete_dims": {},
    },
    {
        "name": "conv_output_linear",
        "description": "out_h = in_h - 2 (linear conv output, kernel fixed)",
        "constraints": {"out_h": "in_h - 2"},
        "concrete_dims": {},
    },
    {
        "name": "double_linear",
        "description": "proj_dim = 2 * embed_dim, ffn_dim = 4 * embed_dim",
        "constraints": {"proj_dim": "2 * embed_dim", "ffn_dim": "4 * embed_dim"},
        "concrete_dims": {},
    },
    {
        "name": "fixed_heads_multi_constraint",
        "description": "Multiple constraints with heads=8 concrete",
        "constraints": {
            "embed_dim": "8 * head_dim",
            "ffn_dim": "4 * embed_dim",
        },
        "concrete_dims": {"8": 8},
    },
    {
        "name": "identity_constraint",
        "description": "out_dim = in_dim (trivial identity)",
        "constraints": {"out_dim": "in_dim"},
        "concrete_dims": {},
    },
    # ── QF_NIA (genuine nonlinear: ≥2 symbolic factors) ──────────────────
    {
        "name": "mha_all_symbolic",
        "description": "embed_dim = heads * head_dim (all symbolic)",
        "constraints": {"embed_dim": "heads * head_dim"},
        "concrete_dims": {},
    },
    {
        "name": "mha_full_transformer",
        "description": "Transformer: embed=h*d, ffn=4*embed, all symbolic",
        "constraints": {
            "embed_dim": "heads * head_dim",
            "ffn_dim": "4 * embed_dim",
        },
        "concrete_dims": {},
    },
    {
        "name": "reshape_product",
        "description": "flat = height * width (reshape, both symbolic)",
        "constraints": {"flat": "height * width"},
        "concrete_dims": {},
    },
    {
        "name": "volume_3d",
        "description": "volume = d * h * w (triple product, all symbolic)",
        "constraints": {"volume": "d * h * w"},
        "concrete_dims": {},
    },
    {
        "name": "cross_attention",
        "description": "Cross-attention: q_dim = h*d_k, kv_dim = h*d_v",
        "constraints": {
            "q_dim": "heads * d_k",
            "kv_dim": "heads * d_v",
        },
        "concrete_dims": {},
    },
    {
        "name": "conv2d_symbolic",
        "description": "out = channels * kernel_h * kernel_w (all symbolic)",
        "constraints": {"out_features": "channels * kernel_h * kernel_w"},
        "concrete_dims": {},
    },
    {
        "name": "batch_seq_product",
        "description": "total_tokens = batch * seq_len (both symbolic)",
        "constraints": {"total_tokens": "batch * seq_len"},
        "concrete_dims": {},
    },
    {
        "name": "multi_head_kv_grouped",
        "description": "Grouped-query: kv_heads * head_dim, num_groups * kv_heads",
        "constraints": {
            "kv_dim": "kv_heads * head_dim",
            "total_heads": "num_groups * kv_heads",
        },
        "concrete_dims": {},
    },
    {
        "name": "strided_conv_nonlinear",
        "description": "out_dim = in_dim * stride (both symbolic)",
        "constraints": {"out_dim": "in_dim * stride"},
        "concrete_dims": {},
    },
    {
        "name": "factored_embed",
        "description": "embed = factor_a * factor_b * factor_c (triple product)",
        "constraints": {"embed": "factor_a * factor_b * factor_c"},
        "concrete_dims": {},
    },
    {
        "name": "nia_with_addition",
        "description": "out = a * b + c * d (two nonlinear terms added)",
        "constraints": {"out": "a * b + c * d"},
        "concrete_dims": {},
    },
    {
        "name": "quadratic_single_var",
        "description": "area = side * side (single var squared — still NIA)",
        "constraints": {"area": "side * side"},
        "concrete_dims": {},
    },
]


def run_analysis():
    """Run the full NIA decidability analysis and return results."""
    results = []
    lia_count = 0
    nia_count = 0
    nia_sat = 0
    nia_unsat = 0
    nia_unknown = 0
    total_time = 0.0

    for bench in BENCHMARKS:
        name = bench["name"]
        constraints = bench["constraints"]
        concrete = bench.get("concrete_dims", {})

        classifier = RelationalConstraintClassifier(concrete)
        infos = classifier.classify_all(constraints)

        # Overall classification: QF_NIA if any constraint is NIA.
        has_nia = any(
            i.classification == RelationalConstraintClass.QF_NIA for i in infos
        )
        overall = (
            RelationalConstraintClass.QF_NIA if has_nia
            else RelationalConstraintClass.QF_LIA_REDUCIBLE
        )

        # Run Z3.
        nia_result = analyze_nia_fragment(constraints, timeout_ms=5000)

        entry = {
            "name": name,
            "description": bench["description"],
            "constraints": {
                k: v if isinstance(v, int) else v
                for k, v in constraints.items()
            },
            "classification": overall.value,
            "per_constraint": [
                {
                    "lhs": i.lhs,
                    "expression": i.expression,
                    "classification": i.classification.value,
                    "symbolic_vars": i.symbolic_vars,
                    "reason": i.reason,
                }
                for i in infos
            ],
            "z3_status": nia_result.status,
            "z3_elapsed_s": round(nia_result.elapsed_s, 6),
            "z3_timed_out": nia_result.timed_out,
            "z3_model": nia_result.model,
        }
        results.append(entry)

        # Accumulate stats.
        if overall == RelationalConstraintClass.QF_LIA_REDUCIBLE:
            lia_count += 1
        else:
            nia_count += 1
            if nia_result.status == "sat":
                nia_sat += 1
            elif nia_result.status == "unsat":
                nia_unsat += 1
            else:
                nia_unknown += 1
        total_time += nia_result.elapsed_s

        status_icon = "✓" if nia_result.status == "sat" else (
            "✗" if nia_result.status == "unsat" else "?"
        )
        print(
            f"  [{status_icon}] {name:40s} "
            f"{overall.value:20s}  "
            f"z3={nia_result.status:8s}  "
            f"{nia_result.elapsed_s*1000:8.2f}ms"
        )

    # Summary.
    total = len(BENCHMARKS)
    nia_solved = nia_sat + nia_unsat
    summary = {
        "total_benchmarks": total,
        "qf_lia_reducible": lia_count,
        "qf_nia": nia_count,
        "reducible_fraction": round(lia_count / total, 4) if total else 0,
        "nia_z3_sat": nia_sat,
        "nia_z3_unsat": nia_unsat,
        "nia_z3_unknown": nia_unknown,
        "nia_z3_success_rate": (
            round(nia_solved / nia_count, 4) if nia_count else 1.0
        ),
        "total_z3_time_s": round(total_time, 6),
    }

    return {"summary": summary, "benchmarks": results}


def main():
    print("=" * 72)
    print("NIA Decidability Analysis for Relational Shape Constraints")
    print("=" * 72)
    print()

    output = run_analysis()
    summary = output["summary"]

    print()
    print("-" * 72)
    print("Summary")
    print("-" * 72)
    print(f"  Total benchmarks:       {summary['total_benchmarks']}")
    print(f"  QF_LIA-reducible:       {summary['qf_lia_reducible']}"
          f"  ({summary['reducible_fraction']*100:.1f}%)")
    print(f"  Genuine QF_NIA:         {summary['qf_nia']}")
    print(f"  Z3 success on NIA:      "
          f"{summary['nia_z3_sat']+summary['nia_z3_unsat']}"
          f"/{summary['qf_nia']}"
          f"  ({summary['nia_z3_success_rate']*100:.1f}%)")
    print(f"    SAT:    {summary['nia_z3_sat']}")
    print(f"    UNSAT:  {summary['nia_z3_unsat']}")
    print(f"    UNKNOWN:{summary['nia_z3_unknown']}")
    print(f"  Total Z3 time:          {summary['total_z3_time_s']*1000:.2f}ms")
    print()

    out_path = os.path.join(
        os.path.dirname(__file__), "nia_decidability_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
