"""
BRANCH_MERGE Decomposition Strategy Benchmarks on Non-Sequential Architectures.

Tests the BRANCH_MERGE strategy (and DAG compositional verification) on
architectures with skip connections, parallel branches, and cross-path
data flow:

  1. U-Net       – skip connections between encoder and decoder paths
  2. ResNet-50   – residual/skip connections with identity blocks
  3. Transformer – cross-attention between encoder and decoder
  4. Inception   – parallel branches with concatenation
  5. FPN         – multi-scale skip connections

For each architecture we:
  - Run with BRANCH_MERGE decomposition strategy
  - Run with LAYER_BOUNDARY for comparison
  - Run monolithic verification for comparison
  - Report decomposition, InterfaceContracts, verification result, timing
  - Identify limitations of BRANCH_MERGE on each topology
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model, extract_computation_graph
from src.assume_guarantee import (
    verify_compositional,
    verify_compositional_dag,
    decompose_graph,
    decompose_graph_dag,
    DecompositionStrategy,
    CompositionalResult,
    InterfaceContract,
    CompositionProofRule,
    DAGCompositionProofRule,
)

# ═════════════════════════════════════════════════════════════════════════════
# Model definitions – non-sequential architectures
# ═════════════════════════════════════════════════════════════════════════════

NONSEQ_MODELS = [
    # ── 1. U-Net (encoder-decoder with skip connections) ─────────────────
    {
        "name": "unet_skip",
        "category": "encoder_decoder",
        "description": "U-Net with skip connections between encoder and decoder paths",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": '''
import torch
import torch.nn as nn

class UNetSmall(nn.Module):
    """Simplified U-Net: 2-level encoder/decoder with skip connections."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        # Bottleneck
        self.bottleneck = nn.Conv2d(32, 64, 3, padding=1)
        # Decoder (channels doubled by cat with skip)
        self.dec2 = nn.Conv2d(96, 32, 3, padding=1)   # 64+32=96
        self.dec1 = nn.Conv2d(48, 16, 3, padding=1)    # 32+16=48
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)      # skip1
        e2 = self.enc2(e1)     # skip2
        # Bottleneck
        b = self.bottleneck(e2)
        # Decoder path with skip connections
        d2 = self.dec2(torch.cat([b, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.out_conv(d1)
''',
    },

    # ── 2. ResNet-50 style (residual blocks with skip connections) ───────
    {
        "name": "resnet_residual",
        "category": "residual",
        "description": "ResNet-style residual blocks with identity skip connections",
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn

class ResNetBlock(nn.Module):
    """ResNet-style network with 3 residual blocks."""
    def __init__(self):
        super().__init__()
        # Block 1
        self.b1_fc1 = nn.Linear(64, 64)
        self.b1_fc2 = nn.Linear(64, 64)
        # Block 2
        self.b2_fc1 = nn.Linear(64, 64)
        self.b2_fc2 = nn.Linear(64, 64)
        # Block 3
        self.b3_fc1 = nn.Linear(64, 64)
        self.b3_fc2 = nn.Linear(64, 64)
        # Classifier
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Residual block 1
        residual = x
        x = self.b1_fc1(x)
        x = self.b1_fc2(x)
        x = x + residual
        # Residual block 2
        residual = x
        x = self.b2_fc1(x)
        x = self.b2_fc2(x)
        x = x + residual
        # Residual block 3
        residual = x
        x = self.b3_fc1(x)
        x = self.b3_fc2(x)
        x = x + residual
        return self.classifier(x)
''',
    },

    # ── 3. Transformer encoder-decoder (cross-attention) ─────────────────
    {
        "name": "transformer_encdec",
        "category": "attention",
        "description": "Transformer encoder-decoder with cross-attention projection",
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn

class TransformerEncDec(nn.Module):
    """Simplified Transformer encoder-decoder with cross-attention."""
    def __init__(self):
        super().__init__()
        # Encoder self-attention projections
        self.enc_q = nn.Linear(64, 64)
        self.enc_k = nn.Linear(64, 64)
        self.enc_v = nn.Linear(64, 64)
        self.enc_proj = nn.Linear(64, 64)
        self.enc_ff1 = nn.Linear(64, 128)
        self.enc_ff2 = nn.Linear(128, 64)
        # Decoder self-attention
        self.dec_q = nn.Linear(64, 64)
        self.dec_k = nn.Linear(64, 64)
        self.dec_v = nn.Linear(64, 64)
        self.dec_proj = nn.Linear(64, 64)
        # Decoder cross-attention (K,V from encoder)
        self.cross_q = nn.Linear(64, 64)
        self.cross_k = nn.Linear(64, 64)
        self.cross_v = nn.Linear(64, 64)
        self.cross_proj = nn.Linear(64, 64)
        self.dec_ff1 = nn.Linear(64, 128)
        self.dec_ff2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Encoder
        eq = self.enc_q(x)
        ek = self.enc_k(x)
        ev = self.enc_v(x)
        enc_out = self.enc_proj(ev)
        enc_out = self.enc_ff1(enc_out)
        enc_out = self.enc_ff2(enc_out)
        # Decoder self-attention
        dq = self.dec_q(x)
        dk = self.dec_k(x)
        dv = self.dec_v(x)
        dec_self = self.dec_proj(dv)
        # Decoder cross-attention using encoder output
        cq = self.cross_q(dec_self)
        ck = self.cross_k(enc_out)
        cv = self.cross_v(enc_out)
        cross_out = self.cross_proj(cv)
        # Decoder feed-forward
        out = self.dec_ff1(cross_out)
        out = self.dec_ff2(out)
        return self.output(out)
''',
    },

    # ── 4. Inception module (parallel branches with concatenation) ───────
    {
        "name": "inception_module",
        "category": "parallel",
        "description": "Inception-style module with 3 parallel branches and concatenation",
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    """Inception-style block with 3 parallel branches merged via cat."""
    def __init__(self):
        super().__init__()
        # Branch A: 1x1 (narrow)
        self.branch_a = nn.Linear(128, 32)
        # Branch B: 1x1 -> 3x3 equivalent
        self.branch_b1 = nn.Linear(128, 64)
        self.branch_b2 = nn.Linear(64, 48)
        # Branch C: 1x1 -> 5x5 equivalent
        self.branch_c1 = nn.Linear(128, 32)
        self.branch_c2 = nn.Linear(32, 32)
        self.branch_c3 = nn.Linear(32, 48)
        # Merge: 32 + 48 + 48 = 128
        self.merge_fc = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        a = self.branch_a(x)          # -> 32
        b = self.branch_b1(x)
        b = self.branch_b2(b)         # -> 48
        c = self.branch_c1(x)
        c = self.branch_c2(c)
        c = self.branch_c3(c)         # -> 48
        merged = torch.cat([a, b, c], dim=1)  # -> 128
        out = self.merge_fc(merged)
        return self.output(out)
''',
    },

    # ── 5. Feature Pyramid Network (multi-scale skip connections) ────────
    {
        "name": "fpn_multiscale",
        "category": "encoder_decoder",
        "description": "Feature Pyramid Network with multi-scale lateral connections",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": '''
import torch.nn as nn

class FPNSmall(nn.Module):
    """Simplified FPN with 3-level backbone and top-down lateral connections."""
    def __init__(self):
        super().__init__()
        # Bottom-up backbone
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        # Lateral connections (1x1 convolutions to match channels)
        self.lat3 = nn.Conv2d(64, 32, 1)
        self.lat2 = nn.Conv2d(32, 32, 1)
        # Top-down refinement
        self.refine2 = nn.Conv2d(32, 32, 3, padding=1)
        self.refine1 = nn.Conv2d(32, 16, 3, padding=1)
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Bottom-up
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        c3_out = self.c3(c2_out)
        # Top-down with lateral connections
        p3 = self.lat3(c3_out)
        p2 = self.lat2(c2_out) + p3
        p2 = self.refine2(p2)
        p1 = p2 + c1_out
        p1 = self.refine1(p1)
        return self.out_conv(p1)
''',
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═════════════════════════════════════════════════════════════════════════════

def collect_contract_info(submodules):
    """Extract InterfaceContract details from a list of SubModules."""
    contracts = []
    for sm in submodules:
        contracts.append({
            "submodule": sm.name,
            "step_range": list(sm.step_range),
            "num_steps": sm.graph.num_steps,
            "input_contract": {
                "name": sm.input_contract.name,
                "input_shapes": {
                    k: list(v) for k, v in sm.input_contract.input_shapes.items()
                },
                "output_shapes": {
                    k: list(v) for k, v in sm.input_contract.output_shapes.items()
                },
                "constraints": sm.input_contract.constraints,
            },
            "output_contract": {
                "name": sm.output_contract.name,
                "input_shapes": {
                    k: list(v) for k, v in sm.output_contract.input_shapes.items()
                },
                "output_shapes": {
                    k: list(v) for k, v in sm.output_contract.output_shapes.items()
                },
                "constraints": sm.output_contract.constraints,
            },
        })
    return contracts


def analyze_decomposition(source, input_shapes, strategy):
    """Run decomposition and return analysis dict (no verification)."""
    try:
        graph = extract_computation_graph(source)
    except Exception as exc:
        return {
            "error": str(exc),
            "num_steps": 0,
            "num_submodules": 0,
            "contracts": [],
        }
    submodules = decompose_graph(
        graph, strategy=strategy, input_shapes=input_shapes, min_block_size=1,
    )
    return {
        "num_steps": graph.num_steps,
        "num_submodules": len(submodules),
        "contracts": collect_contract_info(submodules),
    }


def analyze_dag_decomposition(source, input_shapes):
    """Run DAG decomposition and return analysis dict."""
    try:
        graph = extract_computation_graph(source)
    except Exception as exc:
        return {
            "error": str(exc),
            "topology": "unknown",
            "num_submodules": 0,
            "dag_edges": [],
            "contracts": [],
        }
    submodules, edges, topology = decompose_graph_dag(graph, input_shapes)
    return {
        "topology": topology,
        "num_submodules": len(submodules),
        "dag_edges": edges,
        "contracts": collect_contract_info(submodules),
    }


def run_verification(source, input_shapes, strategy, label):
    """Run verify_compositional with timing, return result dict."""
    t0 = time.perf_counter()
    try:
        result = verify_compositional(
            source=source,
            input_shapes=input_shapes,
            strategy=strategy,
            measure_monolithic=False,
            min_block_size=1,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "label": label,
            "strategy": strategy,
            "safe": result.safe,
            "num_submodules": result.num_submodules,
            "time_ms": round(elapsed, 3),
            "cache_hits": result.cache_hits,
            "interface_checks": [
                {
                    "producer": ic.producer,
                    "consumer": ic.consumer,
                    "compatible": ic.compatible,
                    "message": ic.message,
                }
                for ic in result.interface_checks
            ],
            "submodule_results": {
                name: {
                    "safe": r.safe,
                    "errors": r.errors[:3] if r.errors else [],
                    "time_ms": round(r.verification_time_ms, 3),
                }
                for name, r in result.submodule_results.items()
            },
            "error": None,
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "label": label,
            "strategy": strategy,
            "safe": None,
            "num_submodules": 0,
            "time_ms": round(elapsed, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_dag_verification(source, input_shapes, label):
    """Run verify_compositional_dag with timing."""
    t0 = time.perf_counter()
    try:
        result = verify_compositional_dag(
            source=source,
            input_shapes=input_shapes,
            measure_monolithic=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "label": label,
            "strategy": "dag_branch_merge",
            "safe": result.safe,
            "num_submodules": result.num_submodules,
            "time_ms": round(elapsed, 3),
            "cache_hits": result.cache_hits,
            "interface_checks": [
                {
                    "producer": ic.producer,
                    "consumer": ic.consumer,
                    "compatible": ic.compatible,
                    "message": ic.message,
                }
                for ic in result.interface_checks
            ],
            "submodule_results": {
                name: {
                    "safe": r.safe,
                    "errors": r.errors[:3] if r.errors else [],
                    "time_ms": round(r.verification_time_ms, 3),
                }
                for name, r in result.submodule_results.items()
            },
            "error": None,
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "label": label,
            "strategy": "dag_branch_merge",
            "safe": None,
            "num_submodules": 0,
            "time_ms": round(elapsed, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_monolithic(source, input_shapes):
    """Run monolithic verify_model with timing."""
    t0 = time.perf_counter()
    try:
        result = verify_model(source=source, input_shapes=input_shapes)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "safe": result.safe,
            "time_ms": round(elapsed, 3),
            "num_steps": result.graph.num_steps if result.graph else 0,
            "errors": result.errors[:5] if result.errors else [],
            "error": None,
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "safe": None,
            "time_ms": round(elapsed, 3),
            "num_steps": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ═════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment():
    all_results = []

    print("=" * 95)
    print("BRANCH_MERGE Decomposition on Non-Sequential Architectures")
    print("=" * 95)

    for model in NONSEQ_MODELS:
        name = model["name"]
        source = model["source"]
        input_shapes = model["input_shapes"]
        category = model["category"]

        print(f"\n{'─' * 95}")
        print(f"Model: {name} ({category})")
        print(f"  {model['description']}")
        print(f"{'─' * 95}")

        record = {
            "model_name": name,
            "category": category,
            "description": model["description"],
        }

        # ── 1. Monolithic verification ──────────────────────────────────
        print("  [1] Monolithic verification...", end=" ", flush=True)
        mono = run_monolithic(source, input_shapes)
        record["monolithic"] = mono
        safe_str = "SAFE" if mono["safe"] else ("UNSAFE" if mono["safe"] is False else "ERROR")
        print(f"{safe_str}  {mono['time_ms']:.1f}ms  ({mono['num_steps']} steps)")

        # ── 2. BRANCH_MERGE decomposition analysis ─────────────────────
        print("  [2] BRANCH_MERGE decomposition analysis...", end=" ", flush=True)
        bm_decomp = analyze_decomposition(source, input_shapes, "branch_merge")
        record["branch_merge_decomposition"] = bm_decomp
        print(f"{bm_decomp['num_submodules']} submodules from {bm_decomp['num_steps']} steps")

        # ── 3. BRANCH_MERGE verification ───────────────────────────────
        print("  [3] BRANCH_MERGE verification...", end=" ", flush=True)
        bm_result = run_verification(source, input_shapes, "branch_merge", "branch_merge")
        record["branch_merge_verification"] = bm_result
        bm_safe = "SAFE" if bm_result["safe"] else (
            "UNSAFE" if bm_result["safe"] is False else "ERROR"
        )
        print(f"{bm_safe}  {bm_result['time_ms']:.1f}ms  "
              f"({bm_result['num_submodules']} submod)")

        # ── 4. LAYER_BOUNDARY verification (comparison) ────────────────
        print("  [4] LAYER_BOUNDARY verification...", end=" ", flush=True)
        lb_result = run_verification(source, input_shapes, "layer_boundary", "layer_boundary")
        record["layer_boundary_verification"] = lb_result
        lb_safe = "SAFE" if lb_result["safe"] else (
            "UNSAFE" if lb_result["safe"] is False else "ERROR"
        )
        print(f"{lb_safe}  {lb_result['time_ms']:.1f}ms  "
              f"({lb_result['num_submodules']} submod)")

        # ── 5. DAG compositional verification ──────────────────────────
        print("  [5] DAG compositional verification...", end=" ", flush=True)
        dag_result = run_dag_verification(source, input_shapes, "dag_branch_merge")
        record["dag_verification"] = dag_result
        dag_safe = "SAFE" if dag_result["safe"] else (
            "UNSAFE" if dag_result["safe"] is False else "ERROR"
        )
        print(f"{dag_safe}  {dag_result['time_ms']:.1f}ms  "
              f"({dag_result['num_submodules']} submod)")

        # ── 6. DAG decomposition analysis ──────────────────────────────
        print("  [6] DAG decomposition analysis...", end=" ", flush=True)
        dag_decomp = analyze_dag_decomposition(source, input_shapes)
        record["dag_decomposition"] = dag_decomp
        print(f"topology={dag_decomp['topology']}  "
              f"{dag_decomp['num_submodules']} submod  "
              f"edges={dag_decomp.get('dag_edges', [])}")

        # ── 7. Agreement & speedup summary ─────────────────────────────
        agree_bm = (mono["safe"] == bm_result["safe"]) if bm_result["safe"] is not None else False
        agree_lb = (mono["safe"] == lb_result["safe"]) if lb_result["safe"] is not None else False
        agree_dag = (mono["safe"] == dag_result["safe"]) if dag_result["safe"] is not None else False

        bm_speedup = (mono["time_ms"] / bm_result["time_ms"]) if bm_result["time_ms"] > 0 else 0
        lb_speedup = (mono["time_ms"] / lb_result["time_ms"]) if lb_result["time_ms"] > 0 else 0
        dag_speedup = (mono["time_ms"] / dag_result["time_ms"]) if dag_result["time_ms"] > 0 else 0

        record["agreement"] = {
            "branch_merge_agrees_mono": agree_bm,
            "layer_boundary_agrees_mono": agree_lb,
            "dag_agrees_mono": agree_dag,
        }
        record["speedup"] = {
            "branch_merge_vs_mono": round(bm_speedup, 3),
            "layer_boundary_vs_mono": round(lb_speedup, 3),
            "dag_vs_mono": round(dag_speedup, 3),
        }

        # ── 8. Limitation analysis for BRANCH_MERGE ────────────────────
        limitations = []
        if bm_decomp.get("num_submodules", 0) <= 1:
            limitations.append(
                "BRANCH_MERGE found no branch/merge cut points – model "
                "decomposed as a single block. The strategy requires "
                "explicit multi-consumer outputs or multi-input steps in "
                "the linearized computation graph."
            )
        if bm_result.get("error"):
            limitations.append(f"Verification error: {bm_result['error']}")

        # Detect false-negative: BRANCH_MERGE says UNSAFE but monolithic says SAFE
        if mono["safe"] is True and bm_result.get("safe") is False:
            # Find which submodules reported errors
            failing_subs = [
                (sn, sr) for sn, sr in bm_result.get("submodule_results", {}).items()
                if not sr.get("safe", True)
            ]
            failing_desc = "; ".join(
                f"{sn}: {sr.get('errors', ['?'])[0] if sr.get('errors') else 'shape mismatch'}"
                for sn, sr in failing_subs
            ) if failing_subs else "interface incompatibility"
            limitations.append(
                f"FALSE NEGATIVE: BRANCH_MERGE reports UNSAFE on a "
                f"monolithically-SAFE model. The linear cut-point "
                f"decomposition severs cross-path data dependencies, "
                f"causing spurious shape mismatches at merge points "
                f"where tensors from different branches recombine. "
                f"Failing submodules: {failing_desc}"
            )
            # Check if DAG verification handles it correctly
            if dag_result.get("safe") is True:
                limitations.append(
                    "DAG compositional verification (verify_compositional_dag) "
                    "correctly verifies this architecture by preserving "
                    "cross-path edges in the decomposition graph, "
                    "demonstrating that the limitation is specific to "
                    "BRANCH_MERGE's linear decomposition, not the "
                    "assume-guarantee framework itself."
                )

        # Detect false-positive: BRANCH_MERGE says SAFE but monolithic says UNSAFE
        if mono["safe"] is False and bm_result.get("safe") is True:
            limitations.append(
                "FALSE POSITIVE: BRANCH_MERGE reports SAFE on a "
                "monolithically-UNSAFE model. This indicates unsoundness "
                "in the decomposition — the linear partition may miss "
                "constraints that span branch boundaries."
            )

        incompatible_interfaces = [
            ic for ic in bm_result.get("interface_checks", [])
            if not ic.get("compatible", True)
        ]
        if incompatible_interfaces:
            for ic in incompatible_interfaces:
                limitations.append(
                    f"Interface incompatibility {ic['producer']}→{ic['consumer']}: "
                    f"{ic['message']}. BRANCH_MERGE linear decomposition cannot "
                    f"express cross-path shape constraints for non-sequential data flow."
                )

        # Check for wildcard contracts at merge points (lost precision)
        wildcard_inputs = []
        for c in bm_decomp.get("contracts", []):
            for tname, tshape in c["input_contract"]["input_shapes"].items():
                if tshape == ["*"]:
                    wildcard_inputs.append((c["submodule"], tname))
        if wildcard_inputs:
            names = ", ".join(f"{sm}.{t}" for sm, t in wildcard_inputs)
            limitations.append(
                f"PRECISION LOSS: {len(wildcard_inputs)} input contract(s) "
                f"use wildcard shapes (unconstrained) at merge/skip "
                f"boundaries: {names}. BRANCH_MERGE cannot derive "
                f"precise shape contracts for tensors that cross "
                f"branch boundaries because the producer may be in "
                f"a different linear partition."
            )

        if category in ("encoder_decoder", "parallel") and bm_decomp.get("num_submodules", 0) > 1:
            dag_subs = dag_decomp.get("num_submodules", 0)
            bm_subs = bm_decomp.get("num_submodules", 0)
            if dag_subs != bm_subs:
                limitations.append(
                    f"DAG decomposition found {dag_subs} submodules vs "
                    f"BRANCH_MERGE's {bm_subs}. DAG-aware decomposition "
                    f"captures cross-path dependencies that the linear "
                    f"BRANCH_MERGE cut-point approach misses."
                )

        if not limitations:
            limitations.append("No limitations detected for this architecture.")

        record["branch_merge_limitations"] = limitations

        # ── Print interface contracts ──────────────────────────────────
        if bm_decomp.get("contracts"):
            print(f"  InterfaceContracts (BRANCH_MERGE):")
            for c in bm_decomp["contracts"]:
                ic = c["input_contract"]
                oc = c["output_contract"]
                print(f"    {c['submodule']} [{c['step_range'][0]}-{c['step_range'][1]}]:")
                if ic["input_shapes"]:
                    print(f"      ASSUME  inputs:  {ic['input_shapes']}")
                if ic["constraints"]:
                    for cn in ic["constraints"]:
                        print(f"      ASSUME  {cn}")
                if oc["output_shapes"]:
                    print(f"      GUARANTEE outputs: {oc['output_shapes']}")
                if oc["constraints"]:
                    for cn in oc["constraints"]:
                        print(f"      GUARANTEE {cn}")

        if limitations and limitations[0] != "No limitations detected for this architecture.":
            print(f"  ⚠ Limitations:")
            for lim in limitations:
                print(f"    • {lim}")

        print(f"  Summary: BM={bm_safe}({bm_speedup:.2f}x) "
              f"LB={lb_safe}({lb_speedup:.2f}x) "
              f"DAG={dag_safe}({dag_speedup:.2f}x) "
              f"agree=[BM:{agree_bm},LB:{agree_lb},DAG:{agree_dag}]")

        all_results.append(record)

    # ── Save results ────────────────────────────────────────────────────
    out_path = os.path.join(
        os.path.dirname(__file__), "results",
        "assume_guarantee_nonsequential_results.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'═' * 95}")
    print(f"Results saved to {out_path}")

    # ── Final summary table ─────────────────────────────────────────────
    print(f"\n{'═' * 95}")
    print("SUMMARY")
    print(f"{'═' * 95}")
    print(f"{'Model':<25} {'Cat':<16} {'Mono':>6} "
          f"{'BM':>6} {'LB':>6} {'DAG':>6} "
          f"{'BM-sp':>7} {'DAG-sp':>7} {'Topo':<15}")
    print("─" * 95)

    for r in all_results:
        mono_s = "SAFE" if r["monolithic"]["safe"] else "FAIL"
        bm_s = "SAFE" if r["branch_merge_verification"]["safe"] else "FAIL"
        lb_s = "SAFE" if r["layer_boundary_verification"]["safe"] else "FAIL"
        dag_s = "SAFE" if r["dag_verification"]["safe"] else "FAIL"
        bm_sp = r["speedup"]["branch_merge_vs_mono"]
        dag_sp = r["speedup"]["dag_vs_mono"]
        topo = r["dag_decomposition"].get("topology", "?")
        print(f"{r['model_name']:<25} {r['category']:<16} {mono_s:>6} "
              f"{bm_s:>6} {lb_s:>6} {dag_s:>6} "
              f"{bm_sp:>6.2f}x {dag_sp:>6.2f}x {topo:<15}")

    # Limitation summary
    print(f"\n{'═' * 95}")
    print("BRANCH_MERGE LIMITATION ANALYSIS")
    print(f"{'═' * 95}")
    for r in all_results:
        print(f"\n{r['model_name']} ({r['category']}):")
        for lim in r["branch_merge_limitations"]:
            print(f"  • {lim}")


if __name__ == "__main__":
    run_experiment()
