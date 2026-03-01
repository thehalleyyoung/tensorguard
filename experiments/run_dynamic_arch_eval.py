#!/usr/bin/env python3
"""
Dynamic Architecture Evaluation Experiment for TensorGuard.

Evaluates TensorGuard on modern dynamic-shape architectures:
1. Transformer with dynamic attention masks
2. Mixture of Experts (MoE) with gating/routing
3. Graph Neural Networks (GNN) with variable nodes/edges
4. Dynamic sequence models (LSTM/GRU with variable-length sequences)
5. Conditional computation (data-dependent branching)

Addresses HIGH priority critique: "Evaluate on transformer models with
dynamic attention masks, MoE architectures, or GNNs to assess performance
on modern dynamic-shape architectures."
"""

import json
import os
import sys
import time

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from src.model_checker import verify_model

# ═══════════════════════════════════════════════════════════════════════
# Benchmark definitions
# ═══════════════════════════════════════════════════════════════════════

DYNAMIC_ARCH_BENCHMARKS = [
    # ── 1. Transformer with dynamic attention mask ─────────────────
    {
        "name": "transformer_dynamic_mask_safe",
        "category": "transformer",
        "description": "Transformer encoder with dynamic attention mask, correct shapes",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes", "dynamic_control_flow"],
        "source": '''
import torch
import torch.nn as nn

class DynamicMaskTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.pos_encoding = nn.Embedding(512, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(512, 10000)

    def forward(self, x, mask):
        emb = self.embedding(x)
        out = self.transformer(emb, src_key_padding_mask=mask)
        return self.fc(out)
''',
        "input_shapes": {"x": ("seq_len", "batch"), "mask": ("batch", "seq_len")},
    },
    {
        "name": "transformer_dynamic_mask_dim_bug",
        "category": "transformer",
        "description": "Transformer with Linear dim mismatch after encoder",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch
import torch.nn as nn

class DynamicMaskTransformerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(256, 10000)  # BUG: expects 512 from transformer

    def forward(self, x, mask):
        emb = self.embedding(x)
        out = self.transformer(emb, src_key_padding_mask=mask)
        return self.fc(out)
''',
        "input_shapes": {"x": ("seq_len", "batch"), "mask": ("batch", "seq_len")},
    },
    {
        "name": "multihead_attention_safe",
        "category": "transformer",
        "description": "Multi-head attention with dynamic key/value lengths",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class DynamicMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc = nn.Linear(256, 128)

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        out, _ = self.attn(q, k, v)
        return self.fc(out)
''',
        "input_shapes": {
            "query": ("q_len", "batch", 256),
            "key": ("kv_len", "batch", 256),
            "value": ("kv_len", "batch", 256),
        },
    },
    {
        "name": "multihead_attention_embed_bug",
        "category": "transformer",
        "description": "MHA with mismatched embed dim between projection and attention",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class DynamicMHABug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 128)  # BUG: projects to 128, but attn expects 256
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc = nn.Linear(256, 64)

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        out, _ = self.attn(q, k, v)
        return self.fc(out)
''',
        "input_shapes": {
            "query": ("q_len", "batch", 256),
            "key": ("kv_len", "batch", 256),
            "value": ("kv_len", "batch", 256),
        },
    },

    # ── 2. Mixture of Experts (MoE) ───────────────────────────────
    {
        "name": "moe_gating_safe",
        "category": "moe",
        "description": "MoE with gating network routing to expert MLPs, correct dims",
        "expected_safe": True,
        "dynamic_features": ["dynamic_control_flow", "data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class MixtureOfExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(256, 4)
        self.expert1 = nn.Linear(256, 128)
        self.expert2 = nn.Linear(256, 128)
        self.expert3 = nn.Linear(256, 128)
        self.expert4 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        weights = self.gate(x)
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        e3 = self.expert3(x)
        e4 = self.expert4(x)
        out = e1 + e2 + e3 + e4
        return self.output(out)
''',
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "moe_expert_dim_bug",
        "category": "moe",
        "description": "MoE where one expert has wrong output dim",
        "expected_safe": False,
        "dynamic_features": ["dynamic_control_flow", "data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class MoEBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(256, 4)
        self.expert1 = nn.Linear(256, 128)
        self.expert2 = nn.Linear(256, 64)   # BUG: output 64 != 128
        self.expert3 = nn.Linear(256, 128)
        self.expert4 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        weights = self.gate(x)
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        e3 = self.expert3(x)
        e4 = self.expert4(x)
        out = e1 + e2 + e3 + e4
        return self.output(out)
''',
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "moe_hierarchical_safe",
        "category": "moe",
        "description": "Hierarchical MoE with two-level gating",
        "expected_safe": True,
        "dynamic_features": ["dynamic_control_flow"],
        "source": '''
import torch.nn as nn

class HierarchicalMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_gate = nn.Linear(512, 2)
        self.sub_gate1 = nn.Linear(512, 2)
        self.sub_gate2 = nn.Linear(512, 2)
        self.expert_a1 = nn.Linear(512, 256)
        self.expert_a2 = nn.Linear(512, 256)
        self.expert_b1 = nn.Linear(512, 256)
        self.expert_b2 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        a1 = self.expert_a1(x)
        a2 = self.expert_a2(x)
        b1 = self.expert_b1(x)
        b2 = self.expert_b2(x)
        combined = a1 + a2 + b1 + b2
        return self.fc(combined)
''',
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "moe_gate_input_bug",
        "category": "moe",
        "description": "MoE gate input dim doesn't match input feature dim",
        "expected_safe": False,
        "dynamic_features": ["dynamic_control_flow"],
        "source": '''
import torch.nn as nn

class MoEGateBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(128, 3)  # BUG: expects 128, input is 256
        self.expert1 = nn.Linear(256, 64)
        self.expert2 = nn.Linear(256, 64)
        self.expert3 = nn.Linear(256, 64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        g = self.gate(x)
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        e3 = self.expert3(x)
        out = e1 + e2 + e3
        return self.fc(out)
''',
        "input_shapes": {"x": ("batch", 256)},
    },

    # ── 3. Graph Neural Network (GNN) ─────────────────────────────
    {
        "name": "gnn_message_passing_safe",
        "category": "gnn",
        "description": "GNN with message passing: node features -> MLP -> aggregate",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes", "dynamic_control_flow"],
        "source": '''
import torch.nn as nn

class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_fn = nn.Linear(64, 128)
        self.update_fn = nn.Linear(128, 64)
        self.fc = nn.Linear(64, 10)

    def forward(self, node_features):
        messages = self.message_fn(node_features)
        updated = self.update_fn(messages)
        return self.fc(updated)
''',
        "input_shapes": {"node_features": ("num_nodes", 64)},
    },
    {
        "name": "gnn_message_passing_bug",
        "category": "gnn",
        "description": "GNN with dim mismatch between message and update functions",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class GNNLayerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_fn = nn.Linear(64, 128)
        self.update_fn = nn.Linear(64, 64)  # BUG: expects 128 from message_fn
        self.fc = nn.Linear(64, 10)

    def forward(self, node_features):
        messages = self.message_fn(node_features)
        updated = self.update_fn(messages)
        return self.fc(updated)
''',
        "input_shapes": {"node_features": ("num_nodes", 64)},
    },
    {
        "name": "gnn_multi_layer_safe",
        "category": "gnn",
        "description": "Multi-layer GNN with residual connections",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class MultiLayerGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(32, 64)
        self.conv2 = nn.Linear(64, 64)
        self.conv3 = nn.Linear(64, 64)
        self.classifier = nn.Linear(64, 5)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        return self.classifier(h)
''',
        "input_shapes": {"x": ("num_nodes", 32)},
    },
    {
        "name": "gnn_edge_conditioned_bug",
        "category": "gnn",
        "description": "Edge-conditioned GNN with feature dim mismatch",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class EdgeConditionedGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Linear(16, 64)
        self.edge_encoder = nn.Linear(8, 32)
        self.combine = nn.Linear(128, 64)  # BUG: should be 96 (64+32)
        self.classifier = nn.Linear(64, 3)

    def forward(self, node_feat, edge_feat):
        n = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)
        return self.classifier(n)
''',
        "input_shapes": {
            "node_feat": ("num_nodes", 16),
            "edge_feat": ("num_edges", 8),
        },
    },

    # ── 4. Dynamic sequence model ─────────────────────────────────
    {
        "name": "dynamic_lstm_varlen_safe",
        "category": "dynamic_sequence",
        "description": "LSTM with variable-length sequences, correct shapes",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class VarLenLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5000, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.lstm(emb)
        return self.fc(out)
''',
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    {
        "name": "dynamic_lstm_varlen_bug",
        "category": "dynamic_sequence",
        "description": "LSTM with hidden-to-linear dim mismatch",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class VarLenLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5000, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(128, 10)  # BUG: should be 256

    def forward(self, x):
        emb = self.embedding(x)
        out = self.lstm(emb)
        return self.fc(out)
''',
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    {
        "name": "dynamic_bigru_varlen_safe",
        "category": "dynamic_sequence",
        "description": "Bidirectional GRU with variable-length input",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class VarLenBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 128, bidirectional=True)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''',
        "input_shapes": {"x": ("seq_len", "batch", 64)},
    },
    {
        "name": "dynamic_bigru_varlen_bug",
        "category": "dynamic_sequence",
        "description": "BiGRU forgetting to double hidden size for bidirectional",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class VarLenBiGRUBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 128, bidirectional=True)
        self.fc = nn.Linear(128, 10)  # BUG: should be 256

    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''',
        "input_shapes": {"x": ("seq_len", "batch", 64)},
    },

    # ── 5. Conditional computation ────────────────────────────────
    {
        "name": "conditional_branch_safe",
        "category": "conditional",
        "description": "Model with dual branch paths, both having correct shapes",
        "expected_safe": True,
        "dynamic_features": ["dynamic_control_flow", "data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(256, 128)
        self.branch_b = nn.Linear(256, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        out = a + b
        return self.fc(out)
''',
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "conditional_branch_dim_bug",
        "category": "conditional",
        "description": "Conditional branches produce different output dims",
        "expected_safe": False,
        "dynamic_features": ["dynamic_control_flow", "data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class ConditionalBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(256, 128)
        self.branch_b = nn.Linear(256, 64)   # BUG: 64 != 128
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        out = a + b
        return self.fc(out)
''',
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "adaptive_pooling_safe",
        "category": "conditional",
        "description": "Model with adaptive pooling handling variable spatial dims",
        "expected_safe": True,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class AdaptiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool(h)
        return self.fc(h)
''',
        "input_shapes": {"x": ("batch", 3, "height", "width")},
    },
    {
        "name": "skip_connection_dim_bug",
        "category": "conditional",
        "description": "Skip connection with channel mismatch",
        "expected_safe": False,
        "dynamic_features": ["data_dependent_shapes"],
        "source": '''
import torch.nn as nn

class SkipBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)  # BUG: 32 != 64 for skip
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        out = h1 + h2  # shape mismatch: 64 vs 32 channels
        return self.fc(out)
''',
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmarks():
    results = []
    tp = fp = tn = fn = 0
    category_stats = {}

    for bench in DYNAMIC_ARCH_BENCHMARKS:
        cat = bench["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        t0 = time.monotonic()
        error_msg = None
        try:
            vr = verify_model(bench["source"], input_shapes=bench["input_shapes"])
            elapsed = (time.monotonic() - t0) * 1000

            predicted_safe = vr.safe
            actual_safe = bench["expected_safe"]
            correct = predicted_safe == actual_safe

            if actual_safe and predicted_safe:
                tn += 1
            elif actual_safe and not predicted_safe:
                fp += 1
            elif not actual_safe and not predicted_safe:
                tp += 1
            else:
                fn += 1

            if correct:
                category_stats[cat]["correct"] += 1

            # Collect dynamic features info
            dyn_feats = {}
            dyn_warnings = []
            try:
                dyn_feats = {k: v for k, v in vr.dynamic_features.items()
                             if not isinstance(v, (list, dict))}
                dyn_warnings = vr.dynamic_feature_warnings[:5]
            except Exception:
                pass

            results.append({
                "name": bench["name"],
                "category": cat,
                "description": bench["description"],
                "expected_safe": actual_safe,
                "verdict": "safe" if predicted_safe else "unsafe",
                "is_correct": correct,
                "verification_time_ms": round(elapsed, 2),
                "errors": vr.errors,
                "dynamic_features_detected": dyn_feats,
                "dynamic_feature_warnings": dyn_warnings,
                "expected_dynamic_features": bench.get("dynamic_features", []),
            })

        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            error_msg = str(e)
            results.append({
                "name": bench["name"],
                "category": cat,
                "description": bench["description"],
                "expected_safe": bench["expected_safe"],
                "verdict": "error",
                "is_correct": False,
                "verification_time_ms": round(elapsed, 2),
                "errors": [error_msg],
                "dynamic_features_detected": {},
                "dynamic_feature_warnings": [],
                "expected_dynamic_features": bench.get("dynamic_features", []),
            })

        marker = "✓" if results[-1]["is_correct"] else "✗"
        exp = "safe" if bench["expected_safe"] else "buggy"
        print(f"  {marker} [{cat}] {bench['name']}: "
              f"verdict={results[-1]['verdict']} (expected {exp}) "
              f"[{elapsed:.1f}ms]"
              + (f"  ERROR: {error_msg}" if error_msg else ""))

    total = len(results)
    correct_count = tp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    accuracy = correct_count / total if total > 0 else 0

    summary = {
        "total_benchmarks": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "errors": len([r for r in results if r["verdict"] == "error"]),
        "category_breakdown": {
            cat: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0,
            }
            for cat, s in category_stats.items()
        },
    }

    return {
        "experiment": "dynamic_architecture_evaluation",
        "description": (
            "Evaluates TensorGuard on modern dynamic-shape architectures: "
            "Transformer w/ dynamic attention masks, MoE, GNN, "
            "dynamic sequence models, and conditional computation."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "benchmarks": results,
    }


def main():
    print("=" * 70)
    print("  Dynamic Architecture Evaluation — TensorGuard")
    print("=" * 70)
    print()

    output = run_benchmarks()
    s = output["summary"]

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total benchmarks : {s['total_benchmarks']}")
    print(f"  Correct verdicts : {s['correct']}/{s['total_benchmarks']} "
          f"(accuracy={s['accuracy']*100:.1f}%)")
    print(f"  Precision        : {s['precision']:.4f}")
    print(f"  Recall           : {s['recall']:.4f}")
    print(f"  F1               : {s['f1']:.4f}")
    print(f"  TP={s['tp']}  FP={s['fp']}  TN={s['tn']}  FN={s['fn']}")
    print(f"  Errors           : {s['errors']}")
    print()
    print("  Per-category breakdown:")
    for cat, cs in s["category_breakdown"].items():
        print(f"    {cat:25s}: {cs['correct']}/{cs['total']} "
              f"(accuracy={cs['accuracy']*100:.1f}%)")

    out_path = os.path.join(IMPL_ROOT, "experiments",
                            "dynamic_arch_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
