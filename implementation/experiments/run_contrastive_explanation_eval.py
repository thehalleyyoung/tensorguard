"""
Contrastive explanation evaluation: runs contrastive + narrative explanations
on 10+ buggy benchmarks and compares with raw CEGAR trace output.

Measures:
  - Explanation length (characters / lines)
  - Number of foils generated per bug
  - Constraint isolation accuracy (did Craig interpolation contribute?)
  - Raw CEGAR trace length for comparison

Usage:
    cd implementation && python -m experiments.run_contrastive_explanation_eval
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure the implementation directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cegar_explanation import explain_verification, generate_explanation
from src.contrastive_explanation import (
    ContrastiveExplainer,
    ExplanationCalibrator,
    NarrativeGenerator,
    explain_contrastively,
)
from src.shape_cegar import run_shape_cegar, ShapeCEGARResult

try:
    from src.model_checker import extract_computation_graph
except ImportError:
    extract_computation_graph = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════════
# Buggy benchmark models (10+)
# ═══════════════════════════════════════════════════════════════════════════════

BUGGY_BENCHMARKS = [
    {
        "name": "linear_dim_mismatch",
        "description": "fc1 outputs 256 but fc2 expects 128",
        "source": """\
import torch.nn as nn

class LinearDimMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "conv_channel_mismatch",
        "description": "conv1 outputs 64 channels but conv2 expects 32",
        "source": """\
import torch.nn as nn

class ConvChannelMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "batchnorm_mismatch",
        "description": "conv1 outputs 64 channels but BatchNorm2d expects 32",
        "source": """\
import torch.nn as nn

class BatchNormMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "residual_shape_mismatch",
        "description": "Residual add: x has 128 channels but residual has 64",
        "source": """\
import torch.nn as nn

class ResidualShapeMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x
""",
        "input_shapes": {"x": ("batch", 64, 16, 16)},
    },
    {
        "name": "lstm_linear_mismatch",
        "description": "LSTM hidden_size is 128 but fc expects 64",
        "source": """\
import torch.nn as nn

class LSTMLinearMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = self.fc(output[:, -1, :])
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len", 256)},
    },
    {
        "name": "encoder_decoder_mismatch",
        "description": "Encoder outputs 64-dim but Decoder expects 32-dim",
        "source": """\
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(32, 256)
        self.dec2 = nn.Linear(256, 784)

    def forward(self, x):
        x = torch.relu(self.enc1(x))
        x = self.enc2(x)
        x = torch.relu(self.dec1(x))
        return self.dec2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "attention_proj_mismatch",
        "description": "W_o expects 256 but concatenated heads produce 512",
        "source": """\
import torch.nn as nn

class AttentionProjMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(512, 512)
        self.W_k = nn.Linear(512, 512)
        self.W_v = nn.Linear(512, 512)
        self.W_o = nn.Linear(256, 512)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        return self.W_o(v)
""",
        "input_shapes": {"x": ("batch", "seq_len", 512)},
    },
    {
        "name": "deep_chain_mismatch",
        "description": "fc3 expects 64 but fc2 outputs 128",
        "source": """\
import torch.nn as nn

class DeepChainMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "layernorm_mismatch",
        "description": "LayerNorm expects 256 features but Linear outputs 512",
        "source": """\
import torch.nn as nn

class LayerNormMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 512)
        self.ln = nn.LayerNorm(256)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "embedding_linear_mismatch",
        "description": "Embedding dim 128 but Linear expects 256",
        "source": """\
import torch.nn as nn

class EmbeddingLinearMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc = nn.Linear(256, 64)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    {
        "name": "double_linear_symbolic",
        "description": "Symbolic dim: fc expects 768 but input is symbolic",
        "source": """\
import torch.nn as nn

class DoubleLinearSymbolic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "triple_conv_mismatch",
        "description": "Third conv expects 64 in_channels but second outputs 128",
        "source": """\
import torch.nn as nn

class TripleConvMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 256, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation logic
# ═══════════════════════════════════════════════════════════════════════════════

def _run_single_benchmark(model_info: dict) -> dict:
    """Run CEGAR + contrastive explanation on a single model."""
    source = model_info["source"]
    input_shapes = model_info["input_shapes"]
    name = model_info["name"]

    result: dict = {
        "name": name,
        "description": model_info["description"],
    }

    # --- Run CEGAR ---
    t0 = time.monotonic()
    try:
        cegar_result = run_shape_cegar(
            source,
            input_shapes=input_shapes,
            max_iterations=10,
        )
    except Exception as e:
        result["error"] = f"CEGAR failed: {e}"
        result["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result
    cegar_ms = (time.monotonic() - t0) * 1000

    # --- Extract graph ---
    graph = None
    if extract_computation_graph is not None:
        try:
            graph = extract_computation_graph(source)
        except (ValueError, SyntaxError):
            pass

    # --- Raw CEGAR explanation (baseline) ---
    raw_explanation = generate_explanation(cegar_result, graph=graph, model_name=name)
    raw_text = raw_explanation.render()

    # --- Contrastive explanation ---
    t1 = time.monotonic()
    calibrator = ExplanationCalibrator()
    contrastive_result = explain_contrastively(
        cegar_result,
        graph=graph,
        model_name=name,
        calibrator=calibrator,
    )
    contrastive_ms = (time.monotonic() - t1) * 1000

    # --- Narrative ---
    narrator = NarrativeGenerator(calibrator=calibrator)
    narrative = narrator.generate(cegar_result, graph=graph, model_name=name)
    narrative_text = narrative.render()

    # --- Metrics ---
    contrastive_entries = contrastive_result.get("contrastive", [])
    contrastive_texts = [e["full_text"] for e in contrastive_entries]
    contrastive_full = "\n".join(contrastive_texts)

    has_interpolant = any(
        e.get("interpolant") is not None for e in contrastive_entries
    )

    result.update({
        "verdict": cegar_result.verdict.name,
        "cegar_iterations": cegar_result.iterations,
        "num_predicates": len(cegar_result.discovered_predicates),
        "num_bugs": len(cegar_result.real_bugs),
        # Raw CEGAR metrics
        "raw_trace_length_chars": len(raw_text),
        "raw_trace_length_lines": raw_text.count("\n") + 1,
        # Contrastive metrics
        "num_foils": len(contrastive_entries),
        "contrastive_length_chars": len(contrastive_full),
        "contrastive_length_lines": contrastive_full.count("\n") + 1 if contrastive_full else 0,
        "has_interpolant_constraint": has_interpolant,
        # Narrative metrics
        "narrative_length_chars": len(narrative_text),
        "narrative_length_lines": narrative_text.count("\n") + 1,
        "narrative_has_root_cause": narrative.root_cause is not None,
        # Timing
        "cegar_ms": round(cegar_ms, 1),
        "contrastive_ms": round(contrastive_ms, 1),
        # Samples
        "raw_trace_sample": raw_text[:500],
        "contrastive_sample": contrastive_full[:500] if contrastive_full else "",
        "narrative_sample": narrative_text[:500],
    })

    return result


def main():
    print("=" * 70)
    print("Contrastive Explanation Evaluation")
    print(f"Benchmarks: {len(BUGGY_BENCHMARKS)}")
    print("=" * 70)

    results = []
    for i, model_info in enumerate(BUGGY_BENCHMARKS):
        print(f"\n[{i+1}/{len(BUGGY_BENCHMARKS)}] {model_info['name']}: ", end="")
        sys.stdout.flush()

        entry = _run_single_benchmark(model_info)
        results.append(entry)

        if "error" in entry:
            print(f"ERROR — {entry['error']}")
        else:
            print(
                f"{entry['verdict']} | "
                f"foils={entry['num_foils']} | "
                f"raw={entry['raw_trace_length_lines']}L / "
                f"contrastive={entry['contrastive_length_lines']}L / "
                f"narrative={entry['narrative_length_lines']}L"
            )

    # --- Aggregate summary ---
    successful = [r for r in results if "error" not in r]
    n = len(successful)

    print(f"\n{'=' * 70}")
    print(f"Summary ({n}/{len(results)} successful)")
    print(f"{'=' * 70}")

    if n > 0:
        avg_foils = sum(r["num_foils"] for r in successful) / n
        avg_raw_lines = sum(r["raw_trace_length_lines"] for r in successful) / n
        avg_contrastive_lines = sum(r["contrastive_length_lines"] for r in successful) / n
        avg_narrative_lines = sum(r["narrative_length_lines"] for r in successful) / n
        interpolant_count = sum(1 for r in successful if r["has_interpolant_constraint"])
        root_cause_count = sum(1 for r in successful if r["narrative_has_root_cause"])

        print(f"  Average foils per model:          {avg_foils:.1f}")
        print(f"  Average raw trace lines:          {avg_raw_lines:.1f}")
        print(f"  Average contrastive expl. lines:  {avg_contrastive_lines:.1f}")
        print(f"  Average narrative lines:          {avg_narrative_lines:.1f}")
        print(f"  Models with interpolant isolat.:  {interpolant_count}/{n}")
        print(f"  Models with root cause narrat.:   {root_cause_count}/{n}")

        summary = {
            "total_benchmarks": len(results),
            "successful": n,
            "avg_foils_per_model": round(avg_foils, 2),
            "avg_raw_trace_lines": round(avg_raw_lines, 1),
            "avg_contrastive_lines": round(avg_contrastive_lines, 1),
            "avg_narrative_lines": round(avg_narrative_lines, 1),
            "interpolant_isolation_count": interpolant_count,
            "root_cause_narrative_count": root_cause_count,
        }
    else:
        summary = {"total_benchmarks": len(results), "successful": 0}

    # --- Save results ---
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "contrastive_explanation_results.json",
    )
    output = {
        "summary": summary,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
