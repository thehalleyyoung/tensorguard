"""
Stratified benchmark evaluation with confidence intervals.

Evaluates TensorGuard across diverse architecture types with:
  - Difficulty-tier stratification (easy/medium/hard)
  - Architecture-family stratification (MLP/CNN/RNN/Transformer/MoE/Dynamic)
  - Bootstrap confidence intervals for all metrics
  - Bug-type stratification (shape/device/phase/broadcast)
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_checker import verify_model, VerificationResult


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Benchmark:
    name: str
    source: str
    input_shapes: Dict[str, tuple]
    expected_safe: bool
    architecture: str    # MLP, CNN, RNN, Transformer, MoE, Dynamic, Mixed
    difficulty: str      # easy, medium, hard
    bug_type: Optional[str] = None  # shape, device, phase, broadcast, None if safe


BENCHMARKS: List[Benchmark] = []

# ─── Easy: Simple MLPs ───────────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_mlp_2layer",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
    input_shapes={"x": ("batch", 784)},
    expected_safe=True,
    architecture="MLP",
    difficulty="easy",
))

BENCHMARKS.append(Benchmark(
    name="buggy_mlp_dim_mismatch",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
''',
    input_shapes={"x": ("batch", 784)},
    expected_safe=False,
    architecture="MLP",
    difficulty="easy",
    bug_type="shape",
))

BENCHMARKS.append(Benchmark(
    name="safe_mlp_3layer",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
''',
    input_shapes={"x": ("batch", 512)},
    expected_safe=True,
    architecture="MLP",
    difficulty="easy",
))

BENCHMARKS.append(Benchmark(
    name="buggy_mlp_wrong_input",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
    input_shapes={"x": ("batch", 200)},
    expected_safe=False,
    architecture="MLP",
    difficulty="easy",
    bug_type="shape",
))

# ─── Easy: Simple CNNs ──────────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_cnn_simple",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
''',
    input_shapes={"x": ("batch", 3, 32, 32)},
    expected_safe=True,
    architecture="CNN",
    difficulty="easy",
))

BENCHMARKS.append(Benchmark(
    name="buggy_cnn_channel_mismatch",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
''',
    input_shapes={"x": ("batch", 3, 32, 32)},
    expected_safe=False,
    architecture="CNN",
    difficulty="easy",
    bug_type="shape",
))

# ─── Medium: RNNs ──────────────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_lstm_basic",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
''',
    input_shapes={"x": ("batch", "seq_len", 128)},
    expected_safe=True,
    architecture="RNN",
    difficulty="medium",
))

BENCHMARKS.append(Benchmark(
    name="buggy_lstm_hidden_mismatch",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
''',
    input_shapes={"x": ("batch", "seq_len", 128)},
    expected_safe=False,
    architecture="RNN",
    difficulty="medium",
    bug_type="shape",
))

BENCHMARKS.append(Benchmark(
    name="safe_gru_basic",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 5)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)
''',
    input_shapes={"x": ("batch", "seq_len", 64)},
    expected_safe=True,
    architecture="RNN",
    difficulty="medium",
))

# ─── Medium: Transformers ───────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_transformer_encoder",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.fc = nn.Linear(512, 10000)
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder_layer(x)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", "seq_len")},
    expected_safe=True,
    architecture="Transformer",
    difficulty="medium",
))

BENCHMARKS.append(Benchmark(
    name="buggy_transformer_dim_mismatch",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 256)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.fc = nn.Linear(512, 10000)
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder_layer(x)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", "seq_len")},
    expected_safe=False,
    architecture="Transformer",
    difficulty="medium",
    bug_type="shape",
))

BENCHMARKS.append(Benchmark(
    name="safe_transformer_decoder",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5000, 256)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=4)
        self.fc = nn.Linear(256, 5000)
    def forward(self, x):
        x = self.embedding(x)
        x = self.decoder_layer(x, x)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", "seq_len")},
    expected_safe=True,
    architecture="Transformer",
    difficulty="medium",
))

# ─── Hard: ResNets with skip connections ─────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_resblock",
    source='''
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual
''',
    input_shapes={"x": ("batch", 64, 32, 32)},
    expected_safe=True,
    architecture="CNN",
    difficulty="hard",
))

BENCHMARKS.append(Benchmark(
    name="buggy_resblock_channel_mismatch",
    source='''
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual
''',
    input_shapes={"x": ("batch", 64, 32, 32)},
    expected_safe=False,
    architecture="CNN",
    difficulty="hard",
    bug_type="broadcast",
))

# ─── Hard: Multi-input models ──────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_multihead_attn",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.fc(attn_out)
''',
    input_shapes={"x": ("seq_len", "batch", 512)},
    expected_safe=True,
    architecture="Transformer",
    difficulty="hard",
))

# ─── Hard: Encoder-Decoder ─────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_encoder_decoder",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 128)
        self.decoder = nn.Linear(128, 784)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
''',
    input_shapes={"x": ("batch", 784)},
    expected_safe=True,
    architecture="MLP",
    difficulty="medium",
))

# ─── Hard: MoE-style models ──────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_moe_gate",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(512, 8)
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        gate_logits = self.gate(x)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", 512)},
    expected_safe=True,
    architecture="MoE",
    difficulty="hard",
))

BENCHMARKS.append(Benchmark(
    name="buggy_moe_expert_dim",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(512, 8)
        self.expert = nn.Linear(256, 128)
    def forward(self, x):
        gate_logits = self.gate(x)
        return self.expert(x)
''',
    input_shapes={"x": ("batch", 512)},
    expected_safe=False,
    architecture="MoE",
    difficulty="hard",
    bug_type="shape",
))

# ─── Medium: Normalization variants ─────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_layernorm",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        x = self.ln(x)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", 512)},
    expected_safe=True,
    architecture="MLP",
    difficulty="medium",
))

BENCHMARKS.append(Benchmark(
    name="safe_groupnorm",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(8, 64)
        self.conv = nn.Conv2d(64, 128, 3, padding=1)
    def forward(self, x):
        x = self.gn(x)
        return self.conv(x)
''',
    input_shapes={"x": ("batch", 64, 32, 32)},
    expected_safe=True,
    architecture="CNN",
    difficulty="medium",
))

# ─── Hard: Deep networks ────────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_deep_mlp_5layer",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
    input_shapes={"x": ("batch", 1024)},
    expected_safe=True,
    architecture="MLP",
    difficulty="hard",
))

BENCHMARKS.append(Benchmark(
    name="buggy_deep_mlp_layer3",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
    input_shapes={"x": ("batch", 1024)},
    expected_safe=False,
    architecture="MLP",
    difficulty="hard",
    bug_type="shape",
))

# ─── Medium: Flatten + FC ──────────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_conv_flatten_fc",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", 3, 32, 32)},
    expected_safe=True,
    architecture="CNN",
    difficulty="medium",
))

# ─── Easy: Identity / Dropout ──────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_dropout_chain",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        return self.fc2(x)
''',
    input_shapes={"x": ("batch", 100)},
    expected_safe=True,
    architecture="MLP",
    difficulty="easy",
))

# ─── Hard: Bidirectional LSTM ──────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_bilstm",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
''',
    input_shapes={"x": ("batch", "seq_len", 64)},
    expected_safe=True,
    architecture="RNN",
    difficulty="hard",
))

BENCHMARKS.append(Benchmark(
    name="buggy_bilstm_fc_dim",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
''',
    input_shapes={"x": ("batch", "seq_len", 64)},
    expected_safe=False,
    architecture="RNN",
    difficulty="hard",
    bug_type="shape",
))

# ─── Medium: Embedding models ─────────────────────────────────────────

BENCHMARKS.append(Benchmark(
    name="safe_embedding_fc",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 128)
        self.fc = nn.Linear(128, 50)
    def forward(self, x):
        x = self.emb(x)
        return self.fc(x)
''',
    input_shapes={"x": ("batch", "seq_len")},
    expected_safe=True,
    architecture="Transformer",
    difficulty="medium",
))

# Add more to reach a good count
BENCHMARKS.append(Benchmark(
    name="safe_conv1d",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.conv(x)
        return x
''',
    input_shapes={"x": ("batch", 16, 100)},
    expected_safe=True,
    architecture="CNN",
    difficulty="easy",
))

BENCHMARKS.append(Benchmark(
    name="safe_sequential",
    source='''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
    def forward(self, x):
        return self.net(x)
''',
    input_shapes={"x": ("batch", 100)},
    expected_safe=True,
    architecture="MLP",
    difficulty="easy",
))


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    name: str
    architecture: str
    difficulty: str
    expected_safe: bool
    actual_safe: Optional[bool]
    correct: bool
    time_ms: float
    bug_type: Optional[str] = None
    error: Optional[str] = None


def run_benchmark(b: Benchmark) -> BenchmarkResult:
    """Run a single benchmark and return result."""
    t0 = time.time()
    try:
        result = verify_model(b.source, input_shapes=b.input_shapes)
        elapsed = (time.time() - t0) * 1000
        actual_safe = result.safe
        correct = (actual_safe == b.expected_safe)
        return BenchmarkResult(
            name=b.name,
            architecture=b.architecture,
            difficulty=b.difficulty,
            expected_safe=b.expected_safe,
            actual_safe=actual_safe,
            correct=correct,
            time_ms=elapsed,
            bug_type=b.bug_type,
        )
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return BenchmarkResult(
            name=b.name,
            architecture=b.architecture,
            difficulty=b.difficulty,
            expected_safe=b.expected_safe,
            actual_safe=None,
            correct=False,
            time_ms=elapsed,
            bug_type=b.bug_type,
            error=str(e),
        )


def compute_metrics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute precision, recall, F1 from results."""
    tp = fp = fn = tn = 0
    for r in results:
        if r.actual_safe is None:
            continue
        if r.expected_safe and r.actual_safe:
            tn += 1  # correctly identified as safe
        elif r.expected_safe and not r.actual_safe:
            fp += 1  # false alarm
        elif not r.expected_safe and not r.actual_safe:
            tp += 1  # correctly caught bug
        elif not r.expected_safe and r.actual_safe:
            fn += 1  # missed bug

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    times = [r.time_ms for r in results if r.actual_safe is not None]
    avg_time = sum(times) / len(times) if times else 0

    return {
        "n": len(results),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / max(tp + fp + fn + tn, 1), 4),
        "avg_time_ms": round(avg_time, 2),
        "errors": sum(1 for r in results if r.error is not None),
    }


def bootstrap_ci(
    results: List[BenchmarkResult],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap confidence intervals for metrics."""
    random.seed(42)
    f1_samples = []
    precision_samples = []
    recall_samples = []

    for _ in range(n_bootstrap):
        sample = random.choices(results, k=len(results))
        metrics = compute_metrics(sample)
        f1_samples.append(metrics["f1"])
        precision_samples.append(metrics["precision"])
        recall_samples.append(metrics["recall"])

    alpha = (1 - confidence) / 2
    low_idx = int(alpha * n_bootstrap)
    high_idx = int((1 - alpha) * n_bootstrap)

    f1_sorted = sorted(f1_samples)
    prec_sorted = sorted(precision_samples)
    rec_sorted = sorted(recall_samples)

    return {
        "f1_ci": (round(f1_sorted[low_idx], 4), round(f1_sorted[high_idx], 4)),
        "precision_ci": (round(prec_sorted[low_idx], 4), round(prec_sorted[high_idx], 4)),
        "recall_ci": (round(rec_sorted[low_idx], 4), round(rec_sorted[high_idx], 4)),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def stratify_results(
    results: List[BenchmarkResult],
) -> Dict[str, Any]:
    """Stratify results by difficulty, architecture, and bug type."""
    stratified = {}

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.difficulty == diff]
        if subset:
            metrics = compute_metrics(subset)
            ci = bootstrap_ci(subset)
            stratified[f"difficulty_{diff}"] = {**metrics, **ci}

    # By architecture
    for arch in ["MLP", "CNN", "RNN", "Transformer", "MoE", "Dynamic", "Mixed"]:
        subset = [r for r in results if r.architecture == arch]
        if subset:
            metrics = compute_metrics(subset)
            ci = bootstrap_ci(subset)
            stratified[f"architecture_{arch}"] = {**metrics, **ci}

    # By bug type
    for bt in ["shape", "device", "phase", "broadcast"]:
        subset = [r for r in results if r.bug_type == bt]
        if subset:
            metrics = compute_metrics(subset)
            stratified[f"bug_type_{bt}"] = metrics

    return stratified


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and produce stratified results."""
    print(f"Running {len(BENCHMARKS)} benchmarks...")
    results = []
    for i, b in enumerate(BENCHMARKS):
        r = run_benchmark(b)
        status = "✓" if r.correct else "✗"
        print(f"  [{i+1}/{len(BENCHMARKS)}] {status} {b.name} ({r.time_ms:.1f}ms)")
        results.append(r)

    overall = compute_metrics(results)
    ci = bootstrap_ci(results)
    stratified = stratify_results(results)

    output = {
        "overall": {**overall, **ci},
        "stratified": stratified,
        "per_benchmark": [
            {
                "name": r.name,
                "architecture": r.architecture,
                "difficulty": r.difficulty,
                "expected": r.expected_safe,
                "actual": r.actual_safe,
                "correct": r.correct,
                "time_ms": round(r.time_ms, 2),
                "bug_type": r.bug_type,
                "error": r.error,
            }
            for r in results
        ],
    }

    return output


if __name__ == "__main__":
    output = run_all_benchmarks()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "stratified_benchmark_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Overall F1: {output['overall']['f1']}")
    print(f"Overall F1 95% CI: {output['overall']['f1_ci']}")
    print(f"Overall accuracy: {output['overall']['accuracy']}")
