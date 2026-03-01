"""LSTM/GRU shape verification experiment + expanded deep composition benchmark."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_checker import verify_model

# ────────────────────────────────────────────────────────────────────
# 1. LSTM/GRU Verification Experiment
# ────────────────────────────────────────────────────────────────────
LSTM_GRU_BENCHMARKS = [
    # Safe models
    {"name": "lstm_basic_safe", "expected": True, "category": "lstm",
     "source": '''
import torch.nn as nn
class LSTMBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "bilstm_safe", "expected": True, "category": "lstm",
     "source": '''
import torch.nn as nn
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128, bidirectional=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 64)}},

    {"name": "gru_basic_safe", "expected": True, "category": "gru",
     "source": '''
import torch.nn as nn
class GRUBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "bigru_safe", "expected": True, "category": "gru",
     "source": '''
import torch.nn as nn
class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 128, bidirectional=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 64)}},

    {"name": "emb_lstm_safe", "expected": True, "category": "lstm",
     "source": '''
import torch.nn as nn
class EmbLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 256)
        self.lstm = nn.LSTM(256, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch")}},

    {"name": "stacked_lstm_safe", "expected": True, "category": "lstm",
     "source": '''
import torch.nn as nn
class StackedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(128, 256)
        self.lstm2 = nn.LSTM(256, 64)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h1 = self.lstm1(x)
        h2 = self.lstm2(h1)
        return self.fc(h2)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "emb_gru_safe", "expected": True, "category": "gru",
     "source": '''
import torch.nn as nn
class EmbGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5000, 128)
        self.gru = nn.GRU(128, 64)
        self.fc = nn.Linear(64, 5)
    def forward(self, x):
        e = self.emb(x)
        out = self.gru(e)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch")}},

    {"name": "multilayer_lstm_safe", "expected": True, "category": "lstm",
     "source": '''
import torch.nn as nn
class MultiLayerLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, num_layers=3)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    # Buggy models
    {"name": "lstm_input_mismatch_bug", "expected": False, "category": "lstm",
     "source": '''
import torch.nn as nn
class LSTMInputBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 64)}},

    {"name": "lstm_linear_mismatch_bug", "expected": False, "category": "lstm",
     "source": '''
import torch.nn as nn
class LSTMLinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "bilstm_mismatch_bug", "expected": False, "category": "lstm",
     "source": '''
import torch.nn as nn
class BiLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, bidirectional=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.lstm(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "gru_input_mismatch_bug", "expected": False, "category": "gru",
     "source": '''
import torch.nn as nn
class GRUInputBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 64)}},

    {"name": "gru_linear_mismatch_bug", "expected": False, "category": "gru",
     "source": '''
import torch.nn as nn
class GRULinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "bigru_mismatch_bug", "expected": False, "category": "gru",
     "source": '''
import torch.nn as nn
class BiGRUBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256, bidirectional=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.gru(x)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "emb_lstm_mismatch_bug", "expected": False, "category": "lstm",
     "source": '''
import torch.nn as nn
class EmbLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 256)
        self.lstm = nn.LSTM(128, 64)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch")}},

    {"name": "stacked_lstm_mismatch_bug", "expected": False, "category": "lstm",
     "source": '''
import torch.nn as nn
class StackedLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(128, 256)
        self.lstm2 = nn.LSTM(128, 64)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h1 = self.lstm1(x)
        h2 = self.lstm2(h1)
        return self.fc(h2)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},
]


def run_lstm_gru_experiment():
    """Run LSTM/GRU verification experiment."""
    results = []
    tp = fp = tn = fn = 0

    for bench in LSTM_GRU_BENCHMARKS:
        t0 = time.time()
        result = verify_model(bench["source"], input_shapes=bench["input_shapes"])
        elapsed = (time.time() - t0) * 1000

        predicted_safe = result.safe
        actual_safe = bench["expected"]
        correct = predicted_safe == actual_safe

        if actual_safe and predicted_safe:
            tn += 1
        elif actual_safe and not predicted_safe:
            fp += 1
        elif not actual_safe and not predicted_safe:
            tp += 1
        else:
            fn += 1

        has_cert = result.certificate is not None if result.safe else False
        has_cex = result.counterexample is not None if not result.safe else False

        results.append({
            "name": bench["name"],
            "category": bench["category"],
            "expected_safe": actual_safe,
            "predicted_safe": predicted_safe,
            "correct": correct,
            "time_ms": round(elapsed, 1),
            "has_certificate": has_cert,
            "has_counterexample": has_cex,
        })
        print(f"  {'✓' if correct else '✗'} {bench['name']}: "
              f"expected={'safe' if actual_safe else 'bug'}, "
              f"got={'safe' if predicted_safe else 'bug'} [{elapsed:.1f}ms]")

    total = len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total

    summary = {
        "total_benchmarks": total,
        "correct": tp + tn,
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "lstm_benchmarks": len([r for r in results if r["category"] == "lstm"]),
        "gru_benchmarks": len([r for r in results if r["category"] == "gru"]),
        "certificates_issued": len([r for r in results if r["has_certificate"]]),
        "counterexamples_issued": len([r for r in results if r["has_counterexample"]]),
        "results": results,
    }

    return summary


# ────────────────────────────────────────────────────────────────────
# 2. Expanded Deep Composition: add LSTM-based sequence models
# ────────────────────────────────────────────────────────────────────
LSTM_DEEP_COMPOSITION = [
    {"name": "lstm-linear-chain-bug", "expected": False, "category": "lstm-chain",
     "description": "5-layer LSTM chain with hidden size mismatch at layer 4",
     "source": '''
import torch.nn as nn
class LSTMChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(128, 256)
        self.lstm2 = nn.LSTM(256, 512)
        self.lstm3 = nn.LSTM(512, 256)
        self.lstm4 = nn.LSTM(128, 64)   # BUG: expects 256
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h = self.lstm1(x)
        h = self.lstm2(h)
        h = self.lstm3(h)
        h = self.lstm4(h)
        return self.fc(h)
''', "input_shapes": {"x": ("seq_len", "batch", 128)}},

    {"name": "emb-bilstm-linear-safe", "expected": True, "category": "lstm-embedding",
     "description": "Embedding → BiLSTM → Linear with correct dimensions",
     "source": '''
import torch.nn as nn
class EmbBiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(30000, 300)
        self.lstm = nn.LSTM(300, 256, bidirectional=True)
        self.fc = nn.Linear(512, 5)
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch")}},

    {"name": "emb-bilstm-linear-bug", "expected": False, "category": "lstm-embedding",
     "description": "Embedding → BiLSTM → Linear with forgotten bidirectional doubling",
     "source": '''
import torch.nn as nn
class EmbBiLSTMBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(30000, 300)
        self.lstm = nn.LSTM(300, 256, bidirectional=True)
        self.fc = nn.Linear(256, 5)  # BUG: should be 512
    def forward(self, x):
        e = self.emb(x)
        out = self.lstm(e)
        return self.fc(out)
''', "input_shapes": {"x": ("seq_len", "batch")}},

    {"name": "gru-seq2seq-safe", "expected": True, "category": "gru-seq2seq",
     "description": "GRU encoder-decoder with matching hidden sizes",
     "source": '''
import torch.nn as nn
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(256, 512)
        self.decoder = nn.GRU(512, 512)
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.fc(dec)
''', "input_shapes": {"x": ("seq_len", "batch", 256)}},

    {"name": "gru-seq2seq-hidden-bug", "expected": False, "category": "gru-seq2seq",
     "description": "GRU encoder-decoder with hidden size mismatch",
     "source": '''
import torch.nn as nn
class Seq2SeqBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(256, 512)
        self.decoder = nn.GRU(256, 512)  # BUG: expects 512 from encoder
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.fc(dec)
''', "input_shapes": {"x": ("seq_len", "batch", 256)}},
]


def run_deep_composition_lstm():
    """Run LSTM deep composition benchmarks."""
    results = []
    correct = 0

    for bench in LSTM_DEEP_COMPOSITION:
        t0 = time.time()
        result = verify_model(bench["source"], input_shapes=bench["input_shapes"])
        elapsed = (time.time() - t0) * 1000

        match = result.safe == bench["expected"]
        if match:
            correct += 1

        results.append({
            "name": bench["name"],
            "category": bench["category"],
            "description": bench["description"],
            "expected_safe": bench["expected"],
            "predicted_safe": result.safe,
            "correct": match,
            "time_ms": round(elapsed, 1),
        })
        print(f"  {'✓' if match else '✗'} {bench['name']}: "
              f"expected={'safe' if bench['expected'] else 'bug'}, "
              f"got={'safe' if result.safe else 'bug'}")

    return {
        "total": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 4) if results else 0,
        "results": results,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("LSTM/GRU Verification Experiment")
    print("=" * 60)
    lstm_results = run_lstm_gru_experiment()
    print(f"\nLSTM/GRU Results: F1={lstm_results['f1']}, "
          f"P={lstm_results['precision']}, R={lstm_results['recall']}, "
          f"Accuracy={lstm_results['accuracy']}")

    print("\n" + "=" * 60)
    print("LSTM Deep Composition Benchmark")
    print("=" * 60)
    deep_results = run_deep_composition_lstm()
    print(f"\nDeep Composition (LSTM): {deep_results['correct']}/{deep_results['total']} "
          f"correct ({deep_results['accuracy']*100}%)")

    # Save results
    all_results = {
        "lstm_gru_experiment": lstm_results,
        "lstm_deep_composition": deep_results,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "lstm_gru_experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
