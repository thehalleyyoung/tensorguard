"""Evaluation of relational shape constraints for transformer architectures."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model

# ─── Model sources ────────────────────────────────────────────────────────────

CORRECT_MHA = """\
import torch.nn as nn

class MultiHeadAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
"""

BUGGY_MHA = """\
import torch.nn as nn

class BuggyMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        return self.out_proj(q)
"""

ENCODER_DECODER = """\
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 512)
        self.decoder = nn.Linear(512, 512)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)
"""

INCOMPATIBLE_MODEL = """\
import torch.nn as nn

class IncompatibleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
"""

# ─── Experiments ──────────────────────────────────────────────────────────────


def run_experiment(name, source, input_shapes, constraints, expected):
    """Run a single verification experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Expected:   {expected}")
    print(f"Constraints: {constraints}")

    t0 = time.monotonic()
    result = verify_model(source, input_shapes=input_shapes, constraints=constraints)
    elapsed = (time.monotonic() - t0) * 1000

    status = "SAFE" if result.safe else "BUG FOUND"
    match = (status == expected)
    print(f"Result:     {status}")
    print(f"Match:      {'✓' if match else '✗'}")
    print(f"Time:       {elapsed:.1f} ms")

    return {
        "name": name,
        "status": status,
        "expected": expected,
        "match": match,
        "time_ms": round(elapsed, 2),
        "constraints": {k: str(v) for k, v in constraints.items()} if constraints else {},
        "errors": result.errors if hasattr(result, "errors") and result.errors else [],
    }


def main():
    results = []

    # 1. Correct MHA with relational constraints → SAFE
    results.append(run_experiment(
        name="correct_mha_relational",
        source=CORRECT_MHA,
        input_shapes={"x": ("batch", "seq_len", "embed_dim")},
        constraints={
            "embed_dim": "heads * head_dim",
            "heads": 8,
            "head_dim": 64,
        },
        expected="SAFE",
    ))

    # 2. Buggy MHA (wrong projection dim) → BUG FOUND
    results.append(run_experiment(
        name="buggy_mha_wrong_projection",
        source=BUGGY_MHA,
        input_shapes={"x": ("batch", "seq_len", "embed_dim")},
        constraints={"embed_dim": "heads * head_dim", "heads": 8},
        expected="BUG FOUND",
    ))

    # 3. Correct encoder-decoder with shared embed_dim → SAFE
    results.append(run_experiment(
        name="encoder_decoder_shared_dim",
        source=ENCODER_DECODER,
        input_shapes={"x": ("batch", "seq_len", "embed_dim")},
        constraints={"embed_dim": 512},
        expected="SAFE",
    ))

    # 4. Model with incompatible relational constraints → detected
    results.append(run_experiment(
        name="incompatible_constraints",
        source=INCOMPATIBLE_MODEL,
        input_shapes={"x": ("batch", "seq_len", "embed_dim")},
        constraints={"embed_dim": 512},
        expected="BUG FOUND",
    ))

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    matched = sum(1 for r in results if r["match"])
    print(f"Matched expectations: {matched}/{total}")
    for r in results:
        mark = "✓" if r["match"] else "✗"
        print(f"  {mark} {r['name']}: {r['status']} (expected {r['expected']})")

    # ─── Save results ─────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "relational_constraints_results.json")
    with open(output_path, "w") as f:
        json.dump({"experiments": results, "matched": matched, "total": total}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
