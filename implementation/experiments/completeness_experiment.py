"""
Completeness experiment: verify relative completeness on 20+ linear-fragment models.

Outputs results to experiments/completeness_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.completeness import check_relative_completeness


def _src(body: str) -> str:
    return textwrap.dedent(body).strip()


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    # --- Safe linear models ---
    {
        "name": "SingleLinear",
        "source": _src("""
            import torch.nn as nn
            class SingleLinear(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 5)
                def forward(self, x):
                    return self.fc(x)
        """),
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": True,
    },
    {
        "name": "TwoLayerMLP",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class TwoLayerMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(20, 10)
                    self.fc2 = nn.Linear(10, 5)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    return self.fc2(x)
        """),
        "input_shapes": {"x": ("batch", 20)},
        "expected_safe": True,
    },
    {
        "name": "ThreeLayerMLP",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class ThreeLayerMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 10)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    return self.fc3(x)
        """),
        "input_shapes": {"x": ("batch", 784)},
        "expected_safe": True,
    },
    {
        "name": "DeepMLP_10layers",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class DeepMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(64, 64)
                    self.fc2 = nn.Linear(64, 64)
                    self.fc3 = nn.Linear(64, 64)
                    self.fc4 = nn.Linear(64, 64)
                    self.fc5 = nn.Linear(64, 64)
                    self.fc6 = nn.Linear(64, 64)
                    self.fc7 = nn.Linear(64, 64)
                    self.fc8 = nn.Linear(64, 64)
                    self.fc9 = nn.Linear(64, 64)
                    self.fc10 = nn.Linear(64, 32)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = F.relu(self.fc3(x))
                    x = F.relu(self.fc4(x))
                    x = F.relu(self.fc5(x))
                    x = F.relu(self.fc6(x))
                    x = F.relu(self.fc7(x))
                    x = F.relu(self.fc8(x))
                    x = F.relu(self.fc9(x))
                    return self.fc10(x)
        """),
        "input_shapes": {"x": ("batch", 64)},
        "expected_safe": True,
    },
    {
        "name": "Conv2d_single",
        "source": _src("""
            import torch.nn as nn
            class Conv2dSingle(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, 3)
                def forward(self, x):
                    return self.conv(x)
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": True,
    },
    {
        "name": "Conv_BatchNorm",
        "source": _src("""
            import torch.nn as nn
            class ConvBN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, 3)
                    self.bn = nn.BatchNorm2d(16)
                def forward(self, x):
                    x = self.conv(x)
                    x = self.bn(x)
                    return x
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": True,
    },
    {
        "name": "Conv_Chain",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class ConvChain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 16, 3)
                    self.conv2 = nn.Conv2d(16, 32, 3)
                    self.conv3 = nn.Conv2d(32, 64, 3)
                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    return self.conv3(x)
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": True,
    },
    {
        "name": "ResidualBlock",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class ResBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(64, 64)
                    self.fc2 = nn.Linear(64, 64)
                def forward(self, x):
                    residual = x
                    x = F.relu(self.fc1(x))
                    x = self.fc2(x)
                    x = x + residual
                    return F.relu(x)
        """),
        "input_shapes": {"x": ("batch", 64)},
        "expected_safe": True,
    },
    {
        "name": "LayerNorm_Linear",
        "source": _src("""
            import torch.nn as nn
            class LNLinear(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ln = nn.LayerNorm(64)
                    self.fc = nn.Linear(64, 32)
                def forward(self, x):
                    x = self.ln(x)
                    return self.fc(x)
        """),
        "input_shapes": {"x": ("batch", 64)},
        "expected_safe": True,
    },
    {
        "name": "Dropout_MLP",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class DropoutMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(32, 16)
                    self.drop = nn.Dropout(0.5)
                    self.fc2 = nn.Linear(16, 8)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = self.drop(x)
                    return self.fc2(x)
        """),
        "input_shapes": {"x": ("batch", 32)},
        "expected_safe": True,
    },
    {
        "name": "Wide_Linear",
        "source": _src("""
            import torch.nn as nn
            class WideLinear(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(1024, 512)
                def forward(self, x):
                    return self.fc(x)
        """),
        "input_shapes": {"x": ("batch", 1024)},
        "expected_safe": True,
    },
    {
        "name": "Bottleneck_MLP",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class BottleneckMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(256, 16)
                    self.fc2 = nn.Linear(16, 256)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    return self.fc2(x)
        """),
        "input_shapes": {"x": ("batch", 256)},
        "expected_safe": True,
    },
    {
        "name": "Conv1d_Linear",
        "source": _src("""
            import torch.nn as nn
            class Conv1dNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv1d(16, 32, 3)
                def forward(self, x):
                    return self.conv(x)
        """),
        "input_shapes": {"x": ("batch", 16, 100)},
        "expected_safe": True,
    },
    {
        "name": "Embedding_Linear",
        "source": _src("""
            import torch.nn as nn
            class EmbedNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(1000, 64)
                    self.fc = nn.Linear(64, 32)
                def forward(self, x):
                    x = self.embed(x)
                    return self.fc(x)
        """),
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": True,
    },
    # --- Unsafe linear models ---
    {
        "name": "Mismatch_Linear",
        "source": _src("""
            import torch.nn as nn
            class MismatchLinear(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(10, 20)
                    self.fc2 = nn.Linear(50, 5)
                def forward(self, x):
                    x = self.fc1(x)
                    return self.fc2(x)
        """),
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": False,
    },
    {
        "name": "Wrong_Input_Dim",
        "source": _src("""
            import torch.nn as nn
            class WrongInput(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 5)
                def forward(self, x):
                    return self.fc(x)
        """),
        "input_shapes": {"x": ("batch", 999)},
        "expected_safe": False,
    },
    {
        "name": "Conv_Channel_Mismatch",
        "source": _src("""
            import torch.nn as nn
            class ConvMismatch(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 16, 3)
                    self.conv2 = nn.Conv2d(32, 64, 3)
                def forward(self, x):
                    x = self.conv1(x)
                    return self.conv2(x)
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": False,
    },
    {
        "name": "BN_Mismatch",
        "source": _src("""
            import torch.nn as nn
            class BNMismatch(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, 3)
                    self.bn = nn.BatchNorm2d(32)
                def forward(self, x):
                    x = self.conv(x)
                    return self.bn(x)
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": False,
    },
    {
        "name": "Deep_Mismatch",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class DeepMismatch(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(64, 64)
                    self.fc2 = nn.Linear(64, 64)
                    self.fc3 = nn.Linear(64, 32)
                    self.fc4 = nn.Linear(64, 16)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = F.relu(self.fc3(x))
                    return self.fc4(x)
        """),
        "input_shapes": {"x": ("batch", 64)},
        "expected_safe": False,
    },
    # --- Additional safe variants ---
    {
        "name": "Identity_Passthrough",
        "source": _src("""
            import torch.nn as nn
            class IdentityNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.id = nn.Identity()
                def forward(self, x):
                    return self.id(x)
        """),
        "input_shapes": {"x": ("batch", 32)},
        "expected_safe": True,
    },
    {
        "name": "Multi_Conv_BN_ReLU",
        "source": _src("""
            import torch.nn as nn
            import torch.nn.functional as F
            class MultiConvBN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(16)
                    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                    self.bn2 = nn.BatchNorm2d(32)
                def forward(self, x):
                    x = F.relu(self.bn1(self.conv1(x)))
                    x = F.relu(self.bn2(self.conv2(x)))
                    return x
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": True,
    },
    {
        "name": "GroupNorm_Net",
        "source": _src("""
            import torch.nn as nn
            class GNNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, 3)
                    self.gn = nn.GroupNorm(4, 16)
                def forward(self, x):
                    x = self.conv(x)
                    return self.gn(x)
        """),
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": True,
    },
]


def main():
    results = []
    n_in_fragment = 0
    n_completeness_verified = 0
    verdicts = {"SAFE": 0, "UNSAFE": 0, "N/A": 0, "ERROR": 0}

    print(f"Running completeness experiment on {len(MODELS)} models...")
    print("=" * 72)

    for model in MODELS:
        t0 = time.monotonic()
        r = check_relative_completeness(model["source"], model["input_shapes"])
        elapsed_ms = (time.monotonic() - t0) * 1000

        if r.in_fragment:
            n_in_fragment += 1
        if r.completeness_verified:
            n_completeness_verified += 1
        verdicts[r.tg_verdict] = verdicts.get(r.tg_verdict, 0) + 1

        status = "✓" if r.completeness_verified else "✗"
        print(f"  {status} {model['name']:30s}  fragment={r.in_fragment!s:5s}  "
              f"verdict={r.tg_verdict:6s}  complete={r.completeness_verified!s:5s}  "
              f"({elapsed_ms:.1f}ms)")

        results.append({
            "name": model["name"],
            "in_fragment": r.in_fragment,
            "tg_verdict": r.tg_verdict,
            "completeness_verified": r.completeness_verified,
            "expected_safe": model["expected_safe"],
            "verdict_matches_expectation": (
                (r.tg_verdict == "SAFE") == model["expected_safe"]
                if r.tg_verdict in ("SAFE", "UNSAFE") else None
            ),
            "explanation": r.explanation,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    print("=" * 72)
    print(f"\nSummary:")
    print(f"  Models tested:              {len(MODELS)}")
    print(f"  In linear fragment:         {n_in_fragment}/{len(MODELS)}")
    print(f"  Completeness verified:      {n_completeness_verified}/{len(MODELS)}")
    print(f"  TG verdicts:                {verdicts}")
    print(f"  Verification rate:          "
          f"{n_completeness_verified/len(MODELS)*100:.1f}%")

    summary = {
        "total_models": len(MODELS),
        "in_fragment": n_in_fragment,
        "completeness_verified": n_completeness_verified,
        "verification_rate": round(n_completeness_verified / len(MODELS) * 100, 1),
        "verdicts": verdicts,
        "models": results,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "completeness_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
