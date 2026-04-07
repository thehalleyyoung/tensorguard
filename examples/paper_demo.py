#!/usr/bin/env python3
"""TensorGuard Demo — runs all examples from the papers.

Tests all 8 model examples from paper.tex and pldi_paper.tex:
  7 buggy models  (expected: UNSAFE)
  1 correct model (expected: SAFE)
"""
import sys
import os
import tempfile
import textwrap

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import verify_module

# ── Paper examples ────────────────────────────────────────────────────

EXAMPLES = [
    # ── 1. BuggyTransformer (paper.tex) ──────────────────────────────
    {
        "name": "BuggyTransformer",
        "paper": "paper.tex",
        "expected": "UNSAFE",
        "bug_type": "shape error (proj outputs 256, caller expects 512)",
        "input_shapes": {"x": ("seq", "batch", 512)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class BuggyTransformer(nn.Module):
                def __init__(self, d_model=512, n_heads=8, d_ff=2048):
                    super().__init__()
                    self.attn = nn.MultiheadAttention(d_model, n_heads)
                    self.ff = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.ReLU(),
                        nn.Linear(d_ff, d_model),
                    )
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.dropout = nn.Dropout(0.1)
                    # BUG: projection has wrong output dimension
                    self.proj = nn.Linear(d_model, 256)

                def forward(self, x):           # x: (seq, batch, d_model)
                    attn_out, _ = self.attn(x, x, x)
                    x = self.norm1(x + self.dropout(attn_out))
                    ff_out = self.ff(x)
                    x = self.norm2(x + self.dropout(ff_out))
                    # BUG: proj outputs (seq, batch, 256), x is (seq, batch, 512)
                    # residual add fails: 256 != 512
                    return x + self.proj(x)
        """),
    },
    # ── 2. TransferLearningBug (pldi_paper.tex) ──────────────────────
    {
        "name": "TransferLearningBug",
        "paper": "pldi_paper.tex",
        "expected": "UNSAFE",
        "bug_type": "phase × shape (eval_head expects 128, gets 256)",
        "input_shapes": {"x": ("B", 512)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class TransferLearningBug(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Linear(512, 256)
                    self.train_head = nn.Linear(256, 100)
                    self.eval_head = nn.Linear(128, 10)  # BUG

                def forward(self, x):
                    h = self.backbone(x)
                    if self.training:
                        return self.train_head(h)        # OK
                    else:
                        return self.eval_head(h)         # CRASH
        """),
    },
    # ── 3. PositionEmbeddingBug (pldi_paper.tex) ─────────────────────
    {
        "name": "PositionEmbeddingBug",
        "paper": "pldi_paper.tex",
        "expected": "UNSAFE",
        "bug_type": "device mismatch (buffer on CPU after model.cuda())",
        "input_shapes": {"x": ("B", 512, 768)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class PositionEmbeddingBug(nn.Module):
                def __init__(self, max_len=512, dim=768):
                    super().__init__()
                    self.proj = nn.Linear(dim, dim)
                    # Buffer stays on CPU after model.cuda()
                    self.register_buffer('pos_ids',
                        torch.arange(max_len).unsqueeze(0))

                def forward(self, x):
                    pos = self.pos_ids[:, :x.size(1)]
                    return self.proj(x) + pos  # BUG: device mismatch
        """),
    },
    # ── 4. YOLODetectorBug (pldi_paper.tex) ──────────────────────────
    {
        "name": "YOLODetectorBug",
        "paper": "pldi_paper.tex",
        "expected": "UNSAFE",
        "bug_type": "shape × device × phase (anchor_grid wrong dims, CPU, eval-only)",
        "input_shapes": {"x": ("B", 3, 32, 32)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class YOLODetectorBug(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Conv2d(3, 64, 3, padding=1)
                    self.head = nn.Conv2d(64, 255, 1)
                    # Wrong spatial dims AND stays on CPU
                    self.register_buffer('anchor_grid',
                        torch.randn(1, 3, 10, 10, 2))

                def forward(self, x):
                    feat = self.head(self.backbone(x))
                    if not self.training:  # post-processing
                        B, C, H, W = feat.shape
                        # anchor_grid has spatial 10x10, but feat may
                        # have different H,W; also device mismatch
                        out = feat.view(B, 3, 85, H, W)
                        out = out + self.anchor_grid  # BUG x3
                    return feat
        """),
    },
    # ── 5. FPNBug (pldi_paper.tex) ───────────────────────────────────
    {
        "name": "FPNBug",
        "paper": "pldi_paper.tex",
        "expected": "UNSAFE",
        "bug_type": "shape error (lateral_c4 expects 128 channels, gets 256)",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class FPNBug(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone_c3 = nn.Conv2d(3, 128, 3, padding=1)
                    self.backbone_c4 = nn.Conv2d(128, 256, 3, stride=2,
                                                  padding=1)
                    self.backbone_c5 = nn.Conv2d(256, 512, 3, stride=2,
                                                  padding=1)
                    # BUG: lateral_c4 expects 128 channels but gets 256
                    self.lateral_c4 = nn.Conv2d(128, 256, 1)
                    self.lateral_c5 = nn.Conv2d(512, 256, 1)

                def forward(self, x):
                    c3 = self.backbone_c3(x)
                    c4 = self.backbone_c4(c3)
                    c5 = self.backbone_c5(c4)
                    p5 = self.lateral_c5(c5)
                    # lateral_c4 gets c4 (256 channels), expects 128
                    p4 = self.lateral_c4(c4) + F.interpolate(
                        p5, scale_factor=2)
                    return p4
        """),
    },
    # ── 6. SEBlockBug (pldi_paper.tex) ───────────────────────────────
    {
        "name": "SEBlockBug",
        "paper": "pldi_paper.tex",
        "expected": "UNSAFE",
        "bug_type": "shape error (SE block expects 256, conv outputs 512)",
        "input_shapes": {"x": ("B", 3, 32, 32)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class SEBlockBug(nn.Module):
                def __init__(self, channels=512, reduction=16):
                    super().__init__()
                    self.conv = nn.Conv2d(3, channels, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    # BUG: SE block was designed for channels=256
                    self.se_fc1 = nn.Linear(256, 256 // reduction)
                    self.se_fc2 = nn.Linear(256 // reduction, 256)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    out = self.conv(x)
                    se = self.pool(out).view(out.size(0), -1)
                    se = self.relu(self.se_fc1(se))  # BUG: 512 != 256
                    se = self.sigmoid(self.se_fc2(se))
                    return out * se.unsqueeze(-1).unsqueeze(-1)
        """),
    },
    # ── 7. AnchorGeneratorBug (pldi_paper.tex) ───────────────────────
    {
        "name": "AnchorGeneratorBug",
        "paper": "pldi_paper.tex",
        "expected": "UNSAFE",
        "bug_type": "shape + device + phase (anchor buffer wrong dims, CPU, eval-only)",
        "input_shapes": {"x": ("B", 3, 32, 32)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class AnchorGeneratorBug(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Conv2d(3, 256, 3, padding=1)
                    self.cls_head = nn.Conv2d(256, 9 * 20, 1)
                    # BUG 1: Wrong spatial dims (8x8 vs feature map)
                    # BUG 2: Stays on CPU
                    # BUG 3: Only used in eval (post-processing)
                    self.register_buffer('anchors',
                        torch.randn(1, 9, 8, 8, 4))

                def forward(self, x):
                    feat = self.backbone(x)
                    cls = self.cls_head(feat)
                    if not self.training:
                        B, C, H, W = cls.shape
                        cls = cls.view(B, 9, 20, H, W)
                        # Triple bug: shape + device + phase
                        anchors = self.anchors[:, :, :H, :W, :]
                        output = torch.cat([cls, anchors], dim=2)
                    return cls
        """),
    },
    # ── 8. CorrectMultiTheory (pldi_paper.tex) ───────────────────────
    {
        "name": "CorrectMultiTheory",
        "paper": "pldi_paper.tex",
        "expected": "SAFE",
        "bug_type": "(none — should verify clean)",
        "input_shapes": {"x": ("B", 3, 32, 32)},
        "source": textwrap.dedent("""\
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class CorrectMultiTheory(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 64, 3, padding=1)
                    self.bn = nn.BatchNorm2d(64)
                    self.fc = nn.Linear(64, 10)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.dropout = nn.Dropout(0.5)

                def forward(self, x):
                    x = F.relu(self.bn(self.conv(x)))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)  # (B, 64)
                    if self.training:
                        x = self.dropout(x)  # shape-preserving
                    return self.fc(x)  # (B, 64) -> (B, 10)
        """),
    },
]

# Also include the two existing example files
EXAMPLE_FILES = [
    {
        "name": "QuickStart (SmallResNet)",
        "path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "quickstart.py"),
        "expected": "SAFE",
        "bug_type": "(none — correct ResNet)",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
    },
    {
        "name": "ShapeBug (BuggyModel)",
        "path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_bug.py"),
        "expected": "UNSAFE",
        "bug_type": "shape error (fc expects 256, conv outputs 128)",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
]


def run_demo():
    print("=" * 70)
    print("  TensorGuard Demo — Paper Examples")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    total = len(EXAMPLES) + len(EXAMPLE_FILES)

    # ── Inline paper examples ─────────────────────────────────────────
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"─── [{i}/{total}] {ex['name']} ({ex['paper']}) ───")
        print(f"  Expected: {ex['expected']}  |  Bug: {ex['bug_type']}")

        # Write source to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(ex["source"])
            tmp_path = f.name

        try:
            result = verify_module(
                tmp_path,
                input_shapes=ex["input_shapes"],
                check_devices=True,
                check_phases=True,
            )
            status = result.status
            ok = status == ex["expected"]

            icon = "✓" if ok else "✗"
            color_status = status
            print(f"  Result:   {color_status}  ({result.duration_ms:.0f}ms)")
            if result.bugs:
                for bug in result.bugs:
                    print(f"    → {bug.message}")
            if ok:
                print(f"  {icon} PASS")
                passed += 1
            else:
                print(f"  {icon} FAIL (expected {ex['expected']}, got {status})")
                failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        finally:
            os.unlink(tmp_path)

        print()

    # ── Existing example files ────────────────────────────────────────
    for j, ex in enumerate(EXAMPLE_FILES, len(EXAMPLES) + 1):
        print(f"─── [{j}/{total}] {ex['name']} ───")
        print(f"  Expected: {ex['expected']}  |  Bug: {ex['bug_type']}")
        print(f"  File:     {ex['path']}")

        try:
            result = verify_module(
                ex["path"],
                input_shapes=ex["input_shapes"],
                check_devices=True,
                check_phases=True,
            )
            status = result.status
            ok = status == ex["expected"]

            icon = "✓" if ok else "✗"
            print(f"  Result:   {status}  ({result.duration_ms:.0f}ms)")
            if result.bugs:
                for bug in result.bugs:
                    print(f"    → {bug.message}")
            if ok:
                print(f"  {icon} PASS")
                passed += 1
            else:
                print(f"  {icon} FAIL (expected {ex['expected']}, got {status})")
                failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  ✓ All paper examples verified correctly!")
    else:
        print(f"  ✗ {failed} example(s) did not match expected verdict")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_demo())
