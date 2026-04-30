"""Extended TensorGuard verification with a substantially larger labeled corpus."""
import json, time, sys, os, subprocess, tempfile

REPO = '/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard'

BUGS = [
    # (label, code, input_spec, expected_buggy, category)
    ("conv_view_linear_bug", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16*5*5, 10)
    def forward(self, x):
        x = self.conv(x); x = x.view(x.size(0), -1); return self.fc(x)
''', "x=batch,3,224,224", True, "shape_after_conv"),
    ("matmul_dim_mismatch", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 32); self.b = nn.Linear(64, 5)
    def forward(self, x):
        return self.b(self.a(x))
''', "x=batch,10", True, "linear_chain"),
    ("good_mlp", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32); self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''', "x=batch,10", False, "good_baseline"),
    ("good_conv", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = self.conv(x); x = self.pool(x).flatten(1); return self.fc(x)
''', "x=batch,3,32,32", False, "good_baseline"),
    ("transpose_then_linear_bug", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x.transpose(0,1))
''', "x=batch,10", True, "axis_swap"),
    ("good_resnet_block", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(64, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        return x + self.c2(self.c1(x))
''', "x=batch,64,8,8", False, "skip_connection"),
    ("flatten_bug", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.fc = nn.Linear(100, 10)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))
''', "x=batch,3,224,224", True, "shape_after_conv"),
    ("good_two_branch", '''
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 16); self.b = nn.Linear(10, 16)
    def forward(self, x):
        return self.a(x) + self.b(x)
''', "x=batch,10", False, "good_baseline"),
    ("bad_two_branch_size", '''
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 16); self.b = nn.Linear(10, 8)
    def forward(self, x):
        return self.a(x) + self.b(x)
''', "x=batch,10", True, "broadcast_mismatch"),
    ("good_chain_3", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(20, 30)
        self.l2 = nn.Linear(30, 40)
        self.l3 = nn.Linear(40, 5)
    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))
''', "x=batch,20", False, "good_baseline"),
    ("bad_chain_middle", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(20, 30)
        self.l2 = nn.Linear(40, 50)  # wrong input
        self.l3 = nn.Linear(50, 5)
    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))
''', "x=batch,20", True, "linear_chain"),
    ("good_conv_chain", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.c1(x); x = self.c2(x); x = self.pool(x).flatten(1); return self.fc(x)
''', "x=batch,3,32,32", False, "good_baseline"),
    ("bad_conv_channel", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(8, 32, 3, padding=1)  # expects 8, gets 16
    def forward(self, x):
        return self.c2(self.c1(x))
''', "x=batch,3,32,32", True, "conv_channel"),
    ("bad_view_neg1", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(50, 10)
    def forward(self, x):
        return self.fc(x.view(-1, 100))  # wrong feature dim
''', "x=batch,50", True, "view_mismatch"),
]

def run_tg(args, **kw):
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH','')}"
    return subprocess.run(['python3.11', '-m', 'src.cli.main'] + args,
                          capture_output=True, text=True, cwd=REPO, env=env, **kw)

def detected(stdout, stderr, returncode):
    out = (stdout + stderr).lower()
    return ('error' in out and 'argument' not in out) \
        or 'mismatch' in out \
        or ('shape' in out and 'fail' in out) \
        or returncode != 0

def main():
    t0 = time.time()
    results = []
    for name, code, spec, is_buggy, cat in BUGS:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(code); path = f.name
        t1 = time.time()
        try:
            r = run_tg(['verify', path, '-s', spec], timeout=60)
            tg_ms = (time.time() - t1) * 1000
            flagged = detected(r.stdout, r.stderr, r.returncode)
            correct = (flagged == is_buggy)
            results.append({"name": name, "category": cat, "buggy": is_buggy,
                            "flagged": flagged, "correct": correct,
                            "time_ms": tg_ms})
            print(f"{'✓' if correct else '✗'} {name:25s} cat={cat:18s}"
                  f" buggy={is_buggy} flagged={flagged}  t={tg_ms:.0f}ms")
        finally:
            os.unlink(path)

    tp = sum(1 for r in results if r["buggy"] and r["flagged"])
    fp = sum(1 for r in results if not r["buggy"] and r["flagged"])
    tn = sum(1 for r in results if not r["buggy"] and not r["flagged"])
    fn = sum(1 for r in results if r["buggy"] and not r["flagged"])

    # Per category breakdown
    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, []).append(r)
    cat_summary = {}
    for c, rs in by_cat.items():
        cat_summary[c] = {
            "n": len(rs),
            "n_buggy": sum(1 for x in rs if x["buggy"]),
            "n_correct": sum(1 for x in rs if x["correct"]),
            "mean_time_ms": sum(x["time_ms"] for x in rs) / len(rs),
        }

    summary = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_buggy": tp + fn, "n_safe": tn + fp, "n_total": len(results),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tp + tn) / max(len(results), 1),
        "mean_time_ms": sum(r["time_ms"] for r in results) / max(len(results), 1),
        "elapsed_s": time.time() - t0,
        "by_category": cat_summary,
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    out = os.path.join(REPO, "experiments", "neurips_validation_extended.json")
    json.dump({"results": results, "summary": summary}, open(out, "w"), indent=2)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
