"""Verify TensorGuard headline claims by running it on labeled buggy/safe models."""
import json, time, sys, os, subprocess, tempfile

REPO = '/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard'

# 10 minimal buggy + 10 minimal safe models covering the README/paper claims
BUGS = [
    # (label, code, input_spec, expected_buggy)
    ("conv_view_linear_bug", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16*5*5, 10)
    def forward(self, x):
        x = self.conv(x); x = x.view(x.size(0), -1); return self.fc(x)
''', "x=batch,3,224,224", True),
    ("matmul_dim_mismatch", '''
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 32); self.b = nn.Linear(64, 5)
    def forward(self, x):
        return self.b(self.a(x))
''', "x=batch,10", True),
    ("good_mlp", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32); self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''', "x=batch,10", False),
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
''', "x=batch,3,32,32", False),
    ("transpose_then_linear_bug", '''
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x.transpose(0,1))  # input becomes wrong feature dim
''', "x=batch,10", True),
    ("good_resnet_block", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(64, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        return x + self.c2(self.c1(x))
''', "x=batch,64,8,8", False),
    ("flatten_bug", '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.fc = nn.Linear(100, 10)  # wrong, should be 32*222*222
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))
''', "x=batch,3,224,224", True),
]

def run_tg(args, **kw):
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH','')}"
    return subprocess.run(['python3.11','-m','src.cli.main'] + args,
                          capture_output=True, text=True, cwd=REPO, env=env, **kw)

def detected(stdout, stderr, returncode):
    out = (stdout + stderr).lower()
    return ('error' in out and 'argument' not in out) or 'mismatch' in out or 'shape' in out and 'fail' in out or returncode != 0

if __name__ == "__main__":
    t0 = time.time()
    results = []
    for name, code, spec, is_buggy in BUGS:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(code); path = f.name
        try:
            r = run_tg(['verify', path, '-s', spec], timeout=60)
            flagged = detected(r.stdout, r.stderr, r.returncode)
            correct = (flagged == is_buggy)
            results.append({"name": name, "buggy": is_buggy, "flagged": flagged, "correct": correct})
            print(f"{'✓' if correct else '✗'} {name:30s} buggy={is_buggy} flagged={flagged}")
        finally:
            os.unlink(path)

    tp = sum(1 for r in results if r["buggy"] and r["flagged"])
    fp = sum(1 for r in results if not r["buggy"] and r["flagged"])
    tn = sum(1 for r in results if not r["buggy"] and not r["flagged"])
    fn = sum(1 for r in results if r["buggy"] and not r["flagged"])
    n_buggy = tp + fn
    n_safe = tn + fp
    summary = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_buggy": n_buggy, "n_safe": n_safe,
        "precision": tp / max(tp+fp, 1),
        "recall": tp / max(n_buggy, 1),
        "specificity": tn / max(n_safe, 1),
        "elapsed_s": time.time() - t0,
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    out = os.path.join(REPO, "experiments", "neurips_validation.json")
    json.dump({"results": results, "summary": summary}, open(out, "w"), indent=2)
    print(f"Saved {out}")
