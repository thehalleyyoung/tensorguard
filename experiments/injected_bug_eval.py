"""Injected-bug benchmark.

Take a small set of real PyTorch model files (clean copies cached
under experiments/.cache/injected_bug/) and apply programmatic
mutations that flip a correct file into a buggy one. For each
(file, mutation) pair, we run TensorGuard, FX ShapeProp, and
FakeTensor, and report catch rates.

Mutation classes:
- off_by_one_linear:  nn.Linear(in, out)  -> nn.Linear(in - 1, out)
- wrong_reshape:      .view(B, C, H, W)   -> .view(B, C, H, W + 1)
- transpose_swap:     .transpose(1, 2)    -> .transpose(0, 1)

We only mutate models we can actually instantiate dynamically (so
that FX/Meta have something to compare against). Output:
experiments/injected_bug_eval.json
"""
from __future__ import annotations
import json, os, re, time, subprocess, tempfile, textwrap

REPO  = '/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard'
PYBIN = '/opt/homebrew/bin/python3.11'
OUT   = os.path.join(REPO, 'experiments', 'injected_bug_eval.json')


# 10 small, self-contained models we author (kept inside the script for
# reproducibility; each is a real-world bug archetype).
SOURCES = {
    'tiny_mlp': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 10)
            def forward(self, x):
                return self.fc2(self.fc1(x))
    ''').strip(),
    'mlp_3layer': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 10)
            def forward(self, x):
                return self.fc3(self.fc2(self.fc1(x)))
    ''').strip(),
    'cnn_classifier': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 16, 3, padding=1)
                self.c2 = nn.Conv2d(16, 32, 3, padding=1)
                self.fc = nn.Linear(32 * 8 * 8, 10)
            def forward(self, x):
                x = self.c1(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.c2(x)
                x = nn.functional.max_pool2d(x, 2)
                x = x.view(x.size(0), 32 * 8 * 8)
                return self.fc(x)
    ''').strip(),
    'two_branch_add': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 32)
                self.b = nn.Linear(64, 32)
            def forward(self, x):
                return self.a(x) + self.b(x)
    ''').strip(),
    'transpose_then_linear': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 10)
            def forward(self, x):                 # (B, 32, 64)
                x = x.transpose(1, 2)             # (B, 64, 32)
                x = x.transpose(1, 2)             # (B, 32, 64)
                return self.fc(x)
    ''').strip(),
    'view_chain': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(48, 10)
            def forward(self, x):                 # (B, 4, 12)
                x = x.view(x.size(0), 4 * 12)
                return self.fc(x)
    ''').strip(),
    'conv_chain': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 8, 3, padding=1)
                self.c2 = nn.Conv2d(8, 16, 3, padding=1)
                self.c3 = nn.Conv2d(16, 32, 3, padding=1)
            def forward(self, x):
                return self.c3(self.c2(self.c1(x)))
    ''').strip(),
    'flatten_classifier': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16 * 16 * 16, 5)
            def forward(self, x):
                x = self.c(x)
                x = nn.functional.max_pool2d(x, 2)
                x = x.view(x.size(0), 16 * 16 * 16)
                return self.fc(x)
    ''').strip(),
    'conv_then_flatten_to_linear': textwrap.dedent('''
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(1, 4, 3, padding=1)
                self.fc = nn.Linear(4 * 28 * 28, 10)
            def forward(self, x):
                x = self.c(x)
                x = x.view(x.size(0), 4 * 28 * 28)
                return self.fc(x)
    ''').strip(),
    'broadcast_pair': textwrap.dedent('''
        import torch
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(32, 16)
                self.b = nn.Linear(32, 16)
            def forward(self, x, y):
                return self.a(x) + self.b(y)
    ''').strip(),
}

# Per-model concrete input shape. We use these for both the dynamic
# tools and TG. (For broadcast_pair we cheat and only run TG.)
INPUTS = {
    'tiny_mlp':                    (4, 64),
    'mlp_3layer':                  (4, 128),
    'cnn_classifier':              (1, 3, 32, 32),
    'two_branch_add':              (4, 64),
    'transpose_then_linear':       (1, 32, 64),
    'view_chain':                  (1, 4, 12),
    'conv_chain':                  (1, 3, 16, 16),
    'flatten_classifier':          (1, 3, 32, 32),
    'conv_then_flatten_to_linear': (1, 1, 28, 28),
    'broadcast_pair':              (4, 32),
}

# (mutation_id, regex_pattern, replacement); applied to the source
# of one model at a time. Each mutation flips correctness.
MUTATIONS = [
    # off-by-one in the FIRST nn.Linear's input dim
    ('off_by_one_linear',
     re.compile(r'nn\.Linear\(\s*(\d+)\s*,'),
     (lambda m: 'nn.Linear(' + str(int(m.group(1))-1) + ','),
     'one off in the input dim of the first nn.Linear'),
    # wrong reshape: change last view dim by +1
    ('wrong_reshape',
     re.compile(r'\.view\(([^)]+),\s*(\d+(?:\s*\*\s*\d+)*)\s*\)'),
     (lambda m: '.view(' + m.group(1) + ', ' + m.group(2) + ' + 1)'),
     '+1 added to the last view dim'),
    # transpose swap: .transpose(1,2) -> .transpose(0,1)
    ('transpose_swap',
     re.compile(r'\.transpose\(\s*1\s*,\s*2\s*\)'),
     (lambda m: '.transpose(0, 1)'),
     'transpose(1,2) replaced with transpose(0,1)'),
]


def mutate(src: str, mut) -> tuple[str, bool]:
    """Apply mutation `mut` to source. Returns (new_src, changed)."""
    _, pat, repl, _ = mut
    new, n = pat.subn(repl, src, count=1)
    return new, n > 0


# ---------------------------- TG / FX / Meta runners ---------------------------- #
def run_tensorguard(path: str, ishape: tuple, timeout: int = 60) -> dict:
    ishape_str = 'x=' + ','.join(str(d) for d in ishape)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH','')}"
    t0 = time.time()
    try:
        r = subprocess.run([PYBIN, '-m', 'src.cli.main', 'verify', path,
                            '--input-shape', ishape_str, '--format', 'json'],
                           cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=timeout)
        dt = time.time() - t0
        status, n_bugs, first = None, 0, ''
        try:
            j = json.loads(r.stdout)
            status = j.get('status')
            bugs = j.get('bugs', []) or []
            n_bugs = len(bugs)
            if bugs: first = bugs[0].get('message','')[:200]
        except Exception:
            pass
        if status == 'SAFE' and r.returncode == 0:
            verdict = 'verified'
        elif status == 'UNSAFE' or (r.returncode == 1 and n_bugs):
            verdict = 'rejected'
        else:
            verdict = 'unknown'
        return dict(verdict=verdict, time_s=dt, n_bugs=n_bugs, first_bug=first)
    except subprocess.TimeoutExpired:
        return dict(verdict='timeout', time_s=time.time()-t0)
    except Exception as e:
        return dict(verdict='error', err=repr(e))


_DYN = textwrap.dedent('''
    import json, time, sys, importlib.util, traceback
    SRC = {src!r}; ISHAPE = {ishape!r}; MULTI = {multi!r}
    def _load():
        spec = importlib.util.spec_from_file_location("ib_mod", SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ib_mod"] = mod
        spec.loader.exec_module(mod)
        return mod
    out = dict(import_ok=False, build_ok=False, fx_ok=False, meta_ok=False,
               fx_err='', meta_err='')
    try: mod = _load(); out['import_ok'] = True
    except Exception as e:
        out['fx_err'] = out['meta_err'] = 'import:' + type(e).__name__
        print(json.dumps(out)); sys.exit(0)
    try: m = mod.M(); out['build_ok'] = True
    except Exception as e:
        out['fx_err'] = out['meta_err'] = 'build:' + type(e).__name__
        print(json.dumps(out)); sys.exit(0)
    import torch
    def _xs():
        if MULTI:
            return [torch.randn(*ISHAPE) for _ in range(2)]
        return [torch.randn(*ISHAPE)]
    try:
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
        m.eval()
        gm = symbolic_trace(m)
        ShapeProp(gm).propagate(*_xs())
        out['fx_ok'] = True
    except Exception as e:
        out['fx_err'] = type(e).__name__ + ': ' + str(e)[:160]
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode:
            m2 = mod.M(); m2.eval()
            xs = _xs()
            with torch.no_grad():
                _ = m2(*xs)
        out['meta_ok'] = True
    except Exception as e:
        out['meta_err'] = type(e).__name__ + ': ' + str(e)[:160]
    print(json.dumps(out))
''').strip()


def run_dynamic(path: str, ishape: tuple, multi: bool = False) -> dict:
    code = _DYN.format(src=path, ishape=ishape, multi=multi)
    try:
        r = subprocess.run([PYBIN, '-c', code], capture_output=True,
                           text=True, timeout=120)
        line = (r.stdout.strip().splitlines() or [''])[-1]
        try: return json.loads(line)
        except Exception:
            return dict(import_ok=False, build_ok=False, fx_ok=False,
                        meta_ok=False, parse_err=r.stderr[-200:])
    except subprocess.TimeoutExpired:
        return dict(fx_err='timeout', meta_err='timeout')


def write(src: str) -> str:
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    f.write(src); f.close()
    return f.name


def main():
    rows = []
    print('=== Injected-bug eval ===')
    # First, baseline (clean): TG must say verified, FX/Meta must succeed
    print('\n-- baselines (clean) --')
    for name, src in SOURCES.items():
        ishape = INPUTS[name]
        multi = (name == 'broadcast_pair')
        path = write(src)
        try:
            tg = run_tensorguard(path, ishape)
            dyn = run_dynamic(path, ishape, multi=multi)
        finally:
            try: os.unlink(path)
            except OSError: pass
        rows.append(dict(model=name, mutation='none', clean=True,
                         expected_bug=False,
                         tensorguard=tg, dynamic=dyn))
        print(f'  {name:32s}  TG={tg["verdict"]:9s}  '
              f'FX={"ok" if dyn.get("fx_ok") else "fail"}  '
              f'META={"ok" if dyn.get("meta_ok") else "fail"}')

    # Mutated rows
    print('\n-- mutations --')
    for mid, pat, repl, desc in MUTATIONS:
        for name, src in SOURCES.items():
            new_src, changed = mutate(src, (mid, pat, repl, desc))
            if not changed:
                continue
            ishape = INPUTS[name]
            multi = (name == 'broadcast_pair')
            path = write(new_src)
            try:
                tg = run_tensorguard(path, ishape)
                dyn = run_dynamic(path, ishape, multi=multi)
            finally:
                try: os.unlink(path)
                except OSError: pass
            rows.append(dict(model=name, mutation=mid, clean=False,
                             expected_bug=True, mutation_desc=desc,
                             tensorguard=tg, dynamic=dyn))
            print(f'  {name:24s} +{mid:18s}  '
                  f'TG={tg["verdict"]:9s}  '
                  f'FX={"caught" if not dyn.get("fx_ok") else "missed"}  '
                  f'META={"caught" if not dyn.get("meta_ok") else "missed"}')

    # Aggregate
    def _is_caught_tg(r):  return r['tensorguard']['verdict'] == 'rejected'
    def _is_caught_fx(r):  return not r['dynamic'].get('fx_ok', False) \
                                  and r['dynamic'].get('build_ok', False)
    def _is_caught_meta(r):return not r['dynamic'].get('meta_ok', False) \
                                  and r['dynamic'].get('build_ok', False)

    buggy = [r for r in rows if r['expected_bug']]
    clean = [r for r in rows if not r['expected_bug']]

    summary = dict(
        n_buggy=len(buggy), n_clean=len(clean),
        tensorguard=dict(
            tp=sum(1 for r in buggy if _is_caught_tg(r)),
            fn=sum(1 for r in buggy if not _is_caught_tg(r)),
            fp=sum(1 for r in clean if _is_caught_tg(r)),
            tn=sum(1 for r in clean if not _is_caught_tg(r)),
        ),
        fx_shapeprop=dict(
            tp=sum(1 for r in buggy if _is_caught_fx(r)),
            fn=sum(1 for r in buggy if not _is_caught_fx(r)),
            fp=sum(1 for r in clean if _is_caught_fx(r)),
            tn=sum(1 for r in clean if not _is_caught_fx(r)),
        ),
        meta_forward=dict(
            tp=sum(1 for r in buggy if _is_caught_meta(r)),
            fn=sum(1 for r in buggy if not _is_caught_meta(r)),
            fp=sum(1 for r in clean if _is_caught_meta(r)),
            tn=sum(1 for r in clean if not _is_caught_meta(r)),
        ),
    )
    out = dict(summary=summary, rows=rows, python=PYBIN, repo=REPO,
               mutations=[(mid, desc) for mid, _, _, desc in MUTATIONS],
               models=list(SOURCES.keys()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {OUT}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
