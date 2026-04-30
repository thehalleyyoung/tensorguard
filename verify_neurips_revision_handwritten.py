"""Companion to verify_neurips_revision.py: a HAND-WRITTEN bug that
sits squarely in TensorGuard's DSL fragment (no opaque torchvision
modules), so TG can statically catch it. Compare to FX ShapeProp and
FakeTensor meta-execution.
"""
from __future__ import annotations
import json, os, time, subprocess, tempfile, textwrap

REPO = '/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard'
PYBIN = '/opt/homebrew/bin/python3.11'
OUT = os.path.join(REPO, 'experiments', 'neurips_revision_handwritten_bug.json')

# A hand-written ResNet-style block with the same bug class as
# torchvision-mutated resnet18: backbone produces 512 features but
# classifier expects 256. This is a real bug pattern reported many
# times on PyTorch and HuggingFace issue trackers under variants of
# "I changed the backbone width and forgot to update the head".
BUG_SRC = textwrap.dedent('''
    """ResNet-style classifier with intentional backbone/head mismatch."""
    import torch.nn as nn
    class ResBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1 = nn.Conv2d(c, c, 3, padding=1)
            self.c2 = nn.Conv2d(c, c, 3, padding=1)
        def forward(self, x):
            return x + self.c2(self.c1(x))
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.down3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
            self.b1 = ResBlock(512)
            self.pool = nn.AdaptiveAvgPool2d(1)
            # BUG: backbone outputs 512 channels, classifier expects 256
            self.fc = nn.Linear(256, 1000)
        def forward(self, x):
            x = self.stem(x)
            x = self.down1(x); x = self.down2(x); x = self.down3(x)
            x = self.b1(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)
''').strip() + '\n'

# Same model, fixed (Linear(512,1000)) — used as a control.
GOOD_SRC = BUG_SRC.replace('nn.Linear(256, 1000)', 'nn.Linear(512, 1000)')


def run_tensorguard(path, ishape='x=1,3,224,224', timeout=120):
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH','')}"
    t0 = time.time()
    r = subprocess.run([PYBIN,'-m','src.cli.main','verify',path,
                        '--input-shape',ishape,'--format','json'],
                       cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=timeout)
    dt = time.time() - t0
    status = None
    try:
        j = json.loads(r.stdout)
        status = j.get('status')
        bugs = j.get('bugs', [])
    except Exception:
        status, bugs = None, []
    if r.returncode == 0 and status == 'SAFE':
        verdict = 'verified'
    elif r.returncode == 1 or status == 'UNSAFE':
        verdict = 'rejected'
    else:
        verdict = 'unknown'
    return dict(verdict=verdict, time_s=dt, returncode=r.returncode,
                n_bugs=len(bugs),
                first_bug=(bugs[0].get('message','')[:200] if bugs else ''),
                stderr_tail=r.stderr[-200:] if r.stderr else '')


def run_fx(src, ishape):
    code = textwrap.dedent(f'''
        import json, time
        t0 = time.time()
        try:
            import torch
            from torch.fx import symbolic_trace
            from torch.fx.passes.shape_prop import ShapeProp
            ns = {{}}
            exec({src!r}, ns)
            m = ns["M"]()
            m.eval()
            gm = symbolic_trace(m)
            x = torch.randn(*{ishape!r})
            ShapeProp(gm).propagate(x)
            print(json.dumps(dict(ok=True, time_s=time.time()-t0, err="")))
        except Exception as e:
            print(json.dumps(dict(ok=False, time_s=time.time()-t0,
                                  err=type(e).__name__ + ": " + str(e)[:200])))
    ''').strip()
    r = subprocess.run([PYBIN,'-c',code], capture_output=True,
                       text=True, timeout=180)
    line = (r.stdout.strip().splitlines() or [''])[-1]
    try: return json.loads(line)
    except Exception: return dict(ok=False, time_s=0, err=f'parse: {r.stderr[-200:]!r}')


def run_meta(src, ishape):
    code = textwrap.dedent(f'''
        import json, time
        t0 = time.time()
        try:
            import torch
            from torch._subclasses.fake_tensor import FakeTensorMode
            mode = FakeTensorMode(allow_non_fake_inputs=True)
            ns = {{}}
            exec({src!r}, ns)
            with mode:
                m = ns["M"]()
                m.eval()
                x = torch.randn(*{ishape!r})
                with torch.no_grad():
                    y = m(x)
            print(json.dumps(dict(ok=True, time_s=time.time()-t0,
                                  out_shape=list(y.shape), err="")))
        except Exception as e:
            print(json.dumps(dict(ok=False, time_s=time.time()-t0,
                                  err=type(e).__name__ + ": " + str(e)[:200])))
    ''').strip()
    r = subprocess.run([PYBIN,'-c',code], capture_output=True,
                       text=True, timeout=180)
    line = (r.stdout.strip().splitlines() or [''])[-1]
    try: return json.loads(line)
    except Exception: return dict(ok=False, time_s=0, err=f'parse: {r.stderr[-200:]!r}')


def main():
    out = {}
    for label, src, expect_bug in [('buggy', BUG_SRC, True),
                                   ('control', GOOD_SRC, False)]:
        f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
        f.write(src); f.close()
        path = f.name
        try:
            tg = run_tensorguard(path)
        finally:
            try: os.unlink(path)
            except OSError: pass
        fx = run_fx(src, (1,3,224,224))
        meta = run_meta(src, (1,3,224,224))
        out[label] = dict(expected_bug=expect_bug,
                          tensorguard=tg, fx_shapeprop=fx, meta_forward=meta)
        print(f'{label}: TG={tg["verdict"]}  FX_OK={fx["ok"]}  META_OK={meta["ok"]}')
        if tg['first_bug']:
            print(f'  TG bug: {tg["first_bug"][:160]}')
        if not fx['ok']:
            print(f'  FX err: {fx["err"][:160]}')
        if not meta['ok']:
            print(f'  META err: {meta["err"][:160]}')
    json.dump(out, open(OUT,'w'), indent=2)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
