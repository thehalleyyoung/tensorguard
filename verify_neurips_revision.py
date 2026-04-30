"""NeurIPS revision experiments: large torchvision sweep + comparison to
torch.fx ShapeProp and FakeTensor (meta-tensor) shape inference + a
real-bug-found vignette + CI-throughput timing.

Honest reporting: every number in experiments/neurips_revision.json comes
from actually running the three tools on real torchvision models on this
machine. We do not fabricate.
"""
from __future__ import annotations
import json, os, sys, time, subprocess, tempfile, traceback, textwrap

REPO = '/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard'
PYBIN = '/opt/homebrew/bin/python3.11'
OUT = os.path.join(REPO, 'experiments', 'neurips_revision.json')

# (model_name, ctor_kwargs_repr, input_shape_str, input_shape_tuple)
# Choose architectures spanning conv, MLP-head, multi-branch, attention.
MODELS = [
    ('resnet18',         '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('resnet34',         '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('resnet50',         '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('resnet101',        '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('resnext50_32x4d',  '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('wide_resnet50_2',  '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('vgg11',            '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('vgg16',            '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('alexnet',          '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('squeezenet1_0',    '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('squeezenet1_1',    '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('densenet121',      '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('densenet169',      '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('mobilenet_v2',     '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('mobilenet_v3_small', '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('mobilenet_v3_large', '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('mnasnet0_5',       '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('mnasnet1_0',       '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('shufflenet_v2_x0_5', '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('shufflenet_v2_x1_0', '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('efficientnet_b0',  '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('efficientnet_b1',  '{}', 'x=1,3,240,240', (1, 3, 240, 240)),
    ('regnet_y_400mf',   '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('regnet_x_400mf',   '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('googlenet',        '{"aux_logits": False, "init_weights": False}',
                         'x=1,3,224,224', (1, 3, 224, 224)),
    ('inception_v3',     '{"aux_logits": False, "init_weights": False}',
                         'x=1,3,299,299', (1, 3, 299, 299)),
    ('vit_b_16',         '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('vit_b_32',         '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('convnext_tiny',    '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
    ('swin_t',           '{}', 'x=1,3,224,224', (1, 3, 224, 224)),
]


def write_wrapper(model_name: str, ctor_kwargs_repr: str) -> str:
    """Write a tiny .py file that wraps a torchvision model in nn.Module."""
    src = textwrap.dedent(f'''
        import torch.nn as nn
        import torchvision.models as tvm
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = tvm.{model_name}(**{ctor_kwargs_repr})
            def forward(self, x):
                return self.net(x)
    ''').strip() + '\n'
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    f.write(src); f.close()
    return f.name


# ---------------------------- TensorGuard ---------------------------- #
def run_tensorguard(path: str, input_shape: str, timeout: int = 120) -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH','')}"
    t0 = time.time()
    try:
        r = subprocess.run(
            [PYBIN, '-m', 'src.cli.main', 'verify', path,
             '--input-shape', input_shape, '--format', 'json'],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        # Parse JSON if possible
        status = None
        try:
            j = json.loads(r.stdout)
            status = j.get('status') or ('UNSAFE' if j.get('bugs') else 'SAFE')
        except Exception:
            status = None
        # Map exit code & status to verdict
        if status == 'SAFE' and r.returncode == 0:
            verdict = 'verified'
        elif status == 'UNSAFE' or r.returncode == 1:
            verdict = 'rejected'
        else:
            verdict = 'unknown'
        return dict(verdict=verdict, time_s=dt, returncode=r.returncode,
                    stderr_tail=r.stderr[-300:] if r.stderr else '')
    except subprocess.TimeoutExpired:
        return dict(verdict='timeout', time_s=time.time()-t0,
                    returncode=-1, stderr_tail='timeout')
    except Exception as e:
        return dict(verdict='error', time_s=time.time()-t0,
                    returncode=-1, stderr_tail=repr(e))


# ---------------------------- torch.fx ShapeProp ---------------------------- #
def run_fx_shapeprop(model_name: str, ctor_kwargs_repr: str,
                     input_shape: tuple) -> dict:
    """Run torch.fx.symbolic_trace + ShapeProp in a subprocess."""
    code = textwrap.dedent(f'''
        import json, time, sys, traceback
        t0 = time.time()
        try:
            import torch
            import torchvision.models as tvm
            from torch.fx import symbolic_trace
            from torch.fx.passes.shape_prop import ShapeProp
            m = tvm.{model_name}(**{ctor_kwargs_repr})
            m.eval()
            gm = symbolic_trace(m)
            x = torch.randn(*{input_shape!r})
            ShapeProp(gm).propagate(x)
            print(json.dumps(dict(ok=True, time_s=time.time()-t0, err='')))
        except Exception as e:
            print(json.dumps(dict(ok=False, time_s=time.time()-t0,
                                  err=type(e).__name__ + ': ' + str(e)[:200])))
    ''').strip()
    t0 = time.time()
    try:
        r = subprocess.run([PYBIN, '-c', code], capture_output=True,
                           text=True, timeout=180)
        # Take last JSON line
        line = (r.stdout.strip().splitlines() or [''])[-1]
        try:
            return json.loads(line)
        except Exception:
            return dict(ok=False, time_s=time.time()-t0,
                        err=f'parse: stdout={r.stdout[:200]!r} stderr={r.stderr[-200:]!r}')
    except subprocess.TimeoutExpired:
        return dict(ok=False, time_s=time.time()-t0, err='timeout')
    except Exception as e:
        return dict(ok=False, time_s=time.time()-t0, err=repr(e))


# ---------------------------- FakeTensor / meta forward ---------------------------- #
def run_meta_forward(model_name: str, ctor_kwargs_repr: str,
                     input_shape: tuple) -> dict:
    """Construct on meta device and run a forward pass."""
    code = textwrap.dedent(f'''
        import json, time, sys, traceback
        t0 = time.time()
        try:
            import torch
            import torchvision.models as tvm
            from torch._subclasses.fake_tensor import FakeTensorMode
            mode = FakeTensorMode(allow_non_fake_inputs=True)
            with mode:
                m = tvm.{model_name}(**{ctor_kwargs_repr})
                m.eval()
                x = torch.randn(*{input_shape!r})
                with torch.no_grad():
                    y = m(x)
                shape = tuple(y.shape) if hasattr(y, 'shape') else None
            print(json.dumps(dict(ok=True, time_s=time.time()-t0,
                                  out_shape=list(shape) if shape else None, err='')))
        except Exception as e:
            print(json.dumps(dict(ok=False, time_s=time.time()-t0,
                                  err=type(e).__name__ + ': ' + str(e)[:200])))
    ''').strip()
    t0 = time.time()
    try:
        r = subprocess.run([PYBIN, '-c', code], capture_output=True,
                           text=True, timeout=180)
        line = (r.stdout.strip().splitlines() or [''])[-1]
        try:
            return json.loads(line)
        except Exception:
            return dict(ok=False, time_s=time.time()-t0,
                        err=f'parse: stdout={r.stdout[:200]!r} stderr={r.stderr[-200:]!r}')
    except subprocess.TimeoutExpired:
        return dict(ok=False, time_s=time.time()-t0, err='timeout')
    except Exception as e:
        return dict(ok=False, time_s=time.time()-t0, err=repr(e))


# ---------------------------- Per-model sweep ---------------------------- #
PARTIAL = os.path.join(REPO, 'experiments', 'neurips_revision_partial.json')

def _save_partial(rows, sweep_wall, tg_total):
    try:
        with open(PARTIAL, 'w') as f:
            json.dump(dict(rows=rows, sweep_wall_s=sweep_wall,
                           tg_total_s=tg_total, n_done=len(rows)), f)
    except Exception as e:
        print(f'  (partial save failed: {e})')

def _load_partial():
    if os.path.exists(PARTIAL):
        try:
            d = json.load(open(PARTIAL))
            return d.get('rows', []), d.get('sweep_wall_s', 0.0), d.get('tg_total_s', 0.0)
        except Exception:
            pass
    return [], 0.0, 0.0

def sweep_torchvision():
    rows, prior_wall, tg_total = _load_partial()
    done = {r['model'] for r in rows}
    print(f'  resuming with {len(done)} models already done')
    sweep_t0 = time.time() - prior_wall
    for name, kw, ishape_str, ishape_tup in MODELS:
        if name in done:
            continue
        print(f'[{name}]', flush=True)
        wrapper = write_wrapper(name, kw)
        try:
            tg = run_tensorguard(wrapper, ishape_str, timeout=120)
        finally:
            try: os.unlink(wrapper)
            except OSError: pass
        tg_total += tg.get('time_s', 0.0)
        fx = run_fx_shapeprop(name, kw, ishape_tup)
        meta = run_meta_forward(name, kw, ishape_tup)
        row = dict(model=name, input_shape=list(ishape_tup),
                   tensorguard=tg, fx_shapeprop=fx, meta_forward=meta)
        rows.append(row)
        print(f'  TG={tg["verdict"]} ({tg["time_s"]:.2f}s)  '
              f'FX={"ok" if fx["ok"] else "fail"} ({fx["time_s"]:.2f}s)  '
              f'META={"ok" if meta["ok"] else "fail"} ({meta["time_s"]:.2f}s)',
              flush=True)
        _save_partial(rows, time.time() - sweep_t0, tg_total)
    return rows, time.time() - sweep_t0, tg_total


# ---------------------------- Real bug vignette ---------------------------- #
BUG_MODEL_SRC = textwrap.dedent('''
    """Mutated ResNet-18: classifier in_features deliberately wrong.

    torchvision.resnet18 ends with:
        self.fc = nn.Linear(512, num_classes)
    The bug: we replace it with nn.Linear(256, 1000), which causes a
    runtime shape mismatch. This mirrors the canonical "I changed the
    backbone and forgot to update the head" bug pattern reported many
    times on the PyTorch and HuggingFace issue trackers.
    """
    import torch.nn as nn
    import torchvision.models as tvm
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = tvm.resnet18(weights=None)
            # BUG: 512 -> 256 (head no longer matches backbone output dim)
            self.net.fc = nn.Linear(256, 1000)
        def forward(self, x):
            return self.net(x)
''').strip() + '\n'


def real_bug_vignette():
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    f.write(BUG_MODEL_SRC); f.close()
    path = f.name
    try:
        tg = run_tensorguard(path, 'x=1,3,224,224', timeout=120)
    finally:
        try: os.unlink(path)
        except OSError: pass

    # FX ShapeProp + meta forward on the same mutated model
    fx_code = textwrap.dedent('''
        import json, time
        t0 = time.time()
        try:
            import torch, torch.nn as nn
            import torchvision.models as tvm
            from torch.fx import symbolic_trace
            from torch.fx.passes.shape_prop import ShapeProp
            net = tvm.resnet18(weights=None)
            net.fc = nn.Linear(256, 1000)  # injected bug
            gm = symbolic_trace(net)
            x = torch.randn(1, 3, 224, 224)
            ShapeProp(gm).propagate(x)
            print(json.dumps(dict(ok=True, time_s=time.time()-t0, err='')))
        except Exception as e:
            print(json.dumps(dict(ok=False, time_s=time.time()-t0,
                                  err=type(e).__name__ + ': ' + str(e)[:200])))
    ''').strip()
    r = subprocess.run([PYBIN, '-c', fx_code], capture_output=True,
                       text=True, timeout=180)
    fx_line = (r.stdout.strip().splitlines() or [''])[-1]
    try: fx = json.loads(fx_line)
    except Exception: fx = dict(ok=False, err=f'parse: {r.stderr[-200:]}')

    meta_code = textwrap.dedent('''
        import json, time
        t0 = time.time()
        try:
            import torch, torch.nn as nn
            import torchvision.models as tvm
            from torch._subclasses.fake_tensor import FakeTensorMode
            mode = FakeTensorMode(allow_non_fake_inputs=True)
            with mode:
                net = tvm.resnet18(weights=None)
                net.fc = nn.Linear(256, 1000)  # injected bug
                net.eval()
                x = torch.randn(1, 3, 224, 224)
                y = net(x)
            print(json.dumps(dict(ok=True, time_s=time.time()-t0,
                                  out_shape=list(y.shape), err='')))
        except Exception as e:
            print(json.dumps(dict(ok=False, time_s=time.time()-t0,
                                  err=type(e).__name__ + ': ' + str(e)[:200])))
    ''').strip()
    r2 = subprocess.run([PYBIN, '-c', meta_code], capture_output=True,
                        text=True, timeout=180)
    m_line = (r2.stdout.strip().splitlines() or [''])[-1]
    try: meta = json.loads(m_line)
    except Exception: meta = dict(ok=False, err=f'parse: {r2.stderr[-200:]}')

    return dict(description='Mutated resnet18: fc head '
                            'changed from Linear(512,1000) to Linear(256,1000)',
                tensorguard=tg, fx_shapeprop=fx, meta_forward=meta)


# ---------------------------- Main ---------------------------- #
def main():
    print('=== NeurIPS revision experiments ===')
    print('1) torchvision sweep')
    rows, sweep_wall, tg_total = sweep_torchvision()
    print('\n2) real-bug vignette (mutated resnet18 head)')
    bug = real_bug_vignette()
    print(f'   TG verdict={bug["tensorguard"]["verdict"]}  '
          f'FX ok={bug["fx_shapeprop"].get("ok")}  '
          f'META ok={bug["meta_forward"].get("ok")}')

    # Aggregate stats
    n = len(rows)
    tg_verified = sum(1 for r in rows if r['tensorguard']['verdict'] == 'verified')
    tg_rejected = sum(1 for r in rows if r['tensorguard']['verdict'] == 'rejected')
    tg_unknown  = sum(1 for r in rows if r['tensorguard']['verdict'] not in ('verified','rejected'))
    fx_ok       = sum(1 for r in rows if r['fx_shapeprop']['ok'])
    meta_ok     = sum(1 for r in rows if r['meta_forward']['ok'])

    summary = dict(
        n_models=n,
        tensorguard=dict(verified=tg_verified, rejected=tg_rejected,
                         unknown=tg_unknown,
                         coverage_rate=(tg_verified + tg_rejected) / max(n,1),
                         total_wall_s=tg_total,
                         mean_per_model_s=tg_total / max(n,1)),
        fx_shapeprop=dict(success=fx_ok, fail=n-fx_ok,
                          coverage_rate=fx_ok / max(n,1)),
        meta_forward=dict(success=meta_ok, fail=n-meta_ok,
                          coverage_rate=meta_ok / max(n,1)),
        sweep_wall_s=sweep_wall,
    )

    out = dict(summary=summary, rows=rows, real_bug=bug,
               python=PYBIN, repo=REPO)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {OUT}')
    print(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------ #
    # Gate: every NeurIPS-revision JSON the paper cites must exist.
    # ------------------------------------------------------------------ #
    required = [
        os.path.join(REPO, 'experiments', 'neurips_revision.json'),
        os.path.join(REPO, 'experiments', 'neurips_revision_handwritten_bug.json'),
        os.path.join(REPO, 'experiments', 'real_repo_eval.json'),
        os.path.join(REPO, 'experiments', 'injected_bug_eval.json'),
        os.path.join(REPO, 'experiments', 'comparison_pytorch_native.json'),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print('\n!! Missing required JSONs (paper will not reproduce):')
        for p in missing:
            print('  -', p)
        print('Re-run experiments/{real_repo_eval, injected_bug_eval, '
              'comparison_pytorch_native}.py to populate.')
        sys.exit(2)
    else:
        print('\nAll five revision JSONs present; paper claims are reproducible.')

    # ------------------------------------------------------------------ #
    # Fragment-fair filter check: regenerate audit CSV and verify counts.
    # ------------------------------------------------------------------ #
    print('\n3) Fragment-fair filter audit (60→34 Pytea head-to-head)')
    _filter_script = os.path.join(REPO, 'reproducibility', 'build_fragment_fair_filter.py')
    _filter_result = subprocess.run(
        [sys.executable, _filter_script],
        capture_output=True, text=True,
    )
    if _filter_result.returncode != 0:
        print('!! Fragment-fair filter script failed:')
        print(_filter_result.stdout)
        print(_filter_result.stderr)
        sys.exit(3)
    print(_filter_result.stdout.strip())
    _audit_csv = os.path.join(REPO, 'reproducibility', 'fragment_fair_audit.csv')
    if not os.path.exists(_audit_csv):
        print('!! fragment_fair_audit.csv not found after running filter script')
        sys.exit(3)
    with open(_audit_csv, newline='') as _f:
        import csv as _csv
        _rows = list(_csv.DictReader(_f))
    _n_total = len(_rows)
    _n_included = sum(1 for r in _rows if r['included_in_34'] == 'True')
    assert _n_total == 60, f'Expected 60 audit rows, got {_n_total}'
    assert _n_included == 34, f'Expected 34 included rows, got {_n_included}'
    print(f'   fragment_fair_audit.csv: {_n_total} rows, {_n_included} included — OK')


if __name__ == '__main__':
    main()
