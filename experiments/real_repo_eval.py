"""Real-repo evaluation: download a small set of real, public PyTorch
model files (raw GitHub URLs), run TensorGuard / FX ShapeProp /
FakeTensor on each, and report per-file verdict counts.

We are deliberately HONEST: when a file fails to import (because of
missing project-local dependencies), we report it as such and still
report TG's static-analysis verdict (TG operates on source, not on a
loaded module).

Cached downloads live in experiments/.cache/real_repo/.
Output: experiments/real_repo_eval.json
"""
from __future__ import annotations
import json, os, sys, time, subprocess, tempfile, textwrap, urllib.request

REPO  = '/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard'
PYBIN = '/opt/homebrew/bin/python3.11'
CACHE = os.path.join(REPO, 'experiments', '.cache', 'real_repo')
OUT   = os.path.join(REPO, 'experiments', 'real_repo_eval.json')

# (label, raw URL, model-class name to instantiate, ctor-kwargs repr,
#  input-tuple, free-form notes). We pick small, self-contained model
# files that import only torch.{nn,functional}.
FILES = [
    # nanoGPT model.py: small Transformer LM. Self-contained except for
    # @dataclass GPTConfig; uses only torch+nn.
    ('nanoGPT_model',
     'https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py',
     'GPT', "GPTConfig(block_size=64, vocab_size=50, n_layer=2, n_head=2, n_embd=64)",
     (1, 16), 'instantiated with tiny config'),
    # minGPT model.py: same author, simpler imports.
    ('minGPT_model',
     'https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py',
     'GPT', "GPT.get_default_config()",  # we patch at runtime to small dims
     (1, 16), 'patched to tiny config (vocab_size=50,n_layer=2,n_head=2,n_embd=64,block_size=64)'),
    # MAE encoder. facebookresearch/mae models_mae.py imports timm; we
    # therefore only static-analyze it with TG (FX/Meta will fail at
    # import).
    ('mae_models',
     'https://raw.githubusercontent.com/facebookresearch/mae/main/models_mae.py',
     None, None, (1, 3, 224, 224),
     'imports timm; import-only -> FX/Meta expected to fail to load'),
    # milesial/Pytorch-UNet/unet/unet_model.py + unet_parts.py.
    ('unet_model',
     'https://raw.githubusercontent.com/milesial/Pytorch-UNet/master/unet/unet_model.py',
     'UNet', '(n_channels=3, n_classes=2)',
     (1, 3, 256, 256),
     'requires unet_parts.py (also fetched into the same dir)'),
    ('unet_parts',  # supporting module
     'https://raw.githubusercontent.com/milesial/Pytorch-UNet/master/unet/unet_parts.py',
     None, None, (), 'module dependency of unet_model'),
    # A tiny illustrative Transformer file from labml-nn (transformers/mha.py).
    ('labml_mha',
     'https://raw.githubusercontent.com/labmlai/annotated_deep_learning_paper_implementations/master/labml_nn/transformers/mha.py',
     None, None, (16, 1, 64),
     'multi-head attention; instantiation requires labml; static analyze only'),
]

USER_AGENT = 'tensorguard-eval/1.0 (research-eval; non-commercial)'


def _download(url: str, path: str, timeout: int = 30) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(path, 'wb') as f:
        f.write(data)


def fetch_all() -> dict:
    """Returns label -> local path (cached)."""
    out = {}
    for label, url, *_ in FILES:
        local = os.path.join(CACHE, f'{label}.py')
        try:
            _download(url, local)
            out[label] = dict(path=local, url=url, ok=True,
                              bytes=os.path.getsize(local))
        except Exception as e:
            out[label] = dict(path=local, url=url, ok=False,
                              err=type(e).__name__ + ': ' + str(e)[:200])
    return out


# ---------------------------- TensorGuard ---------------------------- #
def run_tensorguard(path: str, input_shape_tuple: tuple,
                    timeout: int = 120) -> dict:
    if not input_shape_tuple:
        return dict(verdict='skip', reason='no input shape (helper module)')
    ishape = 'x=' + ','.join(str(d) for d in input_shape_tuple)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH','')}"
    t0 = time.time()
    try:
        r = subprocess.run([PYBIN, '-m', 'src.cli.main', 'verify', path,
                            '--input-shape', ishape, '--format', 'json',
                            '--no-device-check', '--no-phase-check'],
                           cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=timeout)
        dt = time.time() - t0
        status = None
        bugs = []
        try:
            j = json.loads(r.stdout)
            status = j.get('status')
            bugs = j.get('bugs', []) or []
        except Exception:
            pass
        # Honest classification:
        #  - shape bugs raised   -> rejected
        #  - no shape bugs, abstained -> abstain (out-of-fragment construct)
        #  - no shape bugs, no abstention -> verified
        shape_bugs = [b for b in bugs
                      if 'SHAPE' in (b.get('message','') or '').upper()]
        abstained = bool(j.get('abstained', False)) if isinstance(j, dict) else False
        opaque = int(j.get('opaque_layer_count', 0)) if isinstance(j, dict) else 0
        if shape_bugs:
            verdict = 'rejected'
        elif abstained:
            verdict = 'abstain'
        elif r.returncode in (0, 1):
            verdict = 'verified'
        else:
            verdict = 'unknown'
        return dict(verdict=verdict, time_s=dt, returncode=r.returncode,
                    n_bugs=len(bugs), n_shape_bugs=len(shape_bugs),
                    abstained=abstained, opaque_layer_count=opaque,
                    first_bug=(bugs[0].get('message','')[:200] if bugs else ''),
                    first_shape_bug=(shape_bugs[0].get('message','')[:200]
                                     if shape_bugs else ''),
                    stderr_tail=r.stderr[-200:] if r.stderr else '')
    except subprocess.TimeoutExpired:
        return dict(verdict='timeout', time_s=time.time()-t0)
    except Exception as e:
        return dict(verdict='error', time_s=time.time()-t0, err=repr(e))


# ---------------------------- Optional dynamic instantiation ---------------------------- #
INSTANTIATE_TPL = textwrap.dedent('''
    import json, time, sys, os, importlib.util, traceback
    SRC = {src!r}
    CLS = {cls!r}
    CTOR = {ctor!r}
    ISHAPE = {ishape!r}

    def _load():
        spec = importlib.util.spec_from_file_location("real_repo_mod", SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["real_repo_mod"] = mod
        # Add the file's dir to sys.path so sibling modules (e.g.
        # unet_parts) are importable.
        sys.path.insert(0, os.path.dirname(SRC))
        spec.loader.exec_module(mod)
        return mod

    def _build(mod):
        cls = getattr(mod, CLS)
        # minGPT special-case: patch CTOR to small dims.
        if "minGPT" in SRC:
            cfg = cls.get_default_config()
            cfg.model_type = None
            cfg.n_layer = 2; cfg.n_head = 2; cfg.n_embd = 64
            cfg.vocab_size = 50; cfg.block_size = 64
            return cls(cfg)
        # nanoGPT special-case: import GPTConfig and instantiate
        if "nanoGPT" in SRC:
            cfg = mod.GPTConfig(block_size=64, vocab_size=50, n_layer=2,
                                n_head=2, n_embd=64)
            return cls(cfg)
        # Generic CTOR string
        return eval("cls" + CTOR.replace(CLS, "cls") if CTOR.startswith(CLS) else "cls" + CTOR,
                    {{"cls": cls, "mod": mod}})

    out = dict(import_ok=False, build_ok=False, fx_ok=False, meta_ok=False,
               err='', t_import=0.0, t_build=0.0, t_fx=0.0, t_meta=0.0)
    try:
        t0 = time.time(); mod = _load(); out['t_import'] = time.time() - t0
        out['import_ok'] = True
    except Exception as e:
        out['err'] = 'import: ' + type(e).__name__ + ': ' + str(e)[:240]
        print(json.dumps(out)); sys.exit(0)

    try:
        t0 = time.time(); m = _build(mod); out['t_build'] = time.time() - t0
        out['build_ok'] = True
    except Exception as e:
        out['err'] = 'build: ' + type(e).__name__ + ': ' + str(e)[:240]
        print(json.dumps(out)); sys.exit(0)

    # Convert input shape into a tensor; integer-LM models want LongTensor.
    import torch
    is_long = ("GPT" in CLS) or ("minGPT" in SRC) or ("nanoGPT" in SRC)
    def _mkx():
        if is_long:
            return torch.randint(0, 50, ISHAPE)
        return torch.randn(*ISHAPE)

    # FX ShapeProp
    try:
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
        m.eval()
        t0 = time.time()
        gm = symbolic_trace(m)
        ShapeProp(gm).propagate(_mkx())
        out['t_fx'] = time.time() - t0
        out['fx_ok'] = True
    except Exception as e:
        out['err'] = (out['err'] + ' | fx: ' if out['err'] else 'fx: ') \
                     + type(e).__name__ + ': ' + str(e)[:200]

    # FakeTensor forward
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        t0 = time.time()
        with mode:
            x = _mkx()
            with torch.no_grad():
                y = m(x)
        out['t_meta'] = time.time() - t0
        out['meta_ok'] = True
    except Exception as e:
        out['err'] = (out['err'] + ' | meta: ' if out['err'] else 'meta: ') \
                     + type(e).__name__ + ': ' + str(e)[:200]

    print(json.dumps(out))
''').strip()


def run_dynamic(path: str, cls: str | None, ctor: str | None,
                ishape: tuple) -> dict:
    if cls is None or not ishape:
        return dict(import_ok=False, note='helper module / no class')
    code = INSTANTIATE_TPL.format(src=path, cls=cls, ctor=ctor or '()',
                                  ishape=ishape)
    try:
        r = subprocess.run([PYBIN, '-c', code], capture_output=True,
                           text=True, timeout=300)
        line = (r.stdout.strip().splitlines() or [''])[-1]
        try:
            return json.loads(line)
        except Exception:
            return dict(import_ok=False, build_ok=False,
                        err=f'parse: stdout={r.stdout[-200:]!r} stderr={r.stderr[-200:]!r}')
    except subprocess.TimeoutExpired:
        return dict(import_ok=False, err='timeout')


def main():
    print('=== Real-repo eval ===')
    files = fetch_all()
    rows = []
    for label, url, cls, ctor, ishape, notes in FILES:
        meta = files.get(label, {})
        if not meta.get('ok'):
            rows.append(dict(label=label, url=url, fetched=False,
                             err=meta.get('err','?')))
            print(f'[{label}] fetch FAILED: {meta.get("err","?")[:160]}')
            continue
        path = meta['path']
        print(f'[{label}] {path}', flush=True)
        tg = run_tensorguard(path, ishape)
        dyn = run_dynamic(path, cls, ctor, ishape) if cls else \
              dict(note='static-only', import_ok=False, build_ok=False,
                   fx_ok=False, meta_ok=False)
        rows.append(dict(label=label, url=url, path=path, fetched=True,
                         bytes=meta['bytes'], notes=notes,
                         input_shape=list(ishape) if ishape else None,
                         tensorguard=tg, dynamic=dyn))
        print(f'  TG={tg.get("verdict")} '
              f'IMPORT={dyn.get("import_ok")} BUILD={dyn.get("build_ok")} '
              f'FX={dyn.get("fx_ok")} META={dyn.get("meta_ok")}')
        if tg.get('first_bug'):
            print(f'  TG bug: {tg["first_bug"][:160]}')

    # Aggregate
    n = len([r for r in rows if r.get('fetched')])
    tg_verified = sum(1 for r in rows if r.get('tensorguard', {}).get('verdict') == 'verified')
    tg_rejected = sum(1 for r in rows if r.get('tensorguard', {}).get('verdict') == 'rejected')
    tg_abstain  = sum(1 for r in rows if r.get('tensorguard', {}).get('verdict') == 'abstain')
    tg_unknown  = sum(1 for r in rows if r.get('tensorguard', {}).get('verdict') in ('unknown', 'error', 'timeout'))
    tg_skipped  = sum(1 for r in rows if r.get('tensorguard', {}).get('verdict') == 'skip')
    fx_ok = sum(1 for r in rows if r.get('dynamic', {}).get('fx_ok'))
    meta_ok = sum(1 for r in rows if r.get('dynamic', {}).get('meta_ok'))
    import_ok = sum(1 for r in rows if r.get('dynamic', {}).get('import_ok'))

    summary = dict(
        n_files_fetched=n, n_files_total=len(FILES),
        tensorguard=dict(verified=tg_verified, rejected=tg_rejected,
                         abstain=tg_abstain,
                         unknown=tg_unknown, skipped=tg_skipped),
        dynamic=dict(import_ok=import_ok, fx_ok=fx_ok, meta_ok=meta_ok),
    )
    out = dict(summary=summary, rows=rows, python=PYBIN, repo=REPO)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {OUT}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
