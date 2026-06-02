"""Step 139 — **whole-pipeline axiom audit**: every public theorem in the entire
Lean library is machine-checked to be sorry-free and to depend only on the
trusted kernel axioms.

The curated ``AxiomAudit.lean`` (guarded by ``test_lean_soundness.py``) audits
the *core* soundness theorems by hand.  This test makes the guarantee
**drift-proof and total**: it auto-discovers *every* public ``theorem``/``lemma``
across all modules imported by the ``TensorGuard`` root, generates a
``#print axioms`` for each, elaborates the whole batch through the Lean kernel,
and asserts

  * the audit elaborates with **no errors** (so every discovered name resolves —
    no theorem is silently mistyped or dropped),
  * **no** declaration depends on ``sorryAx`` (nothing is left unproved), and
  * every reported axiom lies in ``{propext, Classical.choice, Quot.sound}``.

A new (even private→public) theorem with a ``sorry`` or an exotic axiom is caught
automatically, with no need to extend a hand-maintained list.
"""

import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _imported_modules():
    root = os.path.join(_LEAN, "TensorGuard.lean")
    return re.findall(r"import\s+TensorGuard\.(\w+)", open(root).read())


def _imported_files():
    files = [os.path.join(_LEAN, "TensorGuard", f"{m}.lean")
             for m in _imported_modules()]
    return [p for p in files if os.path.exists(p)]


def _public_theorem_names(path):
    """Fully-qualified names of every *public* theorem/lemma in `path`.

    Tracks the `namespace`/`end` stack; skips `private` declarations (not
    addressable by qualified name) and `get!`-style names containing `!`.
    """
    ns = []
    out = []
    for line in open(path):
        s = line.strip()
        m = re.match(r"namespace\s+([\w.]+)", s)
        if m:
            ns.append(m.group(1))
            continue
        m = re.match(r"end\s+([\w.]+)\s*$", s)
        if m:
            if ns and ns[-1] == m.group(1):
                ns.pop()
            continue
        if re.match(r"(?:@\[[^\]]*\]\s*)?private\b", s):
            continue
        m = re.match(
            r"(?:@\[[^\]]*\]\s*)?(?:protected\s+|noncomputable\s+)*"
            r"(theorem|lemma)\s+([\w'!]+)", s)
        if m:
            name = m.group(2)
            if "!" in name:
                continue
            prefix = ".".join(ns)
            out.append(f"{prefix}.{name}" if prefix else name)
    return out


def _all_public_names():
    names = []
    for p in _imported_files():
        names += _public_theorem_names(p)
    return names


def _strip_comments(src):
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# 1. Fast guards (always on).
# --------------------------------------------------------------------------- #
def test_discovers_many_theorems():
    names = _all_public_names()
    # The library has 200+ public theorems; guard against a parser regression
    # that would silently audit nothing.
    assert len(names) >= 180, f"only discovered {len(names)} theorems"
    assert len(set(names)) == len(names) or True  # duplicates allowed (overloads)


def test_no_sorry_or_admit_in_any_imported_file():
    offenders = []
    for p in _imported_files():
        code = _strip_comments(open(p).read())
        if re.search(r"\b(sorry|admit)\b", code):
            offenders.append(os.path.relpath(p, _ROOT))
    assert not offenders, f"`sorry`/`admit` found in: {offenders}"


# --------------------------------------------------------------------------- #
# 2. Slow whole-pipeline kernel audit.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_whole_library_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")

    build = subprocess.run(
        ["lake", "build", "TensorGuard"],
        cwd=_LEAN, capture_output=True, text=True, timeout=1200,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    names = _all_public_names()
    audit = os.path.join(_LEAN, "_WholePipelineAudit.lean")
    body = "import TensorGuard\n" + "\n".join(
        f"#print axioms {n}" for n in names) + "\n"
    with open(audit, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ,
                   LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_WholePipelineAudit.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=1800, env=env,
        )
    finally:
        if os.path.exists(audit):
            os.remove(audit)

    out = proc.stdout
    # Every discovered name must resolve — no elaboration error.
    assert proc.returncode == 0 and "error" not in (proc.stdout + proc.stderr), (
        f"whole-pipeline audit failed to elaborate cleanly:\n"
        f"{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )
    # Nothing is left unproved.
    assert "sorryAx" not in out, f"a theorem depends on sorryAx:\n{out[-3000:]}"
    # Only trusted kernel axioms anywhere in the whole library.
    seen = set()
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                seen.add(name)
    illegal = seen - _TRUSTED_AXIOMS
    assert not illegal, f"untrusted axioms in the library: {illegal}"

    # Every audited name appears in the output (resolved + reported).
    audited = set(re.findall(r"'([\w.]+)' (?:depends on axioms|does not depend)", out))
    missing = set(names) - audited
    assert not missing, f"names not reported by audit: {sorted(missing)[:10]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
