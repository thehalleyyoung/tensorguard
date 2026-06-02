"""A typed, shrinkable module-AST DSL for property-based testing (Step 114).

Step 113 generated random *source strings* directly. That is fine for a one-shot
differential sweep, but it does not give the two things a reviewer-grade
property-based campaign needs:

  1. a structured, *compositional* representation of a whole ``nn.Module`` that a
     generator (Hypothesis or a seeded enumerator) can build up from typed pieces
     -- a small algebra of layers rather than ad-hoc string templates; and
  2. **shrinking**: when a property fails, the ability to reduce the failing
     module to a *minimal* counterexample (fewest layers, smallest dimensions)
     so a human reads a two-line module instead of a forty-line one.

This module provides exactly that. A :class:`ModuleAST` is a real data structure
(an input regime plus an ordered list of typed :class:`Layer` nodes) that

  * renders deterministically to runnable PyTorch source (:func:`render`),
  * can be exercised by Hypothesis via :func:`module_asts` (a ``@composite``
    strategy that builds full module ASTs, not just individual ops), and
  * can be shrunk to a *locally minimal* counterexample by a deterministic
    delta-debugging shrinker (:func:`shrink_to_minimal`) that is independent of
    Hypothesis internals and therefore reproducible byte-for-byte.

The DSL deliberately covers two shape regimes and the transitions between them:

  * **vec** regime: a 2D ``(batch, features)`` tensor flowing through
    ``nn.Linear`` / ``nn.ReLU`` layers;
  * **img** regime: a 4D ``(batch, channels, side, side)`` tensor flowing through
    ``nn.Conv2d`` / ``nn.ReLU`` layers, with a ``Flatten`` node transitioning the
    tensor into the vec regime (after which ``Linear`` layers may follow).

A module is *clean* iff every declared in-dimension matches the actual incoming
dimension at that point; otherwise eager PyTorch raises. Because the in-dims are
declared independently of the flow, a generated chain is compatible-or-not by
chance, which is precisely what we want to stress a *sound* verifier with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Typed layer nodes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Linear:
    in_features: int
    out_features: int


@dataclass(frozen=True)
class Conv2d:
    in_channels: int
    out_channels: int
    kernel: int  # odd; padding == kernel // 2 so the spatial side is preserved


@dataclass(frozen=True)
class ReLU:
    pass


@dataclass(frozen=True)
class Flatten:
    pass


Layer = object  # Linear | Conv2d | ReLU | Flatten


@dataclass(frozen=True)
class ModuleAST:
    """A full module: an input regime/shape plus an ordered list of layers."""

    regime: str  # "vec" or "img"
    input_shape: Tuple[int, ...]
    layers: Tuple[Layer, ...]


# ----------------------------------------------------------------------------
# Rendering to runnable PyTorch source
# ----------------------------------------------------------------------------

_IMPORTS = "import torch\nimport torch.nn as nn\n\n"


def render(ast: ModuleAST) -> Tuple[str, dict]:
    """Render an AST to (source, input_shapes). Pure and deterministic."""

    init_lines: List[str] = []
    fwd_lines: List[str] = []
    idx = 0
    for layer in ast.layers:
        if isinstance(layer, Linear):
            init_lines.append(
                f"        self.l{idx} = nn.Linear("
                f"{layer.in_features}, {layer.out_features})\n"
            )
            fwd_lines.append(f"        x = self.l{idx}(x)\n")
            idx += 1
        elif isinstance(layer, Conv2d):
            init_lines.append(
                f"        self.c{idx} = nn.Conv2d("
                f"{layer.in_channels}, {layer.out_channels}, "
                f"{layer.kernel}, padding={layer.kernel // 2})\n"
            )
            fwd_lines.append(f"        x = self.c{idx}(x)\n")
            idx += 1
        elif isinstance(layer, ReLU):
            fwd_lines.append("        x = torch.relu(x)\n")
        elif isinstance(layer, Flatten):
            fwd_lines.append("        x = x.flatten(1)\n")
        else:  # pragma: no cover - defensive
            raise TypeError(f"unknown layer {layer!r}")

    if not init_lines:
        init_lines.append("        self._noop = nn.Identity()\n")
    if not fwd_lines:
        fwd_lines.append("        x = x\n")

    source = (
        _IMPORTS
        + "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "".join(init_lines)
        + "    def forward(self, x):\n"
        + "".join(fwd_lines)
        + "        return x\n"
    )
    return source, {"x": tuple(ast.input_shape)}


# ----------------------------------------------------------------------------
# Ground-truth oracle (eager PyTorch) + size metric
# ----------------------------------------------------------------------------


def torch_runs_clean(ast: ModuleAST) -> bool:
    """True iff a real eager forward pass executes without raising."""

    import torch

    source, shapes = render(ast)
    ns: dict = {}
    try:
        exec(compile(source, "<module_ast>", "exec"), ns)
        net = ns["Net"]()
        net.eval()
        inputs = [torch.randn(*s) for s in shapes.values()]
        with torch.no_grad():
            net(*inputs)
        return True
    except Exception:
        return False


def size(ast: ModuleAST) -> Tuple[int, int]:
    """A shrink ordering key: (#layers, sum of all integer dims)."""

    total = sum(int(v) for v in ast.input_shape)
    for layer in ast.layers:
        if isinstance(layer, Linear):
            total += layer.in_features + layer.out_features
        elif isinstance(layer, Conv2d):
            total += layer.in_channels + layer.out_channels + layer.kernel
    return (len(ast.layers), total)


# ----------------------------------------------------------------------------
# Deterministic delta-debugging shrinker
# ----------------------------------------------------------------------------

_DIM_LADDER = (1, 2, 3, 4, 5, 8)


def _smaller_dims(value: int) -> List[int]:
    """Candidate strictly-smaller replacement dimensions, smallest first."""

    return [d for d in _DIM_LADDER if d < value]


def _layer_dim_reductions(layer: Layer):
    """Yield (replacement_layer,) candidates that shrink one dim of *layer*."""

    out = []
    if isinstance(layer, Linear):
        for d in _smaller_dims(layer.in_features):
            out.append(replace(layer, in_features=d))
        for d in _smaller_dims(layer.out_features):
            out.append(replace(layer, out_features=d))
    elif isinstance(layer, Conv2d):
        for d in _smaller_dims(layer.in_channels):
            out.append(replace(layer, in_channels=d))
        for d in _smaller_dims(layer.out_channels):
            out.append(replace(layer, out_channels=d))
        if layer.kernel > 1:
            out.append(replace(layer, kernel=1))
    return out


def _input_reductions(ast: ModuleAST) -> List[ModuleAST]:
    out = []
    for i, v in enumerate(ast.input_shape):
        for d in _smaller_dims(v):
            new_shape = ast.input_shape[:i] + (d,) + ast.input_shape[i + 1 :]
            out.append(replace(ast, input_shape=new_shape))
    return out


def shrink_to_minimal(
    ast: ModuleAST,
    fails: Callable[[ModuleAST], bool],
    max_steps: int = 10_000,
) -> ModuleAST:
    """Return a *locally minimal* AST still satisfying ``fails``.

    Greedy delta debugging: repeatedly try (1) deleting a layer, then (2)
    shrinking any single integer dimension, then (3) shrinking the input shape,
    always keeping the first reduction that preserves ``fails``. Terminates at a
    1-minimal fixed point where no single reduction preserves the property. Fully
    deterministic: candidate moves are enumerated in a fixed order, so the same
    starting counterexample always shrinks to the same minimal witness.
    """

    if not fails(ast):
        raise ValueError("starting AST does not satisfy the failure predicate")

    cur = ast
    steps = 0
    changed = True
    while changed and steps < max_steps:
        changed = False

        # (1) delete a layer (smallest index first)
        for i in range(len(cur.layers)):
            cand = replace(cur, layers=cur.layers[:i] + cur.layers[i + 1 :])
            steps += 1
            if fails(cand):
                cur = cand
                changed = True
                break
        if changed:
            continue

        # (2) shrink a single layer dimension
        for i, layer in enumerate(cur.layers):
            done = False
            for repl in _layer_dim_reductions(layer):
                cand = replace(
                    cur, layers=cur.layers[:i] + (repl,) + cur.layers[i + 1 :]
                )
                steps += 1
                if fails(cand):
                    cur = cand
                    changed = True
                    done = True
                    break
            if done:
                break
        if changed:
            continue

        # (3) shrink the input shape
        for cand in _input_reductions(cur):
            steps += 1
            if fails(cand):
                cur = cand
                changed = True
                break

    return cur


# ----------------------------------------------------------------------------
# A seeded, structured enumerator over the DSL (reproducible, no Hypothesis)
# ----------------------------------------------------------------------------

_VEC_FEATS = (4, 8, 16, 32)
_IMG_CH = (1, 3, 8)
_IMG_SIDE = (8, 16)
_KERNELS = (1, 3, 5)


def random_module_ast(rng) -> ModuleAST:
    """Generate one structured module AST from a ``random.Random`` source.

    Declared in-dims are drawn *independently* of the actual flow (with a bias
    toward the matching value), so each adjacent boundary is compatible or not by
    chance -- exactly the distribution needed to stress a sound verifier.
    """

    regime = rng.choice(["vec", "img"])
    layers: List[Layer] = []
    if regime == "vec":
        f0 = rng.choice(_VEC_FEATS)
        cur = f0
        depth = rng.randint(1, 4)
        for _ in range(depth):
            out = rng.choice(_VEC_FEATS)
            decl_in = cur if rng.random() < 0.7 else rng.choice(_VEC_FEATS)
            layers.append(Linear(decl_in, out))
            if rng.random() < 0.4:
                layers.append(ReLU())
            cur = out
        return ModuleAST("vec", (4, f0), tuple(layers))

    # img regime
    c0 = rng.choice(_IMG_CH)
    side = rng.choice(_IMG_SIDE)
    cur = c0
    depth = rng.randint(1, 3)
    for _ in range(depth):
        out = rng.choice(_IMG_CH + (4, 16))
        k = rng.choice(_KERNELS)
        decl_in = cur if rng.random() < 0.7 else rng.choice(_IMG_CH + (4, 16))
        layers.append(Conv2d(decl_in, out, k))
        if rng.random() < 0.4:
            layers.append(ReLU())
        cur = out
    if rng.random() < 0.6:
        layers.append(Flatten())
        flat = cur * side * side
        decl = flat if rng.random() < 0.55 else rng.choice(_VEC_FEATS) * side
        layers.append(Linear(decl, rng.choice((5, 10))))
    return ModuleAST("img", (2, c0, side, side), tuple(layers))


# ----------------------------------------------------------------------------
# Hypothesis strategies for full module ASTs
# ----------------------------------------------------------------------------

try:  # Hypothesis is a test-only dependency; keep import optional.
    from hypothesis import strategies as st

    @st.composite
    def _vec_module(draw) -> ModuleAST:
        f0 = draw(st.sampled_from(_VEC_FEATS))
        depth = draw(st.integers(min_value=1, max_value=4))
        layers: List[Layer] = []
        cur = f0
        for _ in range(depth):
            out = draw(st.sampled_from(_VEC_FEATS))
            # Bias toward the matching in-dim, but allow mismatches.
            decl_in = draw(
                st.one_of(st.just(cur), st.sampled_from(_VEC_FEATS))
            )
            layers.append(Linear(decl_in, out))
            if draw(st.booleans()):
                layers.append(ReLU())
            cur = out
        return ModuleAST("vec", (4, f0), tuple(layers))

    @st.composite
    def _img_module(draw) -> ModuleAST:
        c0 = draw(st.sampled_from(_IMG_CH))
        side = draw(st.sampled_from(_IMG_SIDE))
        depth = draw(st.integers(min_value=1, max_value=3))
        layers: List[Layer] = []
        cur = c0
        for _ in range(depth):
            out = draw(st.sampled_from(_IMG_CH + (4, 16)))
            k = draw(st.sampled_from(_KERNELS))
            decl_in = draw(
                st.one_of(st.just(cur), st.sampled_from(_IMG_CH + (4, 16)))
            )
            layers.append(Conv2d(decl_in, out, k))
            if draw(st.booleans()):
                layers.append(ReLU())
            cur = out
        if draw(st.booleans()):
            layers.append(Flatten())
            flat = cur * side * side
            decl = draw(
                st.one_of(
                    st.just(flat),
                    st.builds(lambda f: f * side, st.sampled_from(_VEC_FEATS)),
                )
            )
            layers.append(Linear(decl, draw(st.sampled_from((5, 10)))))
        return ModuleAST("img", (2, c0, side, side), tuple(layers))

    def module_asts():
        """A Hypothesis strategy generating *full module ASTs* across regimes."""

        return st.one_of(_vec_module(), _img_module())

    HAS_HYPOTHESIS = True
except Exception:  # pragma: no cover - Hypothesis always present in this env
    HAS_HYPOTHESIS = False

    def module_asts():  # type: ignore
        raise RuntimeError("hypothesis is not installed")
