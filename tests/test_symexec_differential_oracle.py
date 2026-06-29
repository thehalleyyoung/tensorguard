"""Roadmap **step 10 — differential oracle harness** tests.

Runs the :mod:`tests._differential` harness over the whole model fixture corpus
and asserts the derived contract is a **sound subset** of the real torch
``state_dict`` for every one — the empirical proof of partiality-correctness.

Layers of testing:

* **Teeth** — pure unit tests of :func:`subset_verdict` proving it *catches*
  wrong-shape and phantom emissions (so a green corpus run is meaningful).
* **Corpus** — every :data:`DIFFERENTIAL_FIXTURES` entry (incl. soundness
  stressors: ``affine=False`` norms, grouped/asymmetric convs, 1/2/3-d convs,
  deep nesting) must yield ``is_sound``; zero unsound emissions in aggregate.
* **Property-based** — Hypothesis generates random small ``nn.Module`` graphs
  from a layer grammar; each must be a sound subset.  This searches far beyond
  the hand-written fixtures for any config where the deriver would emit a
  parameter torch does not register (or with a wrong shape).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.symexec.model_contract import ModelContract  # noqa: E402

from _torch_oracle import DIFFERENTIAL_FIXTURES  # noqa: E402
from _differential import (  # noqa: E402
    Mismatch,
    SubsetVerdict,
    assert_sound_subset,
    differential_verdict,
    subset_verdict,
)


def _contract(params) -> ModelContract:
    return ModelContract(
        model_class="M",
        construction="M()",
        params=dict(params),
        abstained=(),
        resolved_layers=0,
    )


# --------------------------------------------------------------------------- #
# Teeth — the verdict logic must catch unsoundness (pure, torch-free).          #
# --------------------------------------------------------------------------- #
def test_sound_subset_accepts_exact_and_partial():
    oracle = {"a.weight": (8, 4), "a.bias": (8,), "b.weight": (3, 3)}
    # Emit a strict, correct subset (omit b.weight -> that's just partiality).
    v = subset_verdict(_contract({"a.weight": (8, 4), "a.bias": (8,)}), oracle)
    assert v.is_sound
    assert v.missing == ("b.weight",)
    assert v.fraction == pytest.approx(2 / 3)


def test_sound_subset_catches_wrong_shape():
    oracle = {"a.weight": (8, 4)}
    v = subset_verdict(_contract({"a.weight": (4, 8)}), oracle)
    assert not v.is_sound
    assert v.mismatches == (Mismatch("a.weight", (4, 8), (8, 4)),)
    assert "UNSOUND" in v.describe()


def test_sound_subset_catches_phantom_param():
    oracle = {"a.weight": (8, 4)}
    v = subset_verdict(_contract({"a.weight": (8, 4), "ghost.bias": (8,)}), oracle)
    assert not v.is_sound
    assert v.mismatches == (Mismatch("ghost.bias", (8,), None),)


def test_verdict_is_frozen():
    v = subset_verdict(_contract({}), {})
    assert isinstance(v, SubsetVerdict)
    assert v.is_sound and v.fraction == 1.0
    with pytest.raises(Exception):
        v.emitted = 5


# --------------------------------------------------------------------------- #
# Corpus — every fixture is a sound subset of its torch state_dict.             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fx", DIFFERENTIAL_FIXTURES, ids=lambda f: f.name)
def test_fixture_is_sound_subset(fx):
    pytest.importorskip("torch")
    v = assert_sound_subset(fx.source, fx.construction)
    assert v.registered > 0  # the fixture actually has parameters


def test_whole_corpus_zero_unsound_emissions():
    pytest.importorskip("torch")
    total_emit = total_reg = total_match = 0
    bad = []
    for fx in DIFFERENTIAL_FIXTURES:
        v = differential_verdict(fx.source, fx.construction)
        if not v.is_sound:
            bad.append(f"{fx.name}: {v.describe()}")
        total_emit += v.emitted
        total_reg += v.registered
        total_match += len(v.matched)
    assert not bad, "\n".join(bad)
    # Sanity: across the corpus we soundly emit a substantial number of params.
    assert total_match == total_emit  # every emitted param matched (soundness)
    assert total_reg >= total_match


def test_stressors_emit_correct_partial_sets():
    # Pin the *specific* soundness-critical behaviour: affine=False / no-stats
    # norms must omit exactly the right params, and grouped conv divides in-ch.
    pytest.importorskip("torch")
    from src.symexec import derive_model_contract
    from _torch_oracle import state_dict_shapes

    src = next(f for f in DIFFERENTIAL_FIXTURES if f.name == "stress_norm_conv")
    oracle = state_dict_shapes(src.source, src.construction)
    mc = derive_model_contract(src.source, src.construction)
    # affine=False BN: running stats present, weight/bias absent.
    assert "bn_no_affine.weight" not in mc.params
    assert "bn_no_affine.running_mean" in mc.params
    # track_running_stats=False BN: weight/bias present, running stats absent.
    assert "bn_no_stats.weight" in mc.params
    assert "bn_no_stats.running_mean" not in mc.params
    # elementwise_affine=False LN: no params at all.
    assert not any(k.startswith("ln_no_affine") for k in mc.params)
    # grouped conv: weight (out, in//groups, *k).
    assert mc.params["conv_groups.weight"] == (8, 2, 3, 3)
    # asymmetric-kernel, bias=False conv.
    assert mc.params["conv_tuple_k.weight"] == (6, 3, 3, 5)
    assert "conv_tuple_k.bias" not in mc.params
    # And the whole thing is still a sound subset.
    assert subset_verdict(mc, oracle).is_sound


# --------------------------------------------------------------------------- #
# Property-based — random model graphs are always sound subsets.                #
# --------------------------------------------------------------------------- #
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    _HAS_HYPOTHESIS = False


if _HAS_HYPOTHESIS:

    _dim = st.integers(min_value=1, max_value=8)
    _bias = st.booleans()

    @st.composite
    def _layer(draw):
        """A single leaf-layer source fragment (one statement, no name)."""
        kind = draw(st.sampled_from(
            ["linear", "embedding", "layernorm", "conv2d", "batchnorm2d"]
        ))
        if kind == "linear":
            i, o = draw(_dim), draw(_dim)
            b = draw(_bias)
            return f"nn.Linear({i}, {o}, bias={b})"
        if kind == "embedding":
            n, d = draw(_dim), draw(_dim)
            return f"nn.Embedding({n}, {d})"
        if kind == "layernorm":
            d = draw(_dim)
            aff = draw(_bias)
            return f"nn.LayerNorm({d}, elementwise_affine={aff})"
        if kind == "conv2d":
            g = draw(st.sampled_from([1, 2]))
            mult_in = draw(st.integers(min_value=1, max_value=3))
            mult_out = draw(st.integers(min_value=1, max_value=3))
            ic, oc = g * mult_in, g * mult_out
            k = draw(st.sampled_from(["3", "(3, 5)", "1"]))
            b = draw(_bias)
            return f"nn.Conv2d({ic}, {oc}, kernel_size={k}, groups={g}, bias={b})"
        # batchnorm2d
        n = draw(_dim)
        aff = draw(_bias)
        trs = draw(_bias)
        return f"nn.BatchNorm2d({n}, affine={aff}, track_running_stats={trs})"

    @st.composite
    def _model(draw):
        layers = draw(st.lists(_layer(), min_size=1, max_size=5))
        body = "\n".join(
            f"        self.m{i} = {frag}" for i, frag in enumerate(layers)
        )
        source = (
            "import torch.nn as nn\n\n"
            "class M(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"{body}\n"
        )
        return source

    @settings(
        max_examples=80,
        deadline=None,
        derandomize=True,
        database=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(source=_model())
    def test_random_models_are_sound_subsets(source):
        pytest.importorskip("torch")
        v = differential_verdict(source, "M()")
        assert v.is_sound, f"{source}\n{v.describe()}"
