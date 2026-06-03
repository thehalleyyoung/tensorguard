"""Step 234 -- Lean-checked distribution batch/event/log_prob shape rules.

``lean/TensorGuard/Distributions.lean`` mechanizes the concrete shape algebra
behind ``src.distributions_verify`` for Normal, Categorical,
MultivariateNormal, Independent, and the identity/reshape fragment of
TransformedDistribution.  The cases below are theorem-shaped and checked against
the Python verifier plus live ``torch.distributions``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")
import torch.distributions as D  # noqa: E402

from src.distributions_verify import (  # noqa: E402
    TransformSpec,
    verify_distribution,
    verify_log_prob,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "Distributions.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.Distributions.bcDim_same",
    "TensorGuard.Distributions.bcDim_one_left",
    "TensorGuard.Distributions.bcDim_one_right",
    "TensorGuard.Distributions.bcDim_incompatible_example",
    "TensorGuard.Distributions.broadcast_example",
    "TensorGuard.Distributions.broadcast_incompatible_example",
    "TensorGuard.Distributions.normal_broadcast_output",
    "TensorGuard.Distributions.normal_bad_broadcast_rejected",
    "TensorGuard.Distributions.categorical_batch_drops_category_dim",
    "TensorGuard.Distributions.categorical_empty_rank_rejected",
    "TensorGuard.Distributions.categorical_zero_categories_rejected",
    "TensorGuard.Distributions.mvn_batch_event_output",
    "TensorGuard.Distributions.mvn_matrix_square_rejected",
    "TensorGuard.Distributions.mvn_event_mismatch_rejected",
    "TensorGuard.Distributions.mvn_batch_broadcast_rejected",
    "TensorGuard.Distributions.independent_moves_batch_to_event",
    "TensorGuard.Distributions.independent_preserves_when_zero",
    "TensorGuard.Distributions.independent_too_many_rejected",
    "TensorGuard.Distributions.normal_logProb_broadcasts_value",
    "TensorGuard.Distributions.categorical_logProb_drops_no_event",
    "TensorGuard.Distributions.mvn_logProb_drops_event_dim",
    "TensorGuard.Distributions.logProb_bad_value_broadcast_rejected",
    "TensorGuard.Distributions.reshapeShape_output",
    "TensorGuard.Distributions.reshapeShape_wrong_suffix_rejected",
    "TensorGuard.Distributions.reshapeShape_numel_mismatch_rejected",
    "TensorGuard.Distributions.transformed_identity_preserves_shape",
    "TensorGuard.Distributions.transformed_reshape_event_shape",
    "TensorGuard.Distributions.transformed_reshape_reinterprets_batch",
    "TensorGuard.Distributions.transformed_composed_reshape_identity",
    "TensorGuard.Distributions.transformed_wrong_domain_rejected",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _zeros(shape, *, dtype=torch.float32):
    return torch.zeros(tuple(shape), dtype=dtype)


def _ones(shape):
    return torch.ones(tuple(shape), dtype=torch.float32)


def _covariance(shape):
    return torch.eye(shape[-1]).expand(tuple(shape)).clone()


def _real_distribution(name, params):
    if name == "Normal":
        return D.Normal(_zeros(params["loc"]), _ones(params["scale"]), validate_args=False)
    if name == "Categorical":
        return D.Categorical(logits=_zeros(params["logits"]), validate_args=False)
    if name == "MultivariateNormal":
        return D.MultivariateNormal(
            _zeros(params["loc"]),
            covariance_matrix=_covariance(params["covariance_matrix"]),
            validate_args=False,
        )
    raise AssertionError(name)


def _real_log_prob(dist, value_shape, *, dtype=torch.float32, positive=False):
    value = _ones(value_shape) if positive else _zeros(value_shape, dtype=dtype)
    return tuple(dist.log_prob(value).shape)


def _check_constructor(name, params, expected_batch, expected_event):
    verdict = verify_distribution(name, **params)
    real = _real_distribution(name, params)
    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.batch_shape == tuple(real.batch_shape) == expected_batch
    assert verdict.spec.event_shape == tuple(real.event_shape) == expected_event


def _check_constructor_rejected(name, params):
    verdict = verify_distribution(name, **params)
    assert not verdict.ok
    with pytest.raises(Exception):
        real = _real_distribution(name, params)
        real.sample()


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Distributions" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Generated constructor conformance
# --------------------------------------------------------------------------- #
def test_generated_normal_case_matches_real_torch():
    _check_constructor(
        "Normal",
        {"loc": (2, 1, 3), "scale": (1, 4, 3)},
        (2, 4, 3),
        (),
    )
    _check_constructor_rejected("Normal", {"loc": (2,), "scale": (3,)})


def test_generated_categorical_cases_match_real_torch():
    _check_constructor("Categorical", {"logits": (2, 3, 4)}, (2, 3), ())
    _check_constructor_rejected("Categorical", {"logits": ()})
    _check_constructor_rejected("Categorical", {"logits": (2, 0)})


def test_generated_mvn_cases_match_real_torch_use_failures():
    _check_constructor(
        "MultivariateNormal",
        {"loc": (4, 3, 5), "covariance_matrix": (1, 3, 5, 5)},
        (4, 3),
        (5,),
    )
    for params in [
        {"loc": (3,), "covariance_matrix": (3, 4)},
        {"loc": (3,), "covariance_matrix": (4, 4)},
        {"loc": (2, 3), "covariance_matrix": (4, 3, 3)},
    ]:
        verdict = verify_distribution("MultivariateNormal", **params)
        assert not verdict.ok


def test_generated_independent_case_matches_real_torch():
    base = verify_distribution("Normal", loc=(2, 3, 4), scale=(2, 3, 4)).spec
    verdict = verify_distribution("Independent", base=base, reinterpreted_batch_ndims=2)
    real = D.Independent(
        D.Normal(_zeros((2, 3, 4)), _ones((2, 3, 4)), validate_args=False),
        2,
        validate_args=False,
    )
    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.batch_shape == tuple(real.batch_shape) == (2,)
    assert verdict.spec.event_shape == tuple(real.event_shape) == (3, 4)

    assert not verify_distribution("Independent", base=base, reinterpreted_batch_ndims=4).ok
    with pytest.raises(ValueError):
        D.Independent(
            D.Normal(_zeros((2, 3, 4)), _ones((2, 3, 4)), validate_args=False),
            4,
            validate_args=False,
        )


# --------------------------------------------------------------------------- #
# 3. Generated log_prob and transformed-distribution conformance
# --------------------------------------------------------------------------- #
def test_generated_log_prob_cases_match_real_torch():
    normal = verify_distribution("Normal", loc=(2, 3), scale=(2, 3))
    normal_real = D.Normal(_zeros((2, 3)), _ones((2, 3)), validate_args=False)
    assert verify_log_prob(normal, (5, 2, 3)).output_shape == _real_log_prob(
        normal_real, (5, 2, 3)
    )

    categorical = verify_distribution("Categorical", logits=(2, 3, 4))
    categorical_real = D.Categorical(logits=_zeros((2, 3, 4)), validate_args=False)
    assert verify_log_prob(categorical, (3,)).output_shape == _real_log_prob(
        categorical_real, (3,), dtype=torch.long
    )

    mvn = verify_distribution(
        "MultivariateNormal",
        loc=(4, 3, 5),
        covariance_matrix=(4, 3, 5, 5),
    )
    mvn_real = _real_distribution(
        "MultivariateNormal",
        {"loc": (4, 3, 5), "covariance_matrix": (4, 3, 5, 5)},
    )
    assert verify_log_prob(mvn, (7, 4, 3, 1)).output_shape == _real_log_prob(
        mvn_real, (7, 4, 3, 1)
    )
    assert not verify_log_prob(mvn, (4, 3, 6)).ok


def test_generated_transformed_identity_case_matches_real_torch():
    base = verify_distribution("Normal", loc=(2, 3), scale=(2, 3)).spec
    verdict = verify_distribution(
        "TransformedDistribution",
        base=base,
        transforms=["ExpTransform"],
    )
    real = D.TransformedDistribution(
        D.Normal(_zeros((2, 3)), _ones((2, 3)), validate_args=False),
        [D.transforms.ExpTransform()],
        validate_args=False,
    )
    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.batch_shape == tuple(real.batch_shape) == (2, 3)
    assert verdict.spec.event_shape == tuple(real.event_shape) == ()
    assert verify_log_prob(verdict, (5, 2, 3)).output_shape == _real_log_prob(
        real, (5, 2, 3), positive=True
    )


def test_generated_transformed_reshape_case_matches_real_torch():
    base = verify_distribution("Normal", loc=(4, 2, 3), scale=(4, 2, 3)).spec
    independent = verify_distribution("Independent", base=base, reinterpreted_batch_ndims=2)
    verdict = verify_distribution(
        "TransformedDistribution",
        base=independent,
        transforms=[TransformSpec.reshape((2, 3), (6,))],
    )
    real = D.TransformedDistribution(
        D.Independent(
            D.Normal(_zeros((4, 2, 3)), _ones((4, 2, 3)), validate_args=False),
            2,
            validate_args=False,
        ),
        [D.transforms.ReshapeTransform((2, 3), (6,))],
        validate_args=False,
    )
    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.batch_shape == tuple(real.batch_shape) == (4,)
    assert verdict.spec.event_shape == tuple(real.event_shape) == (6,)
    assert verify_log_prob(verdict, (7, 4, 6)).output_shape == _real_log_prob(real, (7, 4, 6))


def test_generated_transformed_batch_reinterpret_and_composition_match_real_torch():
    base = verify_distribution("Normal", loc=(2, 3), scale=(2, 3)).spec
    verdict = verify_distribution(
        "TransformedDistribution",
        base=base,
        transforms=[TransformSpec.reshape((3,), (3,)), "ExpTransform"],
    )
    real = D.TransformedDistribution(
        D.Normal(_zeros((2, 3)), _ones((2, 3)), validate_args=False),
        [D.transforms.ReshapeTransform((3,), (3,)), D.transforms.ExpTransform()],
        validate_args=False,
    )
    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.batch_shape == tuple(real.batch_shape) == (2,)
    assert verdict.spec.event_shape == tuple(real.event_shape) == (3,)

    rejected = verify_distribution(
        "TransformedDistribution",
        base=base,
        transforms=[TransformSpec.reshape((4,), (4,))],
    )
    assert not rejected.ok
    assert rejected.error_kind == "transform_forward_shape"
    with pytest.raises(ValueError):
        D.TransformedDistribution(
            D.Normal(_zeros((2, 3)), _ones((2, 3)), validate_args=False),
            [D.transforms.ReshapeTransform((4,), (4,))],
            validate_args=False,
        )


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.Distributions"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.Distributions"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_DistributionsAxCheck.lean")
    body = "import TensorGuard.Distributions\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS
    ) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_DistributionsAxCheck.lean"],
            cwd=_LEAN,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = proc.stdout
    assert "sorryAx" not in out
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
