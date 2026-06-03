from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.func_autodiff_verify import (  # noqa: E402
    verify_func_autodiff,
    verify_func_grad,
    verify_func_jacfwd,
    verify_func_jacrev,
    verify_func_jvp,
    verify_func_vjp,
)


def _shape_tree(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, tuple):
        return tuple(_shape_tree(v) for v in value)
    if isinstance(value, list):
        return [_shape_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _shape_tree(value[k]) for k in sorted(value)}
    return value


def _ones_tree(value):
    if hasattr(value, "shape"):
        return torch.ones_like(value)
    if isinstance(value, tuple):
        return tuple(_ones_tree(v) for v in value)
    if isinstance(value, list):
        return [_ones_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _ones_tree(value[k]) for k in value}
    raise TypeError(value)


def _func_namespace():
    return pytest.importorskip("torch.func")


def test_grad_scalar_single_and_tuple_argnums_match_real_torch_func():
    func = _func_namespace()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    f = lambda a, b: (a * b).sum()

    single = verify_func_grad([(2, 3), (2, 3)], (), argnums=0)
    tuple_one = verify_func_grad([(2, 3), (2, 3)], (), argnums=(0,))
    tuple_two = verify_func_grad([(2, 3), (2, 3)], (), argnums=(0, 1))

    assert single.ok
    assert single.output_shapes == _shape_tree(func.grad(f, argnums=0)(x, y)) == (2, 3)
    assert tuple_one.ok
    assert tuple_one.output_shapes == _shape_tree(func.grad(f, argnums=(0,))(x, y)) == ((2, 3),)
    assert tuple_two.ok
    assert tuple_two.output_shapes == _shape_tree(func.grad(f, argnums=(0, 1))(x, y)) == ((2, 3), (2, 3))


def test_grad_has_aux_and_ranked_size_one_rejection_match_real_torch_func():
    func = _func_namespace()
    x = torch.randn(2, 3)

    aux_verdict = verify_func_grad([(2, 3)], (), has_aux=True, aux_shapes={"row": (2,)})
    actual_aux = func.grad(lambda a: (a.square().sum(), {"row": a.sum(1)}), has_aux=True)(x)
    assert aux_verdict.ok
    assert aux_verdict.output_shapes == _shape_tree(actual_aux) == ((2, 3), {"row": (2,)})

    bad = verify_func_grad([(2, 3)], (1,))
    assert not bad.ok
    assert bad.error_kind == "scalar_output"
    with pytest.raises(RuntimeError, match="return a scalar Tensor"):
        func.grad(lambda a: a.sum().reshape(1))(x)


@pytest.mark.parametrize("jac_name,checker_name", [("jacrev", "verify_func_jacrev"), ("jacfwd", "verify_func_jacfwd")])
def test_jacrev_and_jacfwd_tensor_outputs_match_real_torch_func(jac_name, checker_name):
    func = _func_namespace()
    jac = getattr(func, jac_name)
    checker = verify_func_jacrev if checker_name == "verify_func_jacrev" else verify_func_jacfwd
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    f = lambda a, b: a @ b.T

    single = checker([(2, 3), (2, 3)], (2, 2), argnums=-1)
    tuple_one = checker([(2, 3), (2, 3)], (2, 2), argnums=(0,))
    tuple_two = checker([(2, 3), (2, 3)], (2, 2), argnums=(0, 1))

    assert single.ok
    assert single.output_shapes == _shape_tree(jac(f, argnums=-1)(x, y)) == (2, 2, 2, 3)
    assert tuple_one.ok
    assert tuple_one.output_shapes == _shape_tree(jac(f, argnums=(0,))(x, y)) == ((2, 2, 2, 3),)
    assert tuple_two.ok
    assert tuple_two.output_shapes == _shape_tree(jac(f, argnums=(0, 1))(x, y)) == (
        (2, 2, 2, 3),
        (2, 2, 2, 3),
    )


@pytest.mark.parametrize("jac_name,checker", [("jacrev", verify_func_jacrev), ("jacfwd", verify_func_jacfwd)])
def test_jacrev_and_jacfwd_pytree_outputs_and_aux_match_real_torch_func(jac_name, checker):
    func = _func_namespace()
    jac = getattr(func, jac_name)
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    body_tree = ((2,), {"z": (3,)})
    f_tree = lambda a, b: (a.sum(1), {"z": (a * b).sum(0)})
    verdict = checker([(2, 3), (2, 3)], body_tree, argnums=(0, 1))
    assert verdict.ok
    assert verdict.output_shapes == _shape_tree(jac(f_tree, argnums=(0, 1))(x, y))
    assert verdict.output_shapes == (((2, 2, 3), (2, 2, 3)), {"z": ((3, 2, 3), (3, 2, 3))})

    aux_verdict = checker([(2, 3)], (), has_aux=True, aux_shapes={"aux": (2,)})
    actual_aux = jac(lambda a: (a.square().sum(), {"aux": a.sum(1)}), has_aux=True)(x)
    assert aux_verdict.ok
    assert aux_verdict.output_shapes == _shape_tree(actual_aux) == ((2, 3), {"aux": (2,)})


def test_jvp_pytree_and_aux_outputs_match_real_torch_func():
    func = _func_namespace()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    tx = torch.ones_like(x)
    ty = torch.ones_like(y)

    body_tree = ((2, 3), (2,))
    verdict = verify_func_jvp([(2, 3), (2, 3)], [(2, 3), (2, 3)], body_tree)
    actual = func.jvp(lambda a, b: (a * b, a.sum(1)), (x, y), (tx, ty))
    assert verdict.ok
    assert verdict.output_shapes == _shape_tree(actual) == (body_tree, body_tree)

    aux_verdict = verify_func_jvp([(2, 3)], [(2, 3)], (), has_aux=True, aux_shapes={"aux": (2,)})
    actual_aux = func.jvp(lambda a: (a.square().sum(), {"aux": a.sum(1)}), (x,), (tx,), has_aux=True)
    assert aux_verdict.ok
    assert aux_verdict.output_shapes == _shape_tree(actual_aux) == ((), (), {"aux": (2,)})


def test_jvp_tangent_shape_mismatch_matches_real_torch_func_failure():
    func = _func_namespace()
    x = torch.randn(2, 3)

    verdict = verify_func_jvp([(2, 3)], [(2, 2)], ())
    assert not verdict.ok
    assert verdict.error_kind == "tangent_shape"
    with pytest.raises(RuntimeError, match="different size"):
        func.jvp(lambda a: a.sum(), (x,), (torch.ones(2, 2),))

    assert verify_func_jvp([(2, 3), (2, 3)], [(2, 3)], ()).error_kind == "tangent_structure"


def test_vjp_output_and_pullback_contracts_match_real_torch_func():
    func = _func_namespace()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    f = lambda a, b: (a * b, a.sum(1))

    out, pullback = func.vjp(f, x, y)
    actual_pullback = pullback(_ones_tree(out))
    verdict = verify_func_vjp([(2, 3), (2, 3)], ((2, 3), (2,)), cotangent_shapes=((2, 3), (2,)))

    assert verdict.ok
    assert verdict.output_shapes == _shape_tree(out) == ((2, 3), (2,))
    assert verdict.pullback_input_shapes == ((2, 3), (2,))
    assert verdict.pullback_output_shapes == _shape_tree(actual_pullback) == ((2, 3), (2, 3))


def test_vjp_aux_and_cotangent_mismatch_match_real_torch_func():
    func = _func_namespace()
    x = torch.randn(2, 3)

    out, pullback, aux = func.vjp(lambda a: (a.sum(1), {"aux": a.mean(0)}), x, has_aux=True)
    verdict = verify_func_vjp([(2, 3)], (2,), has_aux=True, aux_shapes={"aux": (3,)}, cotangent_shapes=(2,))
    assert verdict.ok
    assert verdict.output_shapes == (_shape_tree(out), _shape_tree(aux)) == ((2,), {"aux": (3,)})
    assert verdict.pullback_output_shapes == _shape_tree(pullback(torch.ones_like(out))) == ((2, 3),)

    bad = verify_func_vjp([(2, 3)], (2,), cotangent_shapes=(3,))
    assert not bad.ok
    assert bad.error_kind == "cotangent_shape"
    with pytest.raises(RuntimeError, match="Mismatch in shape"):
        func.vjp(lambda a: a.sum(1), x)[1](torch.ones(3))


def test_argnums_range_and_duplicate_after_negative_normalisation_match_real_torch_func():
    func = _func_namespace()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    f = lambda a, b: (a * b).sum()

    assert verify_func_grad([(2, 3), (2, 3)], (), argnums=2).error_kind == "argnum_range"
    with pytest.raises(RuntimeError, match="argnum=2"):
        func.grad(f, argnums=2)(x, y)

    dup = verify_func_jacrev([(2, 3), (2, 3)], (), argnums=(1, -1))
    assert not dup.ok
    assert dup.error_kind == "argnums_duplicate"
    with pytest.raises(RuntimeError, match="argnums elements must be unique"):
        func.jacrev(f, argnums=(1, -1))(x, y)


def test_symbolic_equality_abstains_without_refuting_tangent_or_cotangent_contracts():
    jvp = verify_func_jvp([("B", 3)], [(5, 3)], ())
    assert jvp.ok
    assert jvp.unknown_reason is not None
    assert "symbolic" in jvp.unknown_reason

    vjp = verify_func_vjp([("B", 3)], ("B",), cotangent_shapes=(5,))
    assert vjp.ok
    assert vjp.unknown_reason is not None
    assert "symbolic" in vjp.unknown_reason


def test_value_dependent_and_pytree_input_contracts_abstain():
    value_dependent = verify_func_jacrev([(2, 3)], (2,), value_dependent=True)
    assert value_dependent.ok
    assert value_dependent.output_shapes is None
    assert "runtime values" in value_dependent.unknown_reason

    pytree_input = verify_func_grad([{"x": (2, 3)}], ())
    assert pytree_input.ok
    assert pytree_input.output_shapes is None
    assert "pytree input" in pytree_input.unknown_reason


def test_dispatcher_and_public_package_exports():
    dispatched = verify_func_autodiff("jacrev", [(2, 3)], (4,), argnums=(0,))
    assert dispatched.ok
    assert dispatched.output_shapes == ((4, 2, 3),)

    dispatched_jvp = verify_func_autodiff("jvp", [(2, 3)], (4,), tangent_shapes=[(2, 3)])
    assert dispatched_jvp.ok
    assert dispatched_jvp.output_shapes == ((4,), (4,))

    import src
    import tensorguard

    assert src.verify_func_grad is verify_func_grad
    assert src.verify_func_jacrev is verify_func_jacrev
    assert src.verify_func_jacfwd is verify_func_jacfwd
    assert src.verify_func_jvp is verify_func_jvp
    assert src.verify_func_vjp is verify_func_vjp
    assert tensorguard.verify_func_grad is verify_func_grad
    assert tensorguard.verify_func_jacrev is verify_func_jacrev
    assert tensorguard.verify_func_jacfwd is verify_func_jacfwd
    assert tensorguard.verify_func_jvp is verify_func_jvp
    assert tensorguard.verify_func_vjp is verify_func_vjp
