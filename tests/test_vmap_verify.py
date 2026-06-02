from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from src.vmap_verify import verify_vmap  # noqa: E402


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


def _empty(shape):
    return torch.empty(tuple(shape))


def test_basic_identity_transfer_matches_real_vmap():
    x = _empty((5, 2, 3))
    actual = torch.vmap(lambda a: a, in_dims=0, out_dims=0)(x)

    verdict = verify_vmap([(5, 2, 3)], (2, 3), in_dims=0, out_dims=0)

    assert verdict.ok
    assert verdict.batch_dim == 5
    assert verdict.body_input_shapes == ((2, 3),)
    assert verdict.output_shapes == tuple(actual.shape)


def test_negative_in_dim_and_nonleading_out_dim_match_real_vmap():
    x = _empty((5, 2, 3))
    actual = torch.vmap(lambda a: a, in_dims=-1, out_dims=1)(x)

    verdict = verify_vmap([(5, 2, 3)], (5, 2), in_dims=-1, out_dims=1)

    assert verdict.ok
    assert verdict.body_input_shapes == ((5, 2),)
    assert verdict.output_shapes == tuple(actual.shape) == (5, 3, 2)


def test_nested_output_tree_and_out_dims_match_real_torch_func_vmap():
    vmap = getattr(getattr(torch, "func", object()), "vmap", torch.vmap)
    x = _empty((5, 2, 3))

    actual = vmap(
        lambda a: (a, [a.sum(-1), {"z": a.sum()}]),
        in_dims=0,
        out_dims=(0, [1, {"z": 0}]),
    )(x)

    verdict = verify_vmap(
        [(5, 2, 3)],
        ((2, 3), [(2,), {"z": ()}]),
        in_dims=0,
        out_dims=(0, [1, {"z": 0}]),
    )

    assert verdict.ok
    assert verdict.output_shapes == _shape_tree(actual)


def test_mapped_batch_size_mismatch_is_refuted_like_real_vmap():
    verdict = verify_vmap([(5, 2), (6, 2)], (2,), in_dims=(0, 0))

    assert not verdict.ok
    assert verdict.error_kind == "batch_size"
    with pytest.raises(ValueError, match="same size in the mapped dimension"):
        torch.vmap(lambda a, b: a + b, in_dims=(0, 0))(_empty((5, 2)), _empty((6, 2)))


def test_invalid_in_dim_and_out_dim_are_refuted_like_real_vmap():
    assert verify_vmap([(5, 2, 3)], (2, 3), in_dims=3).error_kind == "in_dim_range"
    with pytest.raises(ValueError, match="expected in_dim"):
        torch.vmap(lambda a: a, in_dims=3)(_empty((5, 2, 3)))

    assert verify_vmap([(5, 2, 3)], (2, 3), out_dims=3).error_kind == "out_dim_range"
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.vmap(lambda a: a, out_dims=3)(_empty((5, 2, 3)))


def test_no_mapped_inputs_and_top_level_none_match_real_vmap_failures():
    assert verify_vmap([(2, 3)], (2, 3), in_dims=None).error_kind == "in_dim_structure"
    with pytest.raises(ValueError, match="in_dims=None"):
        torch.vmap(lambda a: a, in_dims=None)(_empty((2, 3)))

    assert verify_vmap([(2, 3), (2, 3)], (2, 3), in_dims=(None, None)).error_kind == "no_mapped_inputs"
    with pytest.raises(ValueError, match="at least one Tensor"):
        torch.vmap(lambda a, b: a + b, in_dims=(None, None))(_empty((2, 3)), _empty((2, 3)))


def test_unbatched_constant_output_none_and_leading_out_dim_match_real_vmap():
    x = _empty((5, 2, 3))
    const = _empty((2, 3))

    actual_none = torch.vmap(lambda a, b: b, in_dims=(0, None), out_dims=None)(x, const)
    verdict_none = verify_vmap(
        [(5, 2, 3), (2, 3)],
        (2, 3),
        in_dims=(0, None),
        out_dims=None,
        body_output_batched=False,
    )
    assert verdict_none.ok
    assert verdict_none.output_shapes == tuple(actual_none.shape) == (2, 3)

    actual_leading = torch.vmap(lambda a, b: b, in_dims=(0, None), out_dims=0)(x, const)
    verdict_leading = verify_vmap(
        [(5, 2, 3), (2, 3)],
        (2, 3),
        in_dims=(0, None),
        out_dims=0,
        body_output_batched=False,
    )
    assert verdict_leading.ok
    assert verdict_leading.output_shapes == tuple(actual_leading.shape) == (5, 2, 3)


def test_unbatched_constant_output_nonleading_out_dim_abstains_not_refutes():
    verdict = verify_vmap(
        [(5, 2, 3), (2, 3)],
        (2, 3),
        in_dims=(0, None),
        out_dims=1,
        body_output_batched=False,
    )

    assert verdict.ok
    assert verdict.output_shapes is None
    assert verdict.unknown_reason is not None
    assert "non-leading integer out_dim" in verdict.unknown_reason


def test_symbolic_batch_dims_are_not_refuted_unless_known_sizes_disagree():
    symbolic = verify_vmap([("B", 2), (5, 2)], (2,), in_dims=(0, 0))
    assert symbolic.ok
    assert symbolic.batch_dim == "B"
    assert symbolic.output_shapes == ("B", 2)

    concrete_mismatch = verify_vmap([("B", 2), (4, 2), (5, 2)], (2,), in_dims=(0, 0, 0))
    assert not concrete_mismatch.ok
    assert concrete_mismatch.error_kind == "batch_size"


def test_identity_transfer_differential_fuzz_against_real_vmap():
    rng = random.Random(20240602)
    for _ in range(500):
        rank = rng.randint(1, 4)
        batch = rng.randint(1, 5)
        in_dim = rng.randrange(rank)
        body_shape = tuple(rng.randint(1, 5) for _ in range(rank - 1))
        shape = body_shape[:in_dim] + (batch,) + body_shape[in_dim:]

        rank_after_insert = len(body_shape) + 1
        raw_out_dim = rng.choice(list(range(-rank_after_insert, rank_after_insert)))
        actual = torch.vmap(lambda a: a, in_dims=in_dim, out_dims=raw_out_dim)(_empty(shape))

        verdict = verify_vmap([shape], body_shape, in_dims=in_dim, out_dims=raw_out_dim)

        assert verdict.ok, (shape, in_dim, raw_out_dim, verdict)
        assert verdict.body_input_shapes == (body_shape,)
        assert verdict.output_shapes == tuple(actual.shape)


def test_public_package_exports_vmap_checker():
    import tensorguard

    assert tensorguard.verify_vmap is verify_vmap
