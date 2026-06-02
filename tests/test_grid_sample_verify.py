from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.grid_sample_verify import verify_affine_grid, verify_grid_sample  # noqa: E402


def _tensor(shape, dtype):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(*shape, dtype=dtype)
    return torch.ones(*shape, dtype=dtype)


def _real_grid_sample(
    input_shape,
    grid_shape,
    *,
    input_dtype=torch.float32,
    grid_dtype=torch.float32,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
):
    try:
        out = F.grid_sample(
            _tensor(input_shape, input_dtype),
            _tensor(grid_shape, grid_dtype),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return "ok", tuple(out.shape), out.dtype
    except Exception:
        return "err", None, None


def _check_grid_sample(
    input_shape,
    grid_shape,
    *,
    input_dtype=torch.float32,
    grid_dtype=torch.float32,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
):
    real_status, real_shape, real_dtype = _real_grid_sample(
        input_shape,
        grid_shape,
        input_dtype=input_dtype,
        grid_dtype=grid_dtype,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    verdict = verify_grid_sample(
        input_shape,
        grid_shape,
        input_dtype=input_dtype,
        grid_dtype=grid_dtype,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    assert ("ok" if verdict.ok else "err") == real_status, (
        f"input={input_shape} grid={grid_shape} input_dtype={input_dtype} "
        f"grid_dtype={grid_dtype} mode={mode}: real={real_status} "
        f"static={verdict}"
    )
    if real_status == "ok":
        assert verdict.output_shape == real_shape
        assert verdict.output_dtype == str(real_dtype).replace("torch.", "")
    return verdict


def _real_affine_grid(theta_shape, size, *, theta_dtype=torch.float32, align_corners=False):
    try:
        out = F.affine_grid(
            _tensor(theta_shape, theta_dtype),
            tuple(size),
            align_corners=align_corners,
        )
        return "ok", tuple(out.shape), out.dtype
    except Exception:
        return "err", None, None


def _check_affine_grid(theta_shape, size, *, theta_dtype=torch.float32, align_corners=False):
    real_status, real_shape, real_dtype = _real_affine_grid(
        theta_shape,
        size,
        theta_dtype=theta_dtype,
        align_corners=align_corners,
    )
    verdict = verify_affine_grid(
        theta_shape,
        size,
        theta_dtype=theta_dtype,
        align_corners=align_corners,
    )
    assert ("ok" if verdict.ok else "err") == real_status, (
        f"theta={theta_shape} size={size} dtype={theta_dtype}: "
        f"real={real_status} static={verdict}"
    )
    if real_status == "ok":
        assert verdict.output_shape == real_shape
        assert verdict.output_dtype == str(real_dtype).replace("torch.", "")
    return verdict


@pytest.mark.parametrize(
    "input_shape,grid_shape,kwargs",
    [
        ((2, 3, 4, 5), (2, 6, 7, 2), {}),
        ((2, 3, 4, 5, 6), (2, 7, 8, 9, 3), {}),
        ((1, 0, 4, 5), (1, 6, 7, 2), {}),  # zero channels are legal
        ((0, 3, 4, 5), (0, 6, 7, 2), {}),  # zero batch is legal
        ((1, 3, 4, 5), (1, 0, 7, 2), {}),  # empty output grid is legal
        ((1, 3, 4, 5), (1, 6, 7, 2), {"input_dtype": torch.float16, "grid_dtype": torch.float16}),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"input_dtype": torch.float64, "grid_dtype": torch.float64}),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"mode": "bicubic"}),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"padding_mode": "reflection"}),
    ],
)
def test_grid_sample_valid_contracts_match_real_torch(input_shape, grid_shape, kwargs):
    verdict = _check_grid_sample(input_shape, grid_shape, **kwargs)
    assert verdict.ok


@pytest.mark.parametrize(
    "input_shape,grid_shape,kwargs,kind",
    [
        ((2, 3, 4, 5), (1, 6, 7, 2), {}, "batch"),
        ((1, 3, 4, 5), (1, 6, 7, 3), {}, "grid_last_dim"),
        ((1, 3, 4, 5, 6), (1, 6, 7, 8, 2), {}, "grid_last_dim"),
        ((1, 3, 0, 5), (1, 6, 7, 2), {}, "input_spatial"),
        ((1, 3, 4, 0, 6), (1, 6, 7, 8, 3), {}, "input_spatial"),
        ((1, 3, 4, 5, 6), (1, 6, 7, 8, 3), {"mode": "bicubic"}, "mode_rank"),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"mode": "bad"}, "mode"),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"padding_mode": "bad"}, "padding_mode"),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"align_corners": 1}, "align_corners"),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"input_dtype": torch.float64, "grid_dtype": torch.float32}, "dtype_mismatch"),
        ((1, 3, 4, 5), (1, 6, 7, 2), {"input_dtype": torch.int64, "grid_dtype": torch.int64}, "dtype"),
    ],
)
def test_grid_sample_invalid_contracts_match_real_torch(input_shape, grid_shape, kwargs, kind):
    verdict = _check_grid_sample(input_shape, grid_shape, **kwargs)
    assert not verdict.ok
    assert verdict.error_kind == kind


@pytest.mark.parametrize(
    "theta_shape,size,kwargs",
    [
        ((2, 2, 3), (2, 3, 6, 7), {}),
        ((2, 3, 4), (2, 3, 5, 6, 7), {}),
        ((1, 2, 3), (1, 1, 2, 2), {"theta_dtype": torch.float16}),
        ((1, 2, 3), (1, 1, 2, 2), {"theta_dtype": torch.bfloat16}),
        ((1, 2, 3), (1, 1, 2, 2), {"theta_dtype": torch.float64}),
    ],
)
def test_affine_grid_valid_contracts_match_real_torch(theta_shape, size, kwargs):
    verdict = _check_affine_grid(theta_shape, size, **kwargs)
    assert verdict.ok


@pytest.mark.parametrize(
    "theta_shape,size,kwargs,kind",
    [
        ((2, 2, 2), (2, 3, 6, 7), {}, "theta_matrix"),
        ((2, 3, 4), (2, 3, 6, 7), {}, "theta_matrix"),
        ((1, 2, 3), (2, 3, 6, 7), {}, "batch"),
        ((2, 2, 3), (2, 6, 7), {}, "size_rank"),
        ((0, 2, 3), (0, 3, 6, 7), {}, "size_positive"),
        ((1, 2, 3), (1, 0, 6, 7), {}, "size_positive"),
        ((1, 2, 3), (1, 3, 0, 7), {}, "size_positive"),
        ((1, 2, 3), (1, 3, 6, 7), {"theta_dtype": torch.int64}, "dtype"),
        ((1, 2, 3), (1, 3, 6, 7), {"align_corners": 1}, "align_corners"),
    ],
)
def test_affine_grid_invalid_contracts_match_real_torch(theta_shape, size, kwargs, kind):
    verdict = _check_affine_grid(theta_shape, size, **kwargs)
    assert not verdict.ok
    assert verdict.error_kind == kind


def test_symbolic_dimensions_are_carried_without_false_positive():
    sampled = verify_grid_sample(("N", "C", "H", "W"), ("N", "OH", "OW", 2))
    assert sampled.ok
    assert sampled.output_shape == ("N", "C", "OH", "OW")

    uncertain_batch = verify_grid_sample(("N", 3, 4, 5), (2, 6, 7, 2))
    assert uncertain_batch.ok
    assert uncertain_batch.output_shape == ("N", 3, 6, 7)

    affine = verify_affine_grid(("N", 2, 3), ("N", "C", "H", "W"))
    assert affine.ok
    assert affine.output_shape == ("N", "H", "W", 2)


def test_grid_sample_randomized_differential_fuzz_matches_real_torch():
    rng = random.Random(20240606)
    dtypes = [torch.float32, torch.float64]
    checked = 0
    for _ in range(320):
        rank = rng.choice([4, 5])
        n = rng.choice([1, 2, 3])
        c = rng.choice([1, 2, 4])
        in_spatial = tuple(rng.randint(1, 6) for _ in range(rank - 2))
        out_spatial = tuple(rng.randint(1, 6) for _ in range(rank - 2))
        input_shape = (n, c) + in_spatial

        grid_batch = n if rng.random() < 0.85 else rng.choice([b for b in (1, 2, 3) if b != n])
        grid_last = rank - 2 if rng.random() < 0.85 else rng.choice([1, 2, 3, 4])
        grid_shape = (grid_batch,) + out_spatial + (grid_last,)

        mode = rng.choice(["bilinear", "nearest", "bicubic"])
        input_dtype = rng.choice(dtypes)
        grid_dtype = input_dtype if rng.random() < 0.85 else rng.choice([d for d in dtypes if d != input_dtype])

        _check_grid_sample(
            input_shape,
            grid_shape,
            input_dtype=input_dtype,
            grid_dtype=grid_dtype,
            mode=mode,
            padding_mode=rng.choice(["zeros", "border", "reflection"]),
        )
        checked += 1
    assert checked == 320


def test_affine_grid_randomized_differential_fuzz_matches_real_torch():
    rng = random.Random(20240607)
    # Half/bfloat affine grids with unit spatial dimensions hit CPU-only kernel
    # gaps ("tensor_cpu not implemented") even though the shape contract is valid
    # and the same static contract can run on other devices.  The differential
    # fuzz isolates shape/matrix/batch semantics with portable floating dtypes.
    dtypes = [torch.float32, torch.float64]
    checked = 0
    for _ in range(220):
        size_rank = rng.choice([4, 5])
        n = rng.choice([1, 2, 3])
        c = rng.choice([1, 2, 4])
        spatial = tuple(rng.randint(1, 6) for _ in range(size_rank - 2))
        size = (n, c) + spatial

        rows = size_rank - 2
        cols = rows + 1
        theta_batch = n if rng.random() < 0.85 else rng.choice([b for b in (1, 2, 3) if b != n])
        theta_rows = rows if rng.random() < 0.9 else rng.choice([2, 3, 4])
        theta_cols = cols if rng.random() < 0.9 else rng.choice([2, 3, 4, 5])
        theta_shape = (theta_batch, theta_rows, theta_cols)

        _check_affine_grid(theta_shape, size, theta_dtype=rng.choice(dtypes))
        checked += 1
    assert checked == 220


def test_public_package_exports_grid_sample_verifiers():
    import src
    import tensorguard

    assert src.verify_grid_sample is verify_grid_sample
    assert src.verify_affine_grid is verify_affine_grid
    assert tensorguard.verify_grid_sample is verify_grid_sample
    assert tensorguard.verify_affine_grid is verify_affine_grid
