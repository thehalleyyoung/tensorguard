"""Static tensor-shape contracts for ``torchvision.transforms.v2``.

The verifier is intentionally torch/torchvision independent.  It models the
shape-visible tensor path of common v2 image transforms and abstains on
PIL-only or value-random behaviour rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

Dim = Any
Shape = Tuple[Dim, ...]


@dataclass(frozen=True)
class TransformVerdict:
    """Result of one torchvision-v2 tensor transform contract check."""

    ok: bool
    transform_name: str
    output_shape: Optional[Shape] = None
    error_kind: Optional[str] = None
    message: Optional[str] = None
    unknown_reason: Optional[str] = None


_ALIASES = {
    "resize": "resize",
    "randomresize": "resize",
    "centercrop": "center_crop",
    "fivecrop": "five_crop",
    "tencrop": "ten_crop",
    "randomcrop": "random_crop",
    "randomresizedcrop": "random_resized_crop",
    "pad": "pad",
    "normalize": "normalize",
    "randomhorizontalflip": "horizontal_flip",
    "horizontalflip": "horizontal_flip",
    "randomverticalflip": "vertical_flip",
    "verticalflip": "vertical_flip",
    "colorjitter": "color_jitter",
    "randominvert": "random_invert",
    "randomposterize": "random_posterize",
    "randomsolarize": "random_solarize",
    "randomautocontrast": "random_autocontrast",
    "randomequalize": "random_equalize",
    "randomadjustsharpness": "random_adjust_sharpness",
    "randomgrayscale": "random_grayscale",
    "gaussianblur": "gaussian_blur",
    "identity": "identity",
    "todtype": "to_dtype",
    "topuretensor": "to_pure_tensor",
    "piltoTensor".lower(): "pil_to_tensor",
    "topilimage": "to_pil_image",
}

_PIL_ONLY = frozenset({"pil_to_tensor", "to_pil_image"})
_PRESERVE_ANY_RANK = frozenset({
    "horizontal_flip",
    "vertical_flip",
    "color_jitter",
    "random_invert",
    "random_posterize",
    "random_solarize",
    "random_equalize",
    "random_adjust_sharpness",
    "identity",
    "to_dtype",
    "to_pure_tensor",
})
_PRESERVE_RANK2 = frozenset({"random_grayscale"})
_PRESERVE_RANK3 = frozenset({"gaussian_blur", "random_autocontrast"})


def _canon_transform_name(name: Any) -> str:
    short = str(name).split(".")[-1]
    key = short.replace("_", "").replace("-", "").replace(" ", "").lower()
    return _ALIASES.get(key, short.replace("-", "_").lower())


def _as_shape(shape: Optional[Sequence[Dim]]) -> Optional[Shape]:
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


def _is_int_dim(value: object) -> bool:
    return type(value) is int


def _fail(name: str, kind: str, message: str) -> TransformVerdict:
    return TransformVerdict(False, name, error_kind=kind, message=message)


def _ok(
    name: str,
    output_shape: Optional[Shape],
    *,
    unknown_reason: Optional[str] = None,
) -> TransformVerdict:
    return TransformVerdict(True, name, output_shape=output_shape, unknown_reason=unknown_reason)


def _unknown(name: str, reason: str, output_shape: Optional[Shape] = None) -> TransformVerdict:
    return _ok(name, output_shape, unknown_reason=reason)


def _require_rank(name: str, shape: Shape, minimum: int) -> Optional[TransformVerdict]:
    if len(shape) < minimum:
        return _fail(
            name,
            "rank",
            f"torchvision.transforms.v2 {name} expects tensor rank >= {minimum}, got {len(shape)}",
        )
    return None


def _size_pair(size: Any, *, crop: bool) -> Tuple[Optional[Tuple[int, int]], Optional[int], Optional[str]]:
    """Return ``((h, w), None, None)`` for exact sizes or ``(None, side, None)``
    for Resize's short-edge integer form.
    """

    if isinstance(size, bool):
        return None, None, "size must be an int or a 1/2-tuple of ints, got bool"
    if isinstance(size, int):
        if size <= 0:
            return None, None, f"size entries must be positive, got {size}"
        if crop:
            return (size, size), None, None
        return None, size, None
    if not isinstance(size, (tuple, list)):
        return None, None, "size must be an int or tuple/list of ints"
    values = tuple(size)
    if len(values) == 1:
        only = values[0]
        if isinstance(only, bool) or not isinstance(only, int):
            return None, None, "size entries must be integers"
        if only <= 0:
            return None, None, f"size entries must be positive, got {only}"
        if crop:
            return (only, only), None, None
        return None, only, None
    if len(values) != 2:
        return None, None, f"size must have length 1 or 2, got {len(values)}"
    h, w = values
    if (
        isinstance(h, bool)
        or isinstance(w, bool)
        or not isinstance(h, int)
        or not isinstance(w, int)
    ):
        return None, None, "size entries must be integers"
    if h <= 0 or w <= 0:
        return None, None, f"size entries must be positive, got {(h, w)!r}"
    return (h, w), None, None


def _resize_short_edge(shape: Shape, short_edge: int, max_size: Optional[Any]) -> Tuple[Optional[Shape], Optional[str]]:
    h, w = shape[-2], shape[-1]
    if not (_is_int_dim(h) and _is_int_dim(w)):
        return None, "Resize(int) needs concrete input H/W to preserve aspect ratio statically"
    if h <= 0 or w <= 0:
        return None, f"Resize expects positive spatial dims, got {(h, w)!r}"
    short = min(h, w)
    long = max(h, w)
    new_short = int(short_edge)
    new_long = int(short_edge * long / short)
    if isinstance(max_size, int) and max_size > 0 and new_long > max_size:
        new_short = int(max_size * new_short / new_long)
        new_long = max_size
    if h <= w:
        new_h, new_w = new_short, new_long
    else:
        new_h, new_w = new_long, new_short
    return shape[:-2] + (new_h, new_w), None


def _normalize_padding(padding: Any) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    if isinstance(padding, bool):
        return None, "padding must be an int or a 1, 2, or 4 element tuple/list"
    if isinstance(padding, int):
        return (padding, padding, padding, padding), None
    if not isinstance(padding, (tuple, list)):
        return None, "padding must be an int or a 1, 2, or 4 element tuple/list"
    values = tuple(padding)
    if len(values) not in (1, 2, 4):
        return None, f"padding must have length 1, 2, or 4, got {len(values)}"
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        return None, "padding entries must be integers"
    if len(values) == 1:
        p = values[0]
        return (p, p, p, p), None
    if len(values) == 2:
        left_right, top_bottom = values
        return (left_right, top_bottom, left_right, top_bottom), None
    left, top, right, bottom = values
    return (left, top, right, bottom), None


def normalized_padding_2d(padding: Any) -> Optional[Tuple[int, int, int, int]]:
    """Public helper for model-checker transition constraints."""

    norm, _err = _normalize_padding(padding)
    return norm


def _sequence_len(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _verify_resize(
    name: str,
    shape: Shape,
    *,
    size: Any,
    max_size: Optional[Any],
) -> TransformVerdict:
    rank_err = _require_rank(name, shape, 3)
    if rank_err is not None:
        return rank_err
    exact, short_edge, err = _size_pair(size, crop=False)
    if err is not None:
        return _fail(name, "argument", err)
    if exact is not None:
        h, w = exact
        return _ok(name, shape[:-2] + (h, w))
    if short_edge is None:
        return _unknown(name, "Resize size is not statically known", shape)
    resized, unknown = _resize_short_edge(shape, short_edge, max_size)
    if resized is None:
        return _unknown(name, unknown or "Resize(int) output is symbolic", shape[:-2] + ("_resize_h", "_resize_w"))
    return _ok(name, resized)


def _verify_crop(
    name: str,
    shape: Shape,
    *,
    size: Any,
    random_crop: bool = False,
    pad_if_needed: bool = False,
    padding: Any = None,
) -> TransformVerdict:
    rank_err = _require_rank(name, shape, 2)
    if rank_err is not None:
        return rank_err
    exact, _short, err = _size_pair(size, crop=True)
    if err is not None:
        return _fail(name, "argument", err)
    if exact is None:
        return _unknown(name, f"{name} size is not statically known", shape)
    crop_h, crop_w = exact
    eff_h, eff_w = shape[-2], shape[-1]
    if padding is not None:
        norm, pad_err = _normalize_padding(padding)
        if pad_err is not None:
            return _fail(name, "argument", pad_err)
        assert norm is not None
        left, top, right, bottom = norm
        if _is_int_dim(eff_h):
            eff_h = eff_h + top + bottom
        if _is_int_dim(eff_w):
            eff_w = eff_w + left + right
    if (
        random_crop
        and not pad_if_needed
        and _is_int_dim(eff_h)
        and _is_int_dim(eff_w)
        and (crop_h > eff_h or crop_w > eff_w)
    ):
        return _fail(
            name,
            "shape",
            f"{name} requested crop {(crop_h, crop_w)} exceeds input image {(eff_h, eff_w)}",
        )
    return _ok(name, shape[:-2] + (crop_h, crop_w))


def _verify_pad(name: str, shape: Shape, *, padding: Any) -> TransformVerdict:
    rank_err = _require_rank(name, shape, 3)
    if rank_err is not None:
        return rank_err
    norm, err = _normalize_padding(padding)
    if err is not None:
        return _fail(name, "argument", err)
    assert norm is not None
    left, top, right, bottom = norm
    h, w = shape[-2], shape[-1]
    out_h: Dim
    out_w: Dim
    if _is_int_dim(h):
        out_h = h + top + bottom
        if out_h <= 0:
            return _fail(name, "shape", f"Pad makes image height non-positive: {out_h}")
    else:
        out_h = f"_pad_h({h},{top + bottom})"
    if _is_int_dim(w):
        out_w = w + left + right
        if out_w <= 0:
            return _fail(name, "shape", f"Pad makes image width non-positive: {out_w}")
    else:
        out_w = f"_pad_w({w},{left + right})"
    return _ok(name, shape[:-2] + (out_h, out_w))


def _verify_normalize(
    name: str,
    shape: Shape,
    *,
    mean: Any,
    std: Any,
) -> TransformVerdict:
    rank_err = _require_rank(name, shape, 3)
    if rank_err is not None:
        return rank_err
    mean_len = _sequence_len(mean)
    std_len = _sequence_len(std)
    if mean_len is None or std_len is None:
        return _unknown(name, "Normalize mean/std lengths are not statically known", shape)
    if mean_len <= 0 or std_len <= 0:
        return _fail(name, "argument", "Normalize mean/std must be non-empty sequences")
    c = shape[-3]
    if mean_len != std_len and mean_len != 1 and std_len != 1:
        return _fail(
            name,
            "shape",
            f"Normalize mean length {mean_len} and std length {std_len} are not broadcastable",
        )
    stat_channels = max(mean_len, std_len)
    if _is_int_dim(c):
        if c != stat_channels and c != 1 and stat_channels != 1:
            return _fail(
                name,
                "shape",
                f"Normalize channel dim {c} is not broadcastable with mean/std length {stat_channels}",
            )
        out_c = stat_channels if c == 1 else c
        return _ok(name, shape[:-3] + (out_c,) + shape[-2:])
    if stat_channels == 1:
        return _ok(name, shape)
    return _unknown(
        name,
        f"Normalize symbolic channel dim {c!r} may need broadcasting to {stat_channels}",
        shape[:-3] + (f"_norm_c({c},{stat_channels})",) + shape[-2:],
    )


def verify_torchvision_v2_transform(
    transform_name: Any,
    input_shape: Optional[Sequence[Dim]],
    *,
    size: Any = None,
    padding: Any = None,
    pad_if_needed: bool = False,
    max_size: Optional[Any] = None,
    mean: Any = None,
    std: Any = None,
) -> TransformVerdict:
    """Verify a torchvision-v2 tensor transform's static shape contract.

    ``input_shape`` is the tensor shape on the tensor path.  PIL-only transforms
    and missing/non-tensor shapes return ``ok=True`` with ``unknown_reason`` so
    callers can abstain instead of emitting a false alarm.
    """

    name = _canon_transform_name(transform_name)
    shape = _as_shape(input_shape)
    if name in _PIL_ONLY:
        return _unknown(name, f"{name} produces/consumes a PIL-only non-tensor path")
    if shape is None:
        return _unknown(name, "missing tensor input shape")

    if name == "resize":
        return _verify_resize(name, shape, size=size, max_size=max_size)
    if name == "center_crop":
        return _verify_crop(name, shape, size=size)
    if name in {"random_crop", "random_resized_crop"}:
        return _verify_crop(
            name,
            shape,
            size=size,
            random_crop=(name == "random_crop"),
            pad_if_needed=pad_if_needed,
            padding=padding,
        )
    if name in {"five_crop", "ten_crop"}:
        crop = _verify_crop(name, shape, size=size)
        if not crop.ok or crop.output_shape is None:
            return crop
        count = 5 if name == "five_crop" else 10
        return _ok(name, (count,) + crop.output_shape)
    if name == "pad":
        return _verify_pad(name, shape, padding=padding)
    if name == "normalize":
        return _verify_normalize(name, shape, mean=mean, std=std)
    if name in _PRESERVE_ANY_RANK:
        return _ok(name, shape)
    if name in _PRESERVE_RANK2:
        rank_err = _require_rank(name, shape, 2)
        if rank_err is not None:
            return rank_err
        return _ok(name, shape)
    if name in _PRESERVE_RANK3:
        rank_err = _require_rank(name, shape, 3)
        if rank_err is not None:
            return rank_err
        return _ok(name, shape)

    return TransformVerdict(
        False,
        name,
        error_kind="unsupported",
        message=f"torchvision.transforms.v2 transform {transform_name!r} is not modelled",
        unknown_reason="unsupported torchvision.transforms.v2 transform",
    )


__all__ = [
    "Dim",
    "Shape",
    "TransformVerdict",
    "normalized_padding_2d",
    "verify_torchvision_v2_transform",
]
