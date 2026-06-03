"""Static contracts for common PyTorch loss functions.

The contracts mirror eager PyTorch for the shape, reduction, and dtype checks
that most often fail inside training steps.  They are intentionally
torch-independent and conservative: symbolic dimensions abstain unless a
rank-level or concrete-dimension mismatch proves a runtime error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

Dim = Any
Shape = Tuple[Dim, ...]


@dataclass(frozen=True)
class LossVerdict:
    """Result of checking a loss invocation."""

    ok: bool
    loss_name: str
    output_shape: Optional[Shape] = None
    output_dtype: Optional[str] = None
    error_kind: Optional[str] = None
    message: Optional[str] = None
    unknown_reason: Optional[str] = None


_DTYPE_ALIASES = {
    "torch.float16": "float16",
    "torch.half": "float16",
    "half": "float16",
    "float16": "float16",
    "torch.bfloat16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.float": "float32",
    "torch.float32": "float32",
    "float": "float32",
    "float32": "float32",
    "torch.double": "float64",
    "torch.float64": "float64",
    "double": "float64",
    "float64": "float64",
    "torch.int64": "int64",
    "torch.long": "int64",
    "long": "int64",
    "int64": "int64",
    "torch.int": "int32",
    "torch.int32": "int32",
    "int": "int32",
    "int32": "int32",
    "torch.int16": "int16",
    "short": "int16",
    "int16": "int16",
    "torch.int8": "int8",
    "char": "int8",
    "int8": "int8",
    "torch.uint8": "uint8",
    "byte": "uint8",
    "uint8": "uint8",
    "torch.bool": "bool",
    "bool": "bool",
    "torch.complex64": "complex64",
    "cfloat": "complex64",
    "complex64": "complex64",
    "torch.complex128": "complex128",
    "cdouble": "complex128",
    "complex128": "complex128",
}

_REAL_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_NON_REAL_FLOAT_DTYPES = frozenset({
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
    "complex64",
    "complex128",
})

_LOSS_ALIASES = {
    "crossentropy": "cross_entropy",
    "crossentropyloss": "cross_entropy",
    "cross_entropy": "cross_entropy",
    "nll": "nll_loss",
    "nllloss": "nll_loss",
    "nll_loss": "nll_loss",
    "mse": "mse_loss",
    "mseloss": "mse_loss",
    "mse_loss": "mse_loss",
    "binary_cross_entropy_with_logits": "binary_cross_entropy_with_logits",
    "bcewithlogits": "binary_cross_entropy_with_logits",
    "bcewithlogitsloss": "binary_cross_entropy_with_logits",
    "kl_div": "kl_div",
    "kldiv": "kl_div",
    "kldivloss": "kl_div",
}


def _canon_dtype(dtype: Optional[Any]) -> Optional[str]:
    if dtype is None:
        return None
    text = str(dtype).strip().lower()
    return _DTYPE_ALIASES.get(text, text.replace("torch.", ""))


def _canon_loss_name(loss_name: Any) -> str:
    name = str(loss_name).split(".")[-1]
    key = name.replace("-", "_").replace(" ", "_")
    lowered = key.lower()
    compact = lowered.replace("_", "")
    return _LOSS_ALIASES.get(lowered, _LOSS_ALIASES.get(compact, lowered))


def _as_shape(shape: Optional[Any]) -> Optional[Shape]:
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        return tuple(shape)
    return tuple(shape)


def _same_shape(
    actual: Shape,
    expected: Shape,
    actual_label: str,
    expected_label: str,
) -> Tuple[bool, Optional[str], bool]:
    if len(actual) != len(expected):
        return (
            False,
            f"{actual_label} rank {len(actual)} does not match "
            f"{expected_label} rank {len(expected)}",
            False,
        )
    unknown = False
    for i, (a, e) in enumerate(zip(actual, expected)):
        if isinstance(a, int) and isinstance(e, int):
            if a != e:
                return (
                    False,
                    f"{actual_label} dim {i} is {a}, expected {e} from "
                    f"{expected_label}",
                    False,
                )
        elif a != e:
            unknown = True
    return True, None, unknown


def _broadcast_shapes(left: Shape, right: Shape) -> Tuple[Optional[Shape], Optional[str], bool]:
    out = []
    unknown = False
    for i in range(1, max(len(left), len(right)) + 1):
        a = left[-i] if i <= len(left) else 1
        b = right[-i] if i <= len(right) else 1
        if a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        elif isinstance(a, int) and isinstance(b, int):
            if a != b:
                axis = max(len(left), len(right)) - i
                return (
                    None,
                    f"operands are not broadcastable at aligned dim {axis}: "
                    f"{a} vs {b}",
                    False,
                )
            out.append(a)
        elif a == b:
            out.append(a)
        else:
            unknown = True
            out.append(a)
    out.reverse()
    return tuple(out), None, unknown


def _reduced_shape(unreduced: Shape, reduction: str, loss_name: str) -> Tuple[Optional[Shape], Optional[str]]:
    valid = {"none", "mean", "sum"}
    if loss_name == "kl_div":
        valid.add("batchmean")
    if reduction not in valid:
        return None, f"{loss_name} does not support reduction={reduction!r}"
    if reduction == "none":
        return unreduced, None
    return (), None


def _dtype_error(loss_name: str, message: str) -> LossVerdict:
    return LossVerdict(False, loss_name, error_kind="dtype", message=message)


def _shape_error(loss_name: str, message: str) -> LossVerdict:
    return LossVerdict(False, loss_name, error_kind="shape", message=message)


def _reduction_error(loss_name: str, message: str) -> LossVerdict:
    return LossVerdict(False, loss_name, error_kind="reduction", message=message)


def _real_float(dtype: Optional[str]) -> bool:
    return dtype in _REAL_FLOAT_DTYPES


def _non_real_float(dtype: Optional[str]) -> bool:
    return dtype in _NON_REAL_FLOAT_DTYPES


def _output_dtype(input_dtype: Optional[str], target_dtype: Optional[str]) -> str:
    if input_dtype is not None and _real_float(input_dtype):
        return input_dtype
    if target_dtype is not None and _real_float(target_dtype):
        return target_dtype
    return "float32"


def _check_class_weight(
    loss_name: str,
    weight_shape: Optional[Shape],
    class_dim: Optional[Dim],
) -> Optional[LossVerdict]:
    if weight_shape is None:
        return None
    if len(weight_shape) != 1:
        return _shape_error(
            loss_name,
            f"{loss_name} weight must be 1D with one entry per class, got "
            f"rank {len(weight_shape)}",
        )
    if isinstance(weight_shape[0], int) and isinstance(class_dim, int) and weight_shape[0] != class_dim:
        return _shape_error(
            loss_name,
            f"{loss_name} weight has {weight_shape[0]} classes but input has "
            f"{class_dim}",
        )
    return None


def _verify_cross_entropy(
    loss_name: str,
    input_shape: Shape,
    target_shape: Shape,
    input_dtype: Optional[str],
    target_dtype: Optional[str],
    reduction: str,
    weight_shape: Optional[Shape],
) -> LossVerdict:
    if len(input_shape) < 1:
        return _shape_error(loss_name, "cross_entropy expects input rank >= 1")
    if _non_real_float(input_dtype):
        return _dtype_error(
            loss_name,
            f"cross_entropy input dtype {input_dtype!r} is invalid; logits must "
            "be real floating point",
        )

    class_dim = input_shape[0] if len(input_shape) == 1 else input_shape[1]
    weight_error = _check_class_weight(loss_name, weight_shape, class_dim)
    if weight_error is not None:
        return weight_error

    class_target_shape = () if len(input_shape) == 1 else input_shape[:1] + input_shape[2:]
    probability_mode = len(target_shape) == len(input_shape)
    class_mode = len(target_shape) == len(class_target_shape)
    unknown = False

    if probability_mode:
        same, msg, maybe = _same_shape(target_shape, input_shape, "target", "input")
        if not same:
            return _shape_error(loss_name, msg or "probability target shape mismatch")
        unknown = unknown or maybe
        if target_dtype is not None and not _real_float(target_dtype):
            return _dtype_error(
                loss_name,
                f"cross_entropy probability targets must be real floating point, "
                f"got {target_dtype!r}",
            )
    elif class_mode:
        same, msg, maybe = _same_shape(
            target_shape, class_target_shape, "target", "input without class dimension"
        )
        if not same:
            return _shape_error(loss_name, msg or "class-index target shape mismatch")
        unknown = unknown or maybe
        if target_dtype is not None and target_dtype != "int64":
            return _dtype_error(
                loss_name,
                f"cross_entropy class-index targets must be int64/Long, got "
                f"{target_dtype!r}",
            )
    else:
        return _shape_error(
            loss_name,
            "cross_entropy target must have class-index shape "
            f"{class_target_shape} or probability shape {input_shape}; got "
            f"{target_shape}",
        )

    out, red_err = _reduced_shape(class_target_shape, reduction, loss_name)
    if red_err:
        return _reduction_error(loss_name, red_err)
    return LossVerdict(
        True,
        loss_name,
        output_shape=out,
        output_dtype=_output_dtype(input_dtype, target_dtype),
        unknown_reason="symbolic dimensions in loss contract" if unknown else None,
    )


def _verify_nll_loss(
    loss_name: str,
    input_shape: Shape,
    target_shape: Shape,
    input_dtype: Optional[str],
    target_dtype: Optional[str],
    reduction: str,
    weight_shape: Optional[Shape],
) -> LossVerdict:
    if len(input_shape) < 1:
        return _shape_error(loss_name, "nll_loss expects input rank >= 1")
    if _non_real_float(input_dtype):
        return _dtype_error(
            loss_name,
            f"nll_loss input dtype {input_dtype!r} is invalid; log-probabilities "
            "must be real floating point",
        )
    class_dim = input_shape[0] if len(input_shape) == 1 else input_shape[1]
    weight_error = _check_class_weight(loss_name, weight_shape, class_dim)
    if weight_error is not None:
        return weight_error
    expected = () if len(input_shape) == 1 else input_shape[:1] + input_shape[2:]
    same, msg, unknown = _same_shape(target_shape, expected, "target", "input without class dimension")
    if not same:
        return _shape_error(loss_name, msg or "nll_loss target shape mismatch")
    if target_dtype is not None and target_dtype != "int64":
        return _dtype_error(
            loss_name,
            f"nll_loss targets must be int64/Long class indices, got "
            f"{target_dtype!r}",
        )
    out, red_err = _reduced_shape(expected, reduction, loss_name)
    if red_err:
        return _reduction_error(loss_name, red_err)
    return LossVerdict(
        True,
        loss_name,
        output_shape=out,
        output_dtype=_output_dtype(input_dtype, target_dtype),
        unknown_reason="symbolic dimensions in loss contract" if unknown else None,
    )


def _verify_mse_loss(
    loss_name: str,
    input_shape: Shape,
    target_shape: Shape,
    input_dtype: Optional[str],
    target_dtype: Optional[str],
    reduction: str,
) -> LossVerdict:
    out_shape, msg, unknown = _broadcast_shapes(input_shape, target_shape)
    if out_shape is None:
        return _shape_error(loss_name, f"mse_loss input/target shape mismatch: {msg}")
    if input_dtype in {"complex64", "complex128"} or target_dtype in {"complex64", "complex128"}:
        return _dtype_error(loss_name, "mse_loss does not support complex dtypes")
    if (
        input_dtype is not None
        and target_dtype is not None
        and not _real_float(input_dtype)
        and not _real_float(target_dtype)
    ):
        return _dtype_error(
            loss_name,
            f"mse_loss requires at least one real floating operand, got "
            f"{input_dtype!r} and {target_dtype!r}",
        )
    out, red_err = _reduced_shape(out_shape, reduction, loss_name)
    if red_err:
        return _reduction_error(loss_name, red_err)
    return LossVerdict(
        True,
        loss_name,
        output_shape=out,
        output_dtype=_output_dtype(input_dtype, target_dtype),
        unknown_reason="symbolic broadcast in loss contract" if unknown else None,
    )


def _verify_bce_with_logits(
    loss_name: str,
    input_shape: Shape,
    target_shape: Shape,
    input_dtype: Optional[str],
    target_dtype: Optional[str],
    reduction: str,
    weight_shape: Optional[Shape],
    pos_weight_shape: Optional[Shape],
) -> LossVerdict:
    same, msg, unknown = _same_shape(target_shape, input_shape, "target", "input")
    if not same:
        return _shape_error(
            loss_name,
            f"binary_cross_entropy_with_logits requires target and input to have "
            f"exactly the same shape: {msg}",
        )
    if _non_real_float(input_dtype):
        return _dtype_error(
            loss_name,
            f"binary_cross_entropy_with_logits input dtype {input_dtype!r} is "
            "invalid; logits must be real floating point",
        )
    if _non_real_float(target_dtype):
        return _dtype_error(
            loss_name,
            f"binary_cross_entropy_with_logits target dtype {target_dtype!r} is "
            "invalid; targets must be real floating point",
        )
    for label, aux_shape in (("weight", weight_shape), ("pos_weight", pos_weight_shape)):
        if aux_shape is None:
            continue
        _, aux_msg, aux_unknown = _broadcast_shapes(input_shape, aux_shape)
        if aux_msg is not None:
            return _shape_error(
                loss_name,
                f"binary_cross_entropy_with_logits {label} is not broadcastable "
                f"to input {input_shape}: {aux_msg}",
            )
        unknown = unknown or aux_unknown
    out, red_err = _reduced_shape(input_shape, reduction, loss_name)
    if red_err:
        return _reduction_error(loss_name, red_err)
    return LossVerdict(
        True,
        loss_name,
        output_shape=out,
        output_dtype=_output_dtype(input_dtype, target_dtype),
        unknown_reason="symbolic dimensions in loss contract" if unknown else None,
    )


def _verify_kl_div(
    loss_name: str,
    input_shape: Shape,
    target_shape: Shape,
    input_dtype: Optional[str],
    target_dtype: Optional[str],
    reduction: str,
) -> LossVerdict:
    out_shape, msg, unknown = _broadcast_shapes(input_shape, target_shape)
    if out_shape is None:
        return _shape_error(loss_name, f"kl_div input/target shape mismatch: {msg}")
    if _non_real_float(input_dtype):
        return _dtype_error(
            loss_name,
            f"kl_div input dtype {input_dtype!r} is invalid; log-probabilities "
            "must be real floating point",
        )
    if _non_real_float(target_dtype):
        return _dtype_error(
            loss_name,
            f"kl_div target dtype {target_dtype!r} is invalid; targets must be "
            "real floating point",
        )
    out, red_err = _reduced_shape(out_shape, reduction, loss_name)
    if red_err:
        return _reduction_error(loss_name, red_err)
    return LossVerdict(
        True,
        loss_name,
        output_shape=out,
        output_dtype=_output_dtype(input_dtype, target_dtype),
        unknown_reason="symbolic broadcast in loss contract" if unknown else None,
    )


def verify_loss(
    loss_name: Any,
    input_shape: Any,
    target_shape: Any,
    *,
    input_dtype: Optional[Any] = None,
    target_dtype: Optional[Any] = None,
    reduction: str = "mean",
    weight_shape: Optional[Any] = None,
    pos_weight_shape: Optional[Any] = None,
    log_target: bool = False,
) -> LossVerdict:
    """Verify a PyTorch loss call against known shape/dtype contracts.

    Parameters mirror the tensor arguments of ``nn.*Loss`` and
    ``torch.nn.functional`` losses.  ``log_target`` is accepted for KLDivLoss
    parity; it does not change the static shape/dtype contract.
    """

    del log_target  # value-level semantics; shape/dtype contract is unchanged
    canonical = _canon_loss_name(loss_name)
    inp = _as_shape(input_shape)
    target = _as_shape(target_shape)
    if inp is None or target is None:
        return LossVerdict(
            False,
            canonical,
            error_kind="unknown",
            message="input and target shapes are required",
            unknown_reason="missing input or target shape",
        )
    reduction = str(reduction or "mean")
    input_dt = _canon_dtype(input_dtype)
    target_dt = _canon_dtype(target_dtype)
    weight = _as_shape(weight_shape)
    pos_weight = _as_shape(pos_weight_shape)

    if canonical == "cross_entropy":
        return _verify_cross_entropy(
            canonical, inp, target, input_dt, target_dt, reduction, weight
        )
    if canonical == "nll_loss":
        return _verify_nll_loss(
            canonical, inp, target, input_dt, target_dt, reduction, weight
        )
    if canonical == "mse_loss":
        return _verify_mse_loss(canonical, inp, target, input_dt, target_dt, reduction)
    if canonical == "binary_cross_entropy_with_logits":
        return _verify_bce_with_logits(
            canonical,
            inp,
            target,
            input_dt,
            target_dt,
            reduction,
            weight,
            pos_weight,
        )
    if canonical == "kl_div":
        return _verify_kl_div(canonical, inp, target, input_dt, target_dt, reduction)

    return LossVerdict(
        False,
        canonical,
        error_kind="unsupported",
        message=f"loss {loss_name!r} is not modelled by verify_loss",
        unknown_reason="unsupported loss function",
    )


__all__ = ["LossVerdict", "verify_loss"]
