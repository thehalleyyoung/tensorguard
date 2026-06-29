"""Bytecode fallback for the symbolic-execution engine (roadmap Step 81).

The AST modeler dispatches on node type (``Interpreter._ex_<Node>``) and silently
returns ``Top`` for any expression form it has no handler for — ``Set`` literals,
f-strings (``JoinedStr``/``FormattedValue``), and other straight-line pure
expressions.  Those are sound abstentions, but they throw away information that
is often *statically computable*.

This module recovers it with a **bytecode fallback**: it compiles the orphan
expression and evaluates it with a tiny, explicit opcode *stack machine* over a
whitelist of side-effect-free instructions.  Crucially it **never executes the
analyzed program's code** — there is no ``CALL``, no ``LOAD_ATTR``, no
``IMPORT``, no store to anything but the operand stack — so running the engine on
untrusted source can never trigger arbitrary behaviour.  Anything outside the
whitelist (a jump, a call, an unknown name, a nested code object) raises
:class:`_Abstain` and the fallback yields ``None`` (the engine stays at ``Top``).

When the machine *does* reduce an expression to a concrete Python value, the
value is lifted back into the abstract domain with the existing, audited
:func:`src.symexec.concretize.alpha` (an exact singleton abstraction — sound).
The result is that constructs the AST modeler abstained on (e.g. ``{1, 2, 3}``
or ``f"{n}"`` with ``n`` a known constant) now flow a precise value forward,
shrinking the abstain surface without weakening soundness.

The module is torch-free and imports no heavy backend.
"""

from __future__ import annotations

import ast
import copy
import dis
import operator
from typing import Any, Dict, List, Optional

from .values import AbstractValue

__all__ = [
    "fold_expr",
    "fold_to_abstract",
    "abstract_to_concrete",
    "safe_eval_code",
    "NOT_CONCRETE",
]


class _Abstain(Exception):
    """Raised the moment the stack machine leaves the safe constant fragment."""


#: Sentinel for "this abstract value is not a known concrete constant".
NOT_CONCRETE = object()


# --------------------------------------------------------------------------- #
# Operator tables (resolved by the human-readable ``argrepr``, version-stable) #
# --------------------------------------------------------------------------- #

_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "@": None,  # matrix-mul: never constant-foldable here
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
}

_COMPARE = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

#: Magnitudes above which an exponent / shift is refused, so a constant
#: expression in the analyzed source can never make the analyzer compute a
#: multi-megabyte integer (a cheap denial-of-service guard).
_MAX_SHIFT = 256
_MAX_POW_EXP = 256


def _checked_pow(a: Any, b: Any) -> Any:
    if isinstance(a, int) and isinstance(b, int):
        if b > _MAX_POW_EXP or (isinstance(a, int) and abs(a) > (1 << 32) and b > 8):
            raise _Abstain("oversized power")
    return operator.pow(a, b)


def _checked_lshift(a: Any, b: Any) -> Any:
    if isinstance(b, int) and b > _MAX_SHIFT:
        raise _Abstain("oversized shift")
    return operator.lshift(a, b)


_BINARY["**"] = _checked_pow
_BINARY["<<"] = _checked_lshift


# --------------------------------------------------------------------------- #
# The stack machine                                                           #
# --------------------------------------------------------------------------- #

def safe_eval_code(code, names: Dict[str, Any]) -> Any:
    """Evaluate a *straight-line* constant ``code`` object on the operand stack.

    ``names`` maps loadable identifiers (``LOAD_NAME``/``LOAD_FAST``/
    ``LOAD_GLOBAL``) to their known concrete values; any other name load
    abstains.  Returns the value produced by ``RETURN_VALUE``/``RETURN_CONST``.
    Raises :class:`_Abstain` on the first instruction outside the safe fragment
    (including any control-flow jump — only straight-line code is handled).
    """
    # Reject nested code objects (comprehensions, lambdas, genexprs).
    for const in code.co_consts:
        if isinstance(const, type(code)):
            raise _Abstain("nested code object")

    stack: List[Any] = []
    for instr in dis.get_instructions(code):
        op = instr.opname

        if op in ("RESUME", "NOP", "NOT_TAKEN", "PUSH_NULL", "MAKE_CELL", "RETURN_GENERATOR"):
            if op == "RETURN_GENERATOR":
                raise _Abstain(op)
            continue

        if op == "LOAD_SMALL_INT":
            stack.append(instr.arg)
        elif op == "LOAD_CONST":
            val = instr.argval
            if isinstance(val, type(code)):
                raise _Abstain("code const")
            stack.append(val)
        elif op == "RETURN_CONST":
            return instr.argval
        elif op in ("LOAD_NAME", "LOAD_GLOBAL", "LOAD_FAST", "LOAD_FAST_CHECK",
                    "LOAD_FAST_BORROW", "LOAD_DEREF"):
            name = instr.argval
            if name not in names:
                raise _Abstain(f"unknown name {name!r}")
            stack.append(names[name])
        elif op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST"):
            for name in instr.argval:
                if name not in names:
                    raise _Abstain(f"unknown name {name!r}")
                stack.append(names[name])
        elif op == "BUILD_TUPLE":
            n = instr.arg
            items = stack[len(stack) - n:]
            del stack[len(stack) - n:]
            stack.append(tuple(items))
        elif op == "BUILD_LIST":
            n = instr.arg
            items = stack[len(stack) - n:]
            del stack[len(stack) - n:]
            stack.append(list(items))
        elif op == "BUILD_SET":
            n = instr.arg
            items = stack[len(stack) - n:]
            del stack[len(stack) - n:]
            stack.append(set(items))
        elif op == "BUILD_MAP":
            n = instr.arg
            pairs = stack[len(stack) - 2 * n:]
            del stack[len(stack) - 2 * n:]
            stack.append({pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)})
        elif op == "BUILD_CONST_KEY_MAP":
            n = instr.arg
            keys = stack.pop()
            vals = stack[len(stack) - n:]
            del stack[len(stack) - n:]
            stack.append(dict(zip(keys, vals)))
        elif op == "BUILD_STRING":
            n = instr.arg
            parts = stack[len(stack) - n:]
            del stack[len(stack) - n:]
            stack.append("".join(parts))
        elif op in ("SET_UPDATE", "LIST_EXTEND"):
            it = stack.pop()
            target = stack[-1]
            if op == "SET_UPDATE":
                target.update(it)
            else:
                target.extend(it)
        elif op == "FORMAT_SIMPLE":
            stack.append(format(stack.pop()))
        elif op == "FORMAT_VALUE":
            # 3.13- f-string formatting; conversion flags in arg.
            stack.append(format(stack.pop()))
        elif op == "FORMAT_WITH_SPEC":
            spec = stack.pop()
            stack.append(format(stack.pop(), spec))
        elif op == "CONVERT_VALUE":
            conv = {0: lambda x: x, 1: str, 2: repr, 3: ascii}.get(instr.arg, None)
            if conv is None:
                raise _Abstain("convert")
            stack.append(conv(stack.pop()))
        elif op == "BINARY_OP":
            sym = instr.argrepr.rstrip("=")
            if sym == "[]":
                idx = stack.pop()
                container = stack.pop()
                if not isinstance(container, (tuple, list, str, bytes, dict)):
                    raise _Abstain("subscr non-constant container")
                stack.append(container[idx])
                continue
            fn = _BINARY.get(sym)
            if fn is None:
                raise _Abstain(f"binary {instr.argrepr!r}")
            b = stack.pop()
            a = stack.pop()
            stack.append(fn(a, b))
        elif op == "BINARY_SUBSCR":
            idx = stack.pop()
            container = stack.pop()
            if not isinstance(container, (tuple, list, str, bytes, dict)):
                raise _Abstain("subscr non-constant container")
            stack.append(container[idx])
        elif op == "COMPARE_OP":
            fn = _COMPARE.get(instr.argrepr)
            if fn is None:
                raise _Abstain(f"compare {instr.argrepr!r}")
            b = stack.pop()
            a = stack.pop()
            stack.append(fn(a, b))
        elif op == "IS_OP":
            b = stack.pop()
            a = stack.pop()
            stack.append((a is not b) if instr.arg else (a is b))
        elif op == "CONTAINS_OP":
            b = stack.pop()
            a = stack.pop()
            if not isinstance(b, (tuple, list, str, bytes, set, frozenset, dict)):
                raise _Abstain("contains non-constant container")
            stack.append((a not in b) if instr.arg else (a in b))
        elif op == "UNARY_NEGATIVE":
            stack.append(operator.neg(stack.pop()))
        elif op == "UNARY_INVERT":
            stack.append(operator.invert(stack.pop()))
        elif op == "UNARY_NOT":
            stack.append(not stack.pop())
        elif op == "TO_BOOL":
            stack.append(bool(stack.pop()))
        elif op == "COPY":
            stack.append(stack[-instr.arg])
        elif op == "SWAP":
            i = instr.arg
            stack[-1], stack[-i] = stack[-i], stack[-1]
        elif op == "POP_TOP":
            stack.pop()
        elif op == "RETURN_VALUE":
            return stack.pop()
        else:
            # Any jump, call, store, attribute access, import, yield, … -> abstain.
            raise _Abstain(op)
    raise _Abstain("fell off the end without returning")


# --------------------------------------------------------------------------- #
# AST-level entry points                                                      #
# --------------------------------------------------------------------------- #

# Node types whose mere presence guarantees the expression leaves the safe
# constant fragment (a call, an attribute/await/yield, a lambda or a nested
# comprehension).  Checking up front avoids compiling hopeless expressions.
_DISALLOWED = (
    ast.Call,
    ast.Attribute,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Lambda,
    ast.NamedExpr,
    ast.Starred,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _eligible(node: ast.expr, names: Dict[str, Any]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, _DISALLOWED):
            return False
        if isinstance(child, ast.Name) and child.id not in names:
            return False
    return True


def fold_expr(node: ast.expr, names: Dict[str, Any]) -> Any:
    """Statically evaluate ``node`` to a concrete value, or ``NOT_CONCRETE``.

    ``names`` provides concrete values for the free identifiers the expression
    reads.  Returns the concrete Python value when the whole expression reduces
    inside the safe fragment, otherwise the :data:`NOT_CONCRETE` sentinel.
    """
    if not _eligible(node, names):
        return NOT_CONCRETE
    try:
        expr = ast.Expression(body=copy.deepcopy(node))
        ast.fix_missing_locations(expr)
        code = compile(expr, "<bytecode-fallback>", "eval")
        return safe_eval_code(code, names)
    except _Abstain:
        return NOT_CONCRETE
    except Exception:
        # Any genuine evaluation error (ZeroDivisionError, TypeError on mixed
        # operands, …) means the value is not a sound constant here: abstain.
        return NOT_CONCRETE


def fold_to_abstract(
    node: ast.expr, names: Dict[str, Any]
) -> Optional[AbstractValue]:
    """Like :func:`fold_expr` but lifts the result into the abstract domain.

    Returns an :class:`AbstractValue` (an exact singleton abstraction via
    :func:`src.symexec.concretize.alpha`) on success, else ``None``.
    """
    value = fold_expr(node, names)
    if value is NOT_CONCRETE:
        return None
    from .concretize import alpha

    try:
        return alpha(value)
    except Exception:  # pragma: no cover - alpha is total over our value forms
        return None


# --------------------------------------------------------------------------- #
# Abstract -> concrete projection (to build the ``names`` environment)         #
# --------------------------------------------------------------------------- #

def abstract_to_concrete(value: AbstractValue) -> Any:
    """Project a *known-constant* abstract value back to a Python value.

    Returns :data:`NOT_CONCRETE` whenever the abstract value is not pinned to a
    single concrete constant (e.g. a ``Top``, an unknown-range int, or a tensor).
    Containers are projected element-wise and only when every element is itself
    concrete.
    """
    from .values import (
        BoolVal,
        DictVal,
        FloatVal,
        IntVal,
        ListVal,
        NoneVal,
        SetVal,
        StrVal,
        TupleVal,
    )

    if isinstance(value, NoneVal):
        return None
    if isinstance(value, BoolVal):
        return value.const if value.const is not None else NOT_CONCRETE
    if isinstance(value, IntVal):
        c = value.const
        return c if c is not None else NOT_CONCRETE
    if isinstance(value, FloatVal):
        return value.const if value.const is not None else NOT_CONCRETE
    if isinstance(value, StrVal):
        return value.const if value.const is not None else NOT_CONCRETE
    if isinstance(value, TupleVal):
        if not value.exact_len:
            return NOT_CONCRETE
        out = []
        for e in value.elems:
            ce = abstract_to_concrete(e)
            if ce is NOT_CONCRETE:
                return NOT_CONCRETE
            out.append(ce)
        return tuple(out)
    if isinstance(value, ListVal):
        if value.exact_elems is None:
            return NOT_CONCRETE
        out = []
        for e in value.exact_elems:
            ce = abstract_to_concrete(e)
            if ce is NOT_CONCRETE:
                return NOT_CONCRETE
            out.append(ce)
        return out
    if isinstance(value, DictVal):
        if not value.exact_keys:
            return NOT_CONCRETE
        out = {}
        for k, v in value.known:
            cv = abstract_to_concrete(v)
            if cv is NOT_CONCRETE:
                return NOT_CONCRETE
            out[k] = cv
        return out
    if isinstance(value, SetVal):
        return NOT_CONCRETE  # SetVal does not retain exact elements
    return NOT_CONCRETE
