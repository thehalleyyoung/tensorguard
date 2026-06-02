"""Reference implementation of the proposed upstream PyTorch verification hook.

This module is the *executable* companion to ``docs/upstream/pytorch_proposal.md``
(Step 100).  It demonstrates, against real PyTorch, the three API surfaces the
proposal suggests PyTorch could expose so that *every* ``nn.Module`` can be
statically verified by default:

* :func:`verify_nn_module` -- verify a live ``nn.Module`` instance (TensorGuard
  extracts its source, so no annotations are required) and return the
  :class:`AnalysisResult`.
* :func:`attach_verifier` -- register a ``forward_pre_hook`` that runs
  verification **once**, on the first forward, and raises a precise
  :class:`ShapeVerificationError` *before* the deep runtime stack trace that
  PyTorch would otherwise produce.
* :func:`verifiable` -- an opt-in class decorator that attaches the verifier at
  construction time.

The whole point of the upstream proposal is that this is **opt-in and
non-breaking**: a module with no verifier attached behaves exactly as today, and
when the verifier *is* attached it either (a) proves the module shape/device/
dtype-safe and lets the real forward run, or (b) reports a precise diagnostic at
the boundary instead of a runtime crash deep inside ``aten``.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.api import AnalysisResult, verify_architecture


class ShapeVerificationError(RuntimeError):
    """Raised by an attached verifier when a module is proven unsafe."""

    def __init__(self, result: "AnalysisResult", messages: List[str]):
        self.result = result
        self.messages = messages
        joined = "\n  - ".join(messages)
        super().__init__(
            "TensorGuard rejected this nn.Module before execution:\n  - "
            + joined
        )


@dataclass(frozen=True)
class VerifierHandle:
    """Handle to a registered verification hook (mirrors a torch hook handle)."""

    module_id: int
    _torch_handle: object = None

    def remove(self) -> None:
        if self._torch_handle is not None:
            self._torch_handle.remove()


def _bug_messages(result: "AnalysisResult", limit: int = 5) -> List[str]:
    msgs = []
    for bug in (getattr(result, "bugs", None) or [])[:limit]:
        text = getattr(bug, "message", None) or getattr(bug, "description", "")
        # keep only the first line so multi-line Z3 models don't flood output
        msgs.append(str(text).splitlines()[0])
    return msgs


def verify_nn_module(
    module: "object",
    input_shapes: Optional[Dict[str, tuple]] = None,
    soundness_mode: str = "sound",
    **kwargs,
) -> "AnalysisResult":
    """Verify a live ``nn.Module`` instance by extracting and checking its source.

    No source annotations are required -- this is exactly the ergonomic the
    upstream proposal argues for: verification with zero changes to model code.
    """
    source = textwrap.dedent(inspect.getsource(type(module)))
    return verify_architecture(
        source,
        input_shapes=input_shapes,
        soundness_mode=soundness_mode,
        **kwargs,
    )


def attach_verifier(
    module: "object",
    input_shapes: Optional[Dict[str, tuple]] = None,
    soundness_mode: str = "sound",
    raise_on_unsafe: bool = True,
) -> VerifierHandle:
    """Attach a one-shot verification ``forward_pre_hook`` to *module*.

    On the first forward call the module is verified; if the verdict is
    ``UNSAFE`` and *raise_on_unsafe* is set, a :class:`ShapeVerificationError`
    is raised *before* the real (crashing) forward runs.  Subsequent forwards
    skip re-verification.  Returns a :class:`VerifierHandle` so the hook can be
    removed, exactly like a native torch hook.
    """
    state = {"verified": False}

    def _pre_hook(mod, args):
        if state["verified"]:
            return None
        state["verified"] = True
        result = verify_nn_module(
            mod, input_shapes=input_shapes, soundness_mode=soundness_mode
        )
        if raise_on_unsafe and getattr(result, "verdict", None) == "UNSAFE":
            raise ShapeVerificationError(result, _bug_messages(result))
        return None

    handle = module.register_forward_pre_hook(_pre_hook)
    return VerifierHandle(module_id=id(module), _torch_handle=handle)


def verifiable(
    input_shapes: Optional[Dict[str, tuple]] = None,
    soundness_mode: str = "sound",
    raise_on_unsafe: bool = True,
):
    """Opt-in class decorator: attach a verifier at construction time.

    Usage::

        @verifiable(input_shapes={"x": (2, 8)})
        class Net(nn.Module):
            ...
    """

    def _decorate(cls):
        orig_init = cls.__init__

        def _init(self, *args, **kw):
            orig_init(self, *args, **kw)
            attach_verifier(
                self,
                input_shapes=input_shapes,
                soundness_mode=soundness_mode,
                raise_on_unsafe=raise_on_unsafe,
            )

        cls.__init__ = _init
        return cls

    return _decorate


# --------------------------------------------------------------------------- #
# Phase-1 upstream shim: expose the proposed names on the real torch namespace.
# --------------------------------------------------------------------------- #
# Names this shim grafts onto ``torch.nn.utils`` / ``torch.nn`` so user code can
# call the *exact* proposed upstream API (``torch.nn.utils.verify_module``, …)
# today, with no core PyTorch changes. ``install`` is idempotent and fully
# reversible with ``uninstall`` — it never shadows a name PyTorch already
# defines unless ``force=True``.
_INSTALLED_MARK = "_tensorguard_installed"


def install(*, force: bool = False) -> List[str]:
    """Graft the proposed upstream helpers onto the real ``torch`` namespace.

    Adds ``torch.nn.utils.verify_module`` and ``torch.nn.utils.attach_verifier``
    plus the ``torch.nn.verifiable`` decorator and the
    ``torch.nn.ShapeVerificationError`` exception — exactly the Phase-1 surface
    in ``docs/upstream/pytorch_proposal.md``. Returns the list of dotted names
    that were installed. Idempotent; raises if a target already exists in core
    PyTorch unless *force* is set.
    """
    import torch  # local import: this module must import without torch present

    targets = {
        torch.nn.utils: {
            "verify_module": verify_nn_module,
            "attach_verifier": attach_verifier,
        },
        torch.nn: {
            "verifiable": verifiable,
            "ShapeVerificationError": ShapeVerificationError,
        },
    }
    installed: List[str] = []
    for ns, names in targets.items():
        ns_name = getattr(ns, "__name__", str(ns))
        for attr, value in names.items():
            existing = getattr(ns, attr, None)
            if (
                existing is not None
                and not getattr(existing, _INSTALLED_MARK, False)
                and not force
            ):
                raise RuntimeError(
                    f"{ns_name}.{attr} already exists; pass force=True to override"
                )
            try:
                setattr(value, _INSTALLED_MARK, True)
            except (AttributeError, TypeError):
                pass
            setattr(ns, attr, value)
            installed.append(f"{ns_name}.{attr}")
    return installed


def uninstall() -> List[str]:
    """Remove names previously grafted by :func:`install`. Idempotent."""
    import torch

    removed: List[str] = []
    for ns, attrs in (
        (torch.nn.utils, ("verify_module", "attach_verifier")),
        (torch.nn, ("verifiable", "ShapeVerificationError")),
    ):
        ns_name = getattr(ns, "__name__", str(ns))
        for attr in attrs:
            existing = getattr(ns, attr, None)
            if existing is not None and getattr(existing, _INSTALLED_MARK, False):
                delattr(ns, attr)
                removed.append(f"{ns_name}.{attr}")
    return removed

