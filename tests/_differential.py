"""Roadmap **step 10 — differential oracle harness**.

The empirical proof that the model→weights contract deriver is *partiality-
correct*: for any ``(model_source, construction)`` the derived
:class:`~src.symexec.model_contract.ModelContract` must be a **sound subset** of
the real torch ``state_dict`` — every emitted parameter is present in torch with
an identical shape, and no emitted parameter is absent or wrong.  The contract
may emit *fewer* params (that is partiality, recorded as abstentions); it may
never emit a *wrong* one (that would be a false positive).

This module is the single torch-gated utility the tests call.  It builds on
:mod:`tests._torch_oracle` (the only place torch is constructed) and is pure with
respect to the verdict logic, so the soundness gate can be unit-tested with
hand-built contracts as well as run over the live fixture corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from src.symexec import derive_model_contract
from src.symexec.model_contract import ModelContract

from _torch_oracle import state_dict_shapes


@dataclass(frozen=True)
class Mismatch:
    """One unsound emission: the contract emitted ``name`` with ``emitted`` shape
    but torch registered ``oracle`` (``None`` ⇒ torch doesn't register it)."""

    name: str
    emitted: Tuple[int, ...]
    oracle: Optional[Tuple[int, ...]]


@dataclass(frozen=True)
class SubsetVerdict:
    """The differential verdict for one model.

    ``is_sound`` is the hard property step 10 asserts: ``mismatches`` is empty,
    i.e. the derived contract's ``params`` are a shape-faithful subset of the
    torch ``state_dict``."""

    emitted: int
    registered: int
    matched: Tuple[str, ...]
    mismatches: Tuple[Mismatch, ...]
    missing: Tuple[str, ...]

    @property
    def is_sound(self) -> bool:
        return len(self.mismatches) == 0

    @property
    def fraction(self) -> float:
        if self.registered == 0:
            return 1.0
        return len(self.matched) / self.registered

    def describe(self) -> str:
        if self.is_sound:
            return (f"sound subset: {len(self.matched)}/{self.registered} params "
                    f"emitted, {len(self.missing)} abstained")
        lines = [f"UNSOUND ({len(self.mismatches)} bad emission(s)):"]
        for m in self.mismatches:
            lines.append(f"  {m.name}: emitted {m.emitted} but torch has {m.oracle}")
        return "\n".join(lines)


def subset_verdict(
    contract: ModelContract, oracle: Mapping[str, Tuple[int, ...]]
) -> SubsetVerdict:
    """Pure verdict: compare an already-derived ``contract`` to an ``oracle``
    ``name -> shape`` map.  Torch-free, so it can be unit-tested with deliberately
    unsound hand-built contracts to prove the gate has teeth."""
    norm_oracle: Dict[str, Tuple[int, ...]] = {
        str(k): tuple(int(d) for d in v) for k, v in oracle.items()
    }
    matched = []
    mismatches = []
    for name, shape in contract.params.items():
        shape = tuple(int(d) for d in shape)
        truth = norm_oracle.get(name)
        if truth == shape:
            matched.append(name)
        else:
            mismatches.append(Mismatch(name=name, emitted=shape, oracle=truth))
    missing = [n for n in norm_oracle if n not in contract.params]
    return SubsetVerdict(
        emitted=len(contract.params),
        registered=len(norm_oracle),
        matched=tuple(sorted(matched)),
        mismatches=tuple(sorted(mismatches, key=lambda m: m.name)),
        missing=tuple(sorted(missing)),
    )


def differential_verdict(source: str, construction: str) -> SubsetVerdict:
    """Build the real torch module, derive the contract, and return the verdict.

    Torch-gated: the caller must ``pytest.importorskip('torch')`` first."""
    oracle = state_dict_shapes(source, construction)
    contract = derive_model_contract(source, construction)
    return subset_verdict(contract, oracle)


def assert_sound_subset(source: str, construction: str) -> SubsetVerdict:
    """Assert the derived contract is a sound subset of torch's ``state_dict``,
    returning the verdict for further inspection.  Raises ``AssertionError`` with
    a human-readable diff on any unsound emission."""
    verdict = differential_verdict(source, construction)
    assert verdict.is_sound, f"{construction}:\n{verdict.describe()}"
    return verdict
