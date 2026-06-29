"""Roadmap step 2 — **faithfulness bridge**: the Lean soundness proofs in
``lean/TensorGuard/WeightsSound.lean`` are only meaningful if their abstract
models match the real certifier.  These randomized property tests pin that
correspondence between the Python implementation (`src/symexec/weights.py`) and
the three proved guarantees:

* **Storage** — `_validate_storage` reports *no* storage finding **iff** the byte
  ranges satisfy the Lean ``tiledFrom 0 rs L`` predicate; and every layout it
  accepts is genuinely alias-free (the empirical witness of
  ``Storage.tiled_no_alias`` / ``tiled_total``).
* **Finiteness** — `_scan_nonfinite` fires **iff** some word is non-finite under
  the IEEE all-ones-exponent definition (``Finite.isNonFinite``), matching
  ``scan_sound`` / ``scan_refute`` / ``all_finite_no_fire``.
* **Contract** — `_check_contract(..., partial=True)` emits exactly the missing /
  shape-mismatch keys of the Lean model and *never* an unexpected-key finding,
  and every key it flags was required (``satisfied_no_missing`` /
  ``satisfied_no_mismatch`` / ``missing_in_req`` / ``mismatch_in_req``).

Deterministic (seeded); torch-free.
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import pytest

from src.symexec.weights import (
    WeightTensorInfo,
    _DTYPE_SIZE,
    _check_contract,
    _scan_nonfinite,
    _validate_storage,
    certify_weights_file,
)

_STORAGE_KINDS = {
    "storage_gap", "storage_overlap", "storage_undercovered", "storage_out_of_bounds",
}


# --------------------------------------------------------------------------- #
# Python mirrors of the proved Lean models.                                     #
# --------------------------------------------------------------------------- #
def _py_tiled_from(ranges, L):
    """Mirror of Lean ``tiledFrom 0 (sorted ranges) L``: contiguous from 0,
    each begin == running cursor, final cursor == L."""
    cursor = 0
    for b, e in sorted(ranges):
        if b != cursor or not (cursor <= e):
            return False
        cursor = e
    return cursor == L


def _py_alias_free(ranges):
    """No two ranges share a byte (mirror of ``inRange`` disjointness)."""
    rs = sorted(ranges)
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if max(rs[i][0], rs[j][0]) < min(rs[i][1], rs[j][1]):
                return False
    return True


def _is_nonfinite_f32(word):
    """Mirror of Lean ``Finite.isNonFinite`` for F32 (exp mask 0xFF at shift 23)."""
    return ((word >> 23) & 0xFF) == 0xFF


# --------------------------------------------------------------------------- #
# Storage: _validate_storage <-> tiledFrom / no-alias.                          #
# --------------------------------------------------------------------------- #
def test_storage_matches_tiled_model_and_is_alias_free():
    rng = random.Random(20240622)
    accepted = 0
    rejected = 0
    for _ in range(3000):
        L = rng.randint(0, 24)
        k = rng.randint(0, 5)
        ranges = []
        for _ in range(k):
            b = rng.randint(0, L + 3)
            e = rng.randint(b, L + 3)  # 0 <= begin <= end
            ranges.append((b, e))
        # U8 (size 1) with shape == span keeps byte-length consistent, so only
        # storage findings can arise.
        infos = [
            WeightTensorInfo(name=f"t{i}", dtype="U8", shape=(e - b,), begin=b, end=e)
            for i, (b, e) in enumerate(ranges)
        ]
        findings = _validate_storage(infos, L)
        storage = [f for f in findings if f.kind in _STORAGE_KINDS]
        # No byte-length finding should appear for U8/span tensors.
        assert all(f.kind != "byte_length_mismatch" for f in findings)

        tiled = _py_tiled_from(ranges, L)
        # The certifier skips coverage when there are no tensors (an empty state
        # dict has nothing to misload or alias) — sound, and the only divergence
        # from the Lean ``tiledFrom`` model, which is for >=1 range.
        accept_expected = (len(ranges) == 0) or tiled
        assert (not storage) == accept_expected, (
            f"storage-finding/tiled mismatch: ranges={ranges} L={L} "
            f"storage={[f.kind for f in storage]} tiled={tiled}"
        )
        if not storage:
            accepted += 1
            # The proved guarantee: an accepted layout is alias-free.
            assert _py_alias_free(ranges)
        else:
            rejected += 1
    # Sanity: the generator actually exercised both verdicts.
    assert accepted > 50 and rejected > 50


def test_storage_known_cases():
    # gap
    infos = [WeightTensorInfo("a", "U8", (4,), 0, 4),
             WeightTensorInfo("b", "U8", (4,), 8, 12)]
    assert {f.kind for f in _validate_storage(infos, 12)} & _STORAGE_KINDS == {"storage_gap"}
    # overlap
    infos = [WeightTensorInfo("a", "U8", (8,), 0, 8),
             WeightTensorInfo("b", "U8", (8,), 4, 12)]
    assert "storage_overlap" in {f.kind for f in _validate_storage(infos, 12)}
    # perfect tiling -> no storage finding
    infos = [WeightTensorInfo("a", "U8", (4,), 0, 4),
             WeightTensorInfo("b", "U8", (4,), 4, 8)]
    assert not ({f.kind for f in _validate_storage(infos, 8)} & _STORAGE_KINDS)


# --------------------------------------------------------------------------- #
# Finiteness: _scan_nonfinite <-> isNonFinite.                                  #
# --------------------------------------------------------------------------- #
def _write_f32(path, words):
    raw = b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words)
    header = {"w": {"dtype": "F32", "shape": [len(words)], "data_offsets": [0, len(raw)]}}
    hb = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + raw)
    return hb


def test_finiteness_scan_matches_ieee_model(tmp_path):
    rng = random.Random(424242)
    fired = 0
    clean = 0
    for t in range(1500):
        n = rng.randint(0, 6)
        words = []
        for _ in range(n):
            if rng.random() < 0.4:
                # Force all-ones exponent (NaN/Inf): exp bits = 0xFF at shift 23.
                mant = rng.randint(0, (1 << 23) - 1)
                sign = rng.randint(0, 1) << 31
                words.append(sign | (0xFF << 23) | mant)
            else:
                # Force a finite exponent (not all ones).
                exp = rng.randint(0, 0xFE)
                mant = rng.randint(0, (1 << 23) - 1)
                sign = rng.randint(0, 1) << 31
                words.append(sign | (exp << 23) | mant)

        p = tmp_path / f"f{t}.safetensors"
        hb = _write_f32(p, words)
        info = WeightTensorInfo("w", "F32", (n,), 0, 4 * n)
        findings = _scan_nonfinite(str(p), [info], len(hb))
        emitted = any(f.kind == "non_finite_values" for f in findings)

        expected = any(_is_nonfinite_f32(w) for w in words)
        assert emitted == expected, (
            f"scan/model mismatch words={words} emitted={emitted} expected={expected}"
        )
        # End-to-end certifier agrees.
        cert = certify_weights_file(str(p))
        assert ("non_finite_values" in cert.finding_kinds) == expected
        if expected:
            fired += 1
        else:
            clean += 1
    assert fired > 50 and clean > 50


# --------------------------------------------------------------------------- #
# Contract: _check_contract(partial=) <-> missing / mismatch / partiality.      #
# --------------------------------------------------------------------------- #
def _mk_infos(entries):
    """entries: list of (name, shape). U8 tensors so files are well-formed."""
    out = []
    cursor = 0
    for name, shape in entries:
        n = 1
        for d in shape:
            n *= d
        out.append(WeightTensorInfo(name, "U8", tuple(shape), cursor, cursor + n))
        cursor += n
    return out


def test_contract_partial_matches_model_and_is_sound():
    rng = random.Random(13371337)
    names = [f"p{i}" for i in range(8)]
    saw_missing = saw_mismatch = 0
    for _ in range(3000):
        def rshape():
            return tuple(rng.randint(1, 4) for _ in range(rng.randint(1, 3)))

        # Contract (required) keys with shapes.
        req_names = rng.sample(names, rng.randint(0, 6))
        req = {n: rshape() for n in req_names}

        # Checkpoint: some required keys (maybe reshaped) + some extra keys.
        have_entries = []
        for n in req_names:
            if rng.random() < 0.8:  # present 80% of the time
                shp = req[n] if rng.random() < 0.6 else rshape()  # maybe wrong shape
                have_entries.append((n, shp))
        for n in set(names) - set(req_names):
            if rng.random() < 0.3:
                have_entries.append((n, rshape()))
        rng.shuffle(have_entries)
        have = dict(have_entries)
        infos = _mk_infos(have_entries)

        expected = {n: (None, req[n]) for n in req}  # shape-only (code-derived) contract

        # --- partial check (the code-derived contract path) ---
        partial = _check_contract(infos, expected, partial=True)
        kinds = {}
        for f in partial:
            kinds.setdefault(f.kind, set()).add(f.name)

        model_missing = {n for n in req if n not in have}
        model_mismatch = {n for n in req if n in have and tuple(have[n]) != tuple(req[n])}

        assert kinds.get("contract_missing_key", set()) == model_missing
        assert kinds.get("contract_shape_mismatch", set()) == model_mismatch
        # Partiality soundness: never flag an extra checkpoint tensor.
        assert "contract_unexpected_key" not in kinds
        # Every flagged key was actually required (missing_in_req / mismatch_in_req).
        for f in partial:
            assert f.name in req

        # --- full (non-partial) check additionally flags unexpected keys ---
        full = _check_contract(infos, expected, partial=False)
        full_unexpected = {f.name for f in full if f.kind == "contract_unexpected_key"}
        assert full_unexpected == {n for n in have if n not in req}

        saw_missing += bool(model_missing)
        saw_mismatch += bool(model_mismatch)
    assert saw_missing > 50 and saw_mismatch > 50


def test_satisfied_contract_has_no_findings(tmp_path):
    # Mirror of satisfied_no_missing + satisfied_no_mismatch on the real check.
    entries = [("a", (2, 3)), ("b", (4,)), ("c", (1, 1, 5))]
    infos = _mk_infos(entries)
    expected = {n: (None, s) for n, s in entries}
    assert _check_contract(infos, expected, partial=True) == []
    assert _check_contract(infos, expected, partial=False) == []
