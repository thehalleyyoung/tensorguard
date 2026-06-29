"""Generator for the **negative corpus** of deliberately-malformed safetensors
checkpoints (roadmap step 4).  Torch-free: builds raw bytes by hand.

Each entry produces one ``*.safetensors`` file that triggers exactly one *named*
``WeightsFinding`` kind (the file is named after that kind), together with the
**exact** set of finding kinds the certifier emits for it (usually a singleton;
``storage_out_of_bounds`` necessarily co-occurs with ``storage_undercovered``).

The four ``contract_*`` kinds are *not* file-intrinsic — they need an expected
``name -> (dtype, shape)`` contract — so those entries carry a ``expected``
contract (and ``partial`` flag) in addition to a (well-formed) file.

This module is imported by both the one-time materialiser and the regeneration
consistency test, so the committed corpus is reproducible byte-for-byte.
"""

from __future__ import annotations

import json
import struct
from typing import List, Optional


def pack(header_obj, buffer_bytes: bytes) -> bytes:
    """Pack a safetensors file: 8-byte LE header length, JSON header, data.

    ``sort_keys`` makes the bytes deterministic regardless of dict order."""
    hb = json.dumps(header_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(hb)) + hb + buffer_bytes


def _f32_words(*words: int) -> bytes:
    return struct.pack("<%dI" % len(words), *words)


def corpus() -> List[dict]:
    """The negative corpus: one entry per finding kind.

    Each entry: ``file``, raw ``data`` bytes, exact ``kinds`` set, ``check_finite``,
    optional ``expected`` contract (``name -> [dtype|null, shape]``) and ``partial``.
    """
    out: List[dict] = []

    def add(stem, data, kinds, *, finite=True, expected=None, partial=False):
        out.append({
            "file": f"{stem}.safetensors",
            "data": data,
            "kinds": sorted(kinds),
            "named_kind": stem,
            "check_finite": finite,
            "expected": expected,
            "partial": partial,
        })

    # --- frame / header malformations (caught before any tensor is typed) ---
    add("malformed_frame", struct.pack("<Q", 1000) + b"{}", ["malformed_frame"])
    add("malformed_entry", pack({"w": 5}, b""), ["malformed_entry"])
    add("unknown_dtype",
        pack({"w": {"dtype": "F7", "shape": [1], "data_offsets": [0, 4]}}, b"\x00" * 4),
        ["unknown_dtype"])
    add("malformed_shape",
        pack({"w": {"dtype": "F32", "shape": "bad", "data_offsets": [0, 4]}}, b"\x00" * 4),
        ["malformed_shape"])
    add("malformed_offsets",
        pack({"w": {"dtype": "U8", "shape": [1], "data_offsets": [4, 2]}}, b""),
        ["malformed_offsets"])

    # --- storage / byte-length malformations ---
    add("byte_length_mismatch",
        pack({"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 4]}}, b"\x00" * 4),
        ["byte_length_mismatch"])
    add("storage_out_of_bounds",
        pack({"w": {"dtype": "U8", "shape": [100], "data_offsets": [0, 100]}}, b"\x00" * 4),
        ["storage_out_of_bounds", "storage_undercovered"])  # OOB always breaks coverage
    add("storage_gap",
        pack({"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
              "b": {"dtype": "U8", "shape": [4], "data_offsets": [8, 12]}}, b"\x00" * 12),
        ["storage_gap"])
    add("storage_overlap",
        pack({"a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
              "b": {"dtype": "U8", "shape": [8], "data_offsets": [4, 12]}}, b"\x00" * 12),
        ["storage_overlap"])
    add("storage_undercovered",
        pack({"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}, b"\x00" * 8),
        ["storage_undercovered"])

    # --- numerical (NaN 0x7FC00000, +Inf 0x7F800000) ---
    add("non_finite_values",
        pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
             _f32_words(0x7FC00000, 0x7F800000)),
        ["non_finite_values"])

    # --- contract (code <-> data): well-formed file + an expected contract ---
    good_w = pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\x00" * 8)
    good_we = pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                    "extra": {"dtype": "U8", "shape": [1], "data_offsets": [8, 9]}},
                   b"\x00" * 9)
    add("contract_missing_key", good_w, ["contract_missing_key"],
        expected={"w": ["F32", [2]], "absent": [None, [3]]})
    add("contract_unexpected_key", good_we, ["contract_unexpected_key"],
        expected={"w": [None, [2]]})
    add("contract_shape_mismatch", good_w, ["contract_shape_mismatch"],
        expected={"w": [None, [5]]})
    add("contract_dtype_mismatch", good_w, ["contract_dtype_mismatch"],
        expected={"w": ["F16", [2]]})

    return out


def manifest_entry(e: dict) -> dict:
    """The JSON-serialisable manifest record (no raw bytes)."""
    return {
        "file": e["file"],
        "kinds": e["kinds"],
        "named_kind": e["named_kind"],
        "check_finite": e["check_finite"],
        "expected": e["expected"],
        "partial": e["partial"],
    }


def write_corpus(dirpath) -> dict:
    """Materialise every file plus ``manifest.json`` into ``dirpath``.

    Returns the manifest dict.  Deterministic / idempotent."""
    import os
    os.makedirs(dirpath, exist_ok=True)
    entries = corpus()
    for e in entries:
        with open(os.path.join(dirpath, e["file"]), "wb") as fh:
            fh.write(e["data"])
    manifest = {
        "_doc": "Negative corpus for the weights-layer certifier (roadmap step 4). "
                "Each file triggers exactly its listed finding kinds; none is ever "
                "proven_safe. contract_* files pair a well-formed checkpoint with an "
                "expected contract.",
        "entries": [manifest_entry(e) for e in entries],
    }
    with open(os.path.join(dirpath, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def expected_contract(raw: Optional[dict]):
    """Convert a manifest ``expected`` record to the certifier's contract dict
    ``name -> (dtype_or_None, shape_tuple)``."""
    if raw is None:
        return None
    return {name: (dt, tuple(shape)) for name, (dt, shape) in raw.items()}
