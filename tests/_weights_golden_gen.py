"""Generator for the **golden-certificate regression suite** (roadmap step 5).

For a frozen set of small checkpoints — both ``proven_safe`` *good* files and the
full negative corpus of *bad* files (one per finding kind) — we store the
canonical JSON certificate (`dumps_weights_certificate`) as a golden file and
assert byte-identical reproduction.  This locks the *entire* serialized verdict
(tensors, findings, fingerprints, flags), not just the finding kinds, and proves
the certifier is deterministic.

Torch-free: checkpoint bytes are built by hand (reusing ``_weights_bad_gen.pack``)
and the bad cases are imported directly from ``_weights_bad_gen.corpus()`` so the
two corpora can never drift apart.

The certificate's ``filename`` field is normalised to the file's *basename* so
the golden bytes are machine-independent (``file_sha256`` and
``structural_fingerprint`` are already content/structure derived and stable).
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
from typing import List, Optional

import _weights_bad_gen as bad
from _weights_bad_gen import pack, expected_contract

from src.symexec.weights import certify_weights_file, dumps_weights_certificate


def _f32(*words: int) -> bytes:
    return struct.pack("<%dI" % len(words), *words)


def good_cases() -> List[dict]:
    """Well-formed checkpoints that must certify ``proven_safe`` (plus a couple
    that exercise the contract / finite *flags* without producing findings)."""
    out: List[dict] = []

    def add(name, data, *, finite=True, expected=None, partial=False):
        out.append({
            "name": name,
            "file": f"{name}.safetensors",
            "data": data,
            "check_finite": finite,
            "expected": expected,
            "partial": partial,
            "good": True,
        })

    # Single finite F32 tensor.
    add("good_scalar",
        pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
             _f32(0x3F800000, 0x40000000)))  # 1.0, 2.0

    # Two tensors, mixed dtype, contiguous + fully covered.
    multi = pack({"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                  "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}},
                 b"\x01\x02\x03\x04" + _f32(0, 0))
    add("good_multi", multi)

    # Empty state dict — sound (coverage is vacuous for zero tensors).
    add("good_empty", pack({}, b""))

    # Full contract that matches exactly (dtype + shape).
    add("good_contract_full", multi,
        expected={"a": ["U8", [4]], "b": ["F32", [2]]})

    # Partial contract: file has an extra key the contract does not mention;
    # partial=True suppresses contract_unexpected_key, so still proven_safe.
    we = pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
               "extra": {"dtype": "U8", "shape": [1], "data_offsets": [8, 9]}},
              _f32(0, 0) + b"\x00")
    add("good_contract_partial", we, expected={"w": ["F32", [2]]}, partial=True)

    # check_finite=False: the *same bytes* that would be non-finite are accepted
    # because finiteness is not checked (proves the flag changes the verdict).
    add("good_skip_finite",
        pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
             _f32(0x7FC00000, 0x7F800000)),  # NaN, +Inf — but unchecked
        finite=False)

    return out


def bad_cases() -> List[dict]:
    """The negative corpus (step 4), adapted to the golden case schema."""
    out: List[dict] = []
    for e in bad.corpus():
        out.append({
            "name": e["named_kind"],
            "file": e["file"],
            "data": e["data"],
            "check_finite": e["check_finite"],
            "expected": e["expected"],
            "partial": e["partial"],
            "good": False,
        })
    return out


def cases() -> List[dict]:
    """All golden cases (good first, then bad), de-duplicated filenames asserted."""
    cs = good_cases() + bad_cases()
    names = [c["file"] for c in cs]
    assert len(names) == len(set(names)), "duplicate golden filenames"
    return cs


def cert_for_case(dirpath, case) -> "object":
    """Certify a materialised case, with ``filename`` normalised to its basename."""
    path = os.path.join(dirpath, case["file"])
    cert = certify_weights_file(
        path,
        check_finite=case["check_finite"],
        expected=expected_contract(case["expected"]),
        contract_partial=case["partial"],
    )
    return dataclasses.replace(cert, filename=case["file"])


def golden_text(dirpath, case) -> str:
    return dumps_weights_certificate(cert_for_case(dirpath, case)) + "\n"


def manifest_entry(c: dict) -> dict:
    return {
        "name": c["name"],
        "file": c["file"],
        "cert": f"certs/{c['name']}.json",
        "check_finite": c["check_finite"],
        "expected": c["expected"],
        "partial": c["partial"],
        "good": c["good"],
    }


def write_golden(dirpath) -> dict:
    """Materialise every checkpoint + golden certificate + manifest. Deterministic."""
    os.makedirs(dirpath, exist_ok=True)
    certs_dir = os.path.join(dirpath, "certs")
    os.makedirs(certs_dir, exist_ok=True)
    cs = cases()
    for c in cs:
        with open(os.path.join(dirpath, c["file"]), "wb") as fh:
            fh.write(c["data"])
    # Certificates depend on the files existing on disk, so do them after writing.
    for c in cs:
        with open(os.path.join(certs_dir, f"{c['name']}.json"), "w") as fh:
            fh.write(golden_text(dirpath, c))
    manifest = {
        "_doc": "Golden-certificate regression suite for the weights-layer "
                "certifier (roadmap step 5). Each checkpoint's canonical JSON "
                "certificate is frozen and must reproduce byte-identically. "
                "Certificate filename fields are basenames for portability.",
        "entries": [manifest_entry(c) for c in cs],
    }
    with open(os.path.join(dirpath, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest
