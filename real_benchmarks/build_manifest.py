"""Materialize the frozen benchmark corpus and (re)generate ``manifest.json``.

Running ``python -m real_benchmarks.build_manifest`` (or executing this file
directly from inside ``real_benchmarks/``) writes one standalone repro file per
corpus entry into ``clean/`` or ``buggy/`` and records a SHA-256 hash of each
file in ``manifest.json`` along with all ground-truth labels and provenance.

The generated repro files are deterministic functions of ``corpus_def.py``: the
file body is the entry's ``source`` prefixed with a provenance header. Because
the manifest stores a hash of the exact bytes written here, ``load.py`` can
detect any post-freeze drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Support both ``python -m real_benchmarks.build_manifest`` and direct execution.
try:
    from . import corpus_def
except ImportError:  # pragma: no cover - direct-execution fallback
    sys.path.insert(0, THIS_DIR)
    import corpus_def  # type: ignore

MANIFEST_PATH = os.path.join(THIS_DIR, "manifest.json")
VERSION_PATH = os.path.join(THIS_DIR, "VERSION")


def _provenance_header(entry):
    lines = ['"""']
    if entry["label"] == "buggy":
        lines.append(f"TensorGuard benchmark corpus -- BUGGY model ({entry['id']}).")
        lines.append("")
        if entry.get("source_url"):
            lines.append(f"GitHub Issue: {entry['source_url']}")
        else:
            lines.append(f"Provenance: {entry['provenance_type']}")
        if entry.get("expected_error_substring"):
            lines.append(f"Expected Error: {entry['expected_error_substring']}")
    else:
        lines.append(f"TensorGuard benchmark corpus -- CLEAN model ({entry['id']}).")
    lines.append("")
    lines.append(entry["note"])
    lines.append('"""')
    return "\n".join(lines) + "\n\n"


def _shapes_literal(input_shapes):
    items = ", ".join(
        f"{k!r}: ({', '.join(str(d) for d in v)})" + ("," if len(v) == 1 else "")
        for k, v in input_shapes.items()
    )
    return "{" + items + "}"


def render_repro(entry):
    """Return the exact source text of the repro file for ``entry``."""
    header = _provenance_header(entry)
    shapes = f"INPUT_SHAPES = {_shapes_literal(entry['input_shapes'])}\n\n\n"
    return header + shapes + entry["source"]


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(write_files=True):
    entries = corpus_def.all_entries()
    items = []
    for entry in entries:
        subdir = "clean" if entry["label"] == "clean" else "buggy"
        rel_path = f"{subdir}/{entry['id']}.py"
        abs_path = os.path.join(THIS_DIR, subdir, f"{entry['id']}.py")
        text = render_repro(entry)
        if write_files:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w") as fh:
                fh.write(text)
        items.append(
            {
                "id": entry["id"],
                "label": entry["label"],
                "domain": entry["domain"],
                "category": entry["category"],
                "provenance_type": entry["provenance_type"],
                "source_url": entry["source_url"],
                "note": entry["note"],
                "expected_error_substring": entry["expected_error_substring"],
                "input_shapes": entry["input_shapes"],
                "expected_verdict": entry["expected_verdict"],
                "check_devices": entry["check_devices"],
                "check_gradients": entry["check_gradients"],
                "repro_file": rel_path,
                "sha256": _sha256(text),
            }
        )

    n_clean = sum(1 for it in items if it["label"] == "clean")
    n_buggy = sum(1 for it in items if it["label"] == "buggy")
    manifest = {
        "meta": {
            "name": "tensorguard-real-benchmarks",
            "version": corpus_def.CORPUS_VERSION,
            "frozen": True,
            "total": len(items),
            "clean": n_clean,
            "buggy": n_buggy,
            "schema_version": 1,
            "generated_by": "real_benchmarks/build_manifest.py",
            "description": (
                "Frozen, versioned ground-truth corpus of real PyTorch nn.Module "
                "architectures labeled clean (should verify SAFE) or buggy (should "
                "verify UNSAFE). Each file is content-addressed by sha256 so the "
                "corpus cannot silently drift; load.py re-verifies every hash."
            ),
        },
        "items": items,
    }
    if write_files:
        with open(MANIFEST_PATH, "w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        with open(VERSION_PATH, "w") as fh:
            fh.write(corpus_def.CORPUS_VERSION + "\n")
    return manifest


if __name__ == "__main__":
    m = build(write_files=True)
    print(
        f"Wrote {m['meta']['total']} repro files "
        f"({m['meta']['clean']} clean / {m['meta']['buggy']} buggy) "
        f"and manifest v{m['meta']['version']}."
    )
