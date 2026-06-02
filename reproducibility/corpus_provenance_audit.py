"""Deterministic harness: provenance + license-compatibility audit (Step 102).

Proves the extended benchmark corpus is **redistributable** by construction:

1. **Provenance completeness** -- every materialized case has a structured
   provenance record (origin, generator family, inspiration reference, authors,
   license, SPDX).
2. **No copied third-party code** -- every on-disk case source is scanned for
   markers that would indicate copied licensed code (copyright lines, SPDX
   headers, license boilerplate, AGPL/GPL/proprietary notices). The corpus is
   synthetically generated, so none must be present.
3. **License compatibility** -- the dataset is released under the repository's
   MIT license; a small compatibility table records that MIT is permissive and
   imposes no copyleft/attribution-incompatible obligations, so the corpus may
   be redistributed alongside the code.

Only booleans / counts / license strings are recorded, so the artifact is
byte-identical across machines.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.provenance import (  # noqa: E402
    DATASET_LICENSE,
    DATASET_SPDX,
    all_provenance,
)

OUT_JSON = REPO / "reproducibility" / "corpus_provenance_audit.json"
OUT_MD = REPO / "reproducibility" / "corpus_provenance_audit.md"
CASES_DIR = REPO / "corpus_extended" / "cases"

# Markers that would betray copied, separately-licensed third-party source.
_COPY_MARKERS = [
    re.compile(r"copyright", re.IGNORECASE),
    re.compile(r"SPDX-License-Identifier"),
    re.compile(r"\bGPL\b"),
    re.compile(r"\bAGPL\b"),
    re.compile(r"\bLGPL\b"),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"licensed under", re.IGNORECASE),
    re.compile(r"proprietary", re.IGNORECASE),
]

# Permissive licenses that MIT-licensed redistribution is compatible with.
_COMPATIBLE = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "Unlicense"}


def _scan_case_sources() -> dict:
    files = sorted(CASES_DIR.glob("*.py"))
    flagged = []
    for f in files:
        text = f.read_text()
        for pat in _COPY_MARKERS:
            if pat.search(text):
                flagged.append({"file": f.name, "marker": pat.pattern})
                break
    return {"n_files": len(files), "flagged": flagged}


def measure() -> dict:
    prov = all_provenance()
    n = len(prov)

    required = {"id", "origin", "generator", "authors", "license", "spdx",
                "redistributable", "copied_third_party_code"}
    complete = all(required.issubset(p.keys()) for p in prov)
    all_redistributable = all(p["redistributable"] is True for p in prov)
    none_copied = all(p["copied_third_party_code"] is False for p in prov)
    all_synthetic = all(p["origin"] == "synthetic_generated" for p in prov)

    # How many cases cite a public issue as inspiration (reference only).
    n_with_seed = sum(1 for p in prov if p["seed_reference"])

    scan = _scan_case_sources()
    no_copy_markers = len(scan["flagged"]) == 0

    license_compatible = DATASET_LICENSE in _COMPATIBLE

    repo_license_ok = (REPO / "LICENSE").exists()

    redistributable = (
        complete
        and all_redistributable
        and none_copied
        and all_synthetic
        and no_copy_markers
        and license_compatible
        and repo_license_ok
    )

    return {
        "n_cases": n,
        "n_case_files_scanned": scan["n_files"],
        "provenance_complete": complete,
        "all_redistributable": all_redistributable,
        "none_copied_third_party": none_copied,
        "all_synthetic_generated": all_synthetic,
        "n_cases_with_seed_reference": n_with_seed,
        "no_copy_markers_in_sources": no_copy_markers,
        "copy_marker_flags": scan["flagged"],
        "dataset_license": DATASET_LICENSE,
        "dataset_spdx": DATASET_SPDX,
        "license_compatible_with_redistribution": license_compatible,
        "repo_license_present": repo_license_ok,
        "corpus_is_redistributable": redistributable,
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# Extended corpus provenance & license-compatibility audit",
        "",
        f"Audited **{data['n_cases']}** cases "
        f"({data['n_case_files_scanned']} materialized source files). The corpus "
        "is synthetically generated, so it is redistributable by construction: "
        "no third-party source is copied, and it is released under the "
        f"repository's **{data['dataset_license']}** license.",
        "",
        "| property | value |",
        "| --- | --- |",
        f"| provenance record complete for every case | "
        f"{data['provenance_complete']} |",
        f"| every case marked redistributable | {data['all_redistributable']} |",
        f"| no case copies third-party code | "
        f"{data['none_copied_third_party']} |",
        f"| every case synthetically generated | "
        f"{data['all_synthetic_generated']} |",
        f"| cases citing a public issue as inspiration (reference only) | "
        f"{data['n_cases_with_seed_reference']} |",
        f"| no copyright/SPDX/license markers in sources | "
        f"{data['no_copy_markers_in_sources']} |",
        f"| dataset license | `{data['dataset_license']}` "
        f"(SPDX `{data['dataset_spdx']}`) |",
        f"| license permissive / redistribution-compatible | "
        f"{data['license_compatible_with_redistribution']} |",
        f"| repository LICENSE present | {data['repo_license_present']} |",
        "",
        f"**Corpus is redistributable: {data['corpus_is_redistributable']}.** "
        "Every case is original generated work; public issue URLs are recorded "
        "only as inspiration references, with no code copied from them.",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        old_json = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        old_md = OUT_MD.read_text() if OUT_MD.exists() else ""
        if old_json != new_json or old_md != new_md:
            print("MISMATCH: corpus_provenance_audit artifacts differ")
            return 1
        print("OK: corpus_provenance_audit artifacts byte-identical")
        return 0
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    print(f"Wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(run(check=args.check))
