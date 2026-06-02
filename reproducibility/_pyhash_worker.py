"""Worker for the cross-Python determinism proof (Step 107).

Scores a deterministic slice of the extended corpus and prints a single
verdict-set SHA-256 to stdout. Invoked by ``cross_python_determinism.py`` in
fresh subprocesses under different ``PYTHONHASHSEED`` values; if the verifier's
verdict depended on hash-randomised dict/set iteration order (the dominant
source of cross-Python and cross-run nondeterminism), these digests would
diverge.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.generators import all_cases  # noqa: E402
from src.api import verify_architecture  # noqa: E402

MODE = "sound"
STRIDE = 5


def main() -> int:
    cases = all_cases()[::STRIDE]
    vmap = {}
    for c in cases:
        r = verify_architecture(
            c.source,
            input_shapes={k: tuple(v) for k, v in c.input_shapes.items()},
            soundness_mode=MODE,
        )
        vmap[c.id] = str(r.verdict)
    payload = json.dumps(vmap, sort_keys=True).encode("utf-8")
    sys.stdout.write(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
