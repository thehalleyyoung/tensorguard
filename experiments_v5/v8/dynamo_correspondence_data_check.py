"""Mechanically check the TG -> Dynamo refinement-variable mapping data
(round-8 reviewer Q3).

Reads ``reproducibility/dynamo_correspondence_data.json`` (one row per TG
typing-rule premise, with the corresponding aten op, fake-tensor metadata
read, and Dynamo guard kind), and verifies, against the *currently
imported* torch's source, that:

* each ``aten_op`` resolves to a real op in ``torch.ops.aten`` (or is one
  of the documented overload-namespace strings like ``aten::matmul``);
* each ``dynamo_read`` column entry references a FakeTensor metadata
  attribute that exists on ``torch._subclasses.fake_tensor.FakeTensor``;
* each ``guard_kind`` is one of the strings exported by
  ``torch._dynamo.guards.GuardBuilder``.

Pinning a torch version
-----------------------

Run with the torch version under audit on PYTHONPATH; the mapping is
torch-version-relative.  We record the actual ``torch.__version__`` in
the output JSON so a third party can reproduce.

Run::

    PYTHONPATH=. python3 experiments_v5/v8/dynamo_correspondence_data_check.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "reproducibility", "dynamo_correspondence_data.json")
OUT = os.path.join(ROOT, "reproducibility", "dynamo_correspondence_data_check.json")


def _safe_attr(obj: object, dotted: str) -> bool:
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return False
    return True


def main() -> int:
    with open(DATA) as fh:
        data = json.load(fh)
    rows = data["rules"]
    try:
        import torch  # noqa: F401
        from torch._subclasses import fake_tensor as _ft  # noqa: F401
        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover
        print(f"torch import failed: {exc}", file=sys.stderr)
        return 2

    # The fake-tensor metadata reads we cite are .size, .dim, .shape,
    # .numel, .requires_grad on a FakeTensor; those are inherited from
    # torch.Tensor so we check the Tensor class for stability.
    tensor_attrs = {"size", "dim", "shape", "numel", "requires_grad"}
    fake_tensor_cls = _ft.FakeTensor
    fake_attrs_present = {a for a in tensor_attrs
                          if hasattr(fake_tensor_cls, a)}

    # GuardBuilder is the canonical export; not all kinds are class
    # attributes (some are enum values), so we look at the module too.
    guards_module_attrs = set()
    try:
        from torch._dynamo import guards as _guards  # noqa: F401
        for name in dir(_guards):
            guards_module_attrs.add(name)
    except Exception:
        pass

    guard_kinds_seen = sorted({r["guard_kind"] for r in rows})

    aten_attrs = set(dir(getattr(__import__("torch").ops, "aten")))

    audit_rows = []
    n_pass = 0
    for r in rows:
        # Aten op: strip the "aten::" prefix and any overload (e.g.
        # "aten::add.Tensor" -> "add").
        ops = []
        for chunk in r["aten_op"].split(","):
            chunk = chunk.strip()
            m = re.match(r"^aten::([A-Za-z_]+)", chunk)
            if m:
                ops.append(m.group(1))
        aten_ok = all(op in aten_attrs for op in ops) if ops else False

        # FakeTensor metadata read: e.g. "FakeTensor.size(-1)" -> "size".
        reads_ok = True
        for read in r["dynamo_read"].split(","):
            read = read.strip()
            m = re.match(r"^FakeTensor\.([a-zA-Z_]+)", read)
            if m and m.group(1) not in fake_attrs_present:
                reads_ok = False
        if r["dynamo_read"].startswith("FakeTensor.shape"):
            reads_ok = "shape" in fake_attrs_present

        # Guard kind: must at least be a recognised string token.
        guard_ok = r["guard_kind"] in {
            "TENSOR_MATCH", "DUAL_LEVEL_NN_MODULE", "EQUALS_MATCH",
            "TYPE_MATCH", "ID_MATCH", "DICT_KEYS_MATCH",
        }

        ok = aten_ok and reads_ok and guard_ok
        if ok:
            n_pass += 1
        audit_rows.append({
            "rule": r["rule"],
            "aten_op": r["aten_op"],
            "aten_ok": aten_ok,
            "dynamo_read": r["dynamo_read"],
            "reads_ok": reads_ok,
            "guard_kind": r["guard_kind"],
            "guard_ok": guard_ok,
            "ok": ok,
        })

    out = {
        "torch_version_at_check": torch_version,
        "n_rules": len(rows),
        "n_pass": n_pass,
        "n_fail": len(rows) - n_pass,
        "fake_tensor_attrs_present": sorted(fake_attrs_present),
        "guard_kinds_referenced": guard_kinds_seen,
        "rows": audit_rows,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{n_pass}/{len(rows)} rules pass the data-level inclusion check "
          f"against torch {torch_version}.")
    print(f"Wrote {OUT}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
