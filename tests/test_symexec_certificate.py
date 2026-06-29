"""Tests for proof-carrying bug certificates (Step 94) and replay (Step 95)."""

import dataclasses

import pytest

from src.symexec import (
    BugCertificate,
    CERTIFICATE_VERSION,
    PRECONDITIONS,
    all_verified,
    certificate_from_dict,
    certificate_to_dict,
    certify,
    certify_result,
    dumps_certificates,
    loads_certificates,
    replay,
    replay_all,
    replay_text,
)
from src.symexec.engine import analyze_source

# --------------------------------------------------------------------------- #
# Sources that force each modeled failure class.                              #
# --------------------------------------------------------------------------- #
MATMUL = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = a @ b\n"
)
RESHAPE = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, 3); y = x.reshape(5, 5)\n"
)
DIVZERO = "if __name__ == '__main__':\n    n = 0; y = 10 // n\n"
BROADCAST = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(3, 4); b = torch.randn(3, 5); c = a + b\n"
)
AXIS = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, 3); y = x.sum(dim=5)\n"
)
INDEX = "if __name__ == '__main__':\n    xs = [1, 2, 3]; y = xs[5]\n"
CAT = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = torch.cat([a, b], dim=0)\n"
)
EINSUM = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(4, 5)\n"
    "    c = torch.einsum('ij,jk->ik', a, b)\n"
)
LINEAR = (
    "import torch, torch.nn as nn\n"
    "if __name__ == '__main__':\n"
    "    m = nn.Linear(10, 5); x = torch.randn(2, 7); y = m(x)\n"
)
NEGDIM = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, -3)\n"
)
RETURN_ARITY = (
    "import torch, torch.nn as nn\n"
    "class M(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return x\n"
    "if __name__ == '__main__':\n"
    "    m = M(); a, b = m(torch.randn(2, 3))\n"
)
UNPACK = "if __name__ == '__main__':\n    a, b, c = (1, 2)\n"


def _one_cert(source):
    r = analyze_source(source)
    assert r.bugs, "expected at least one bug"
    return certify(r.bugs[0], "<test>")


# --------------------------------------------------------------------------- #
# Precondition vocabulary.                                                     #
# --------------------------------------------------------------------------- #
def test_preconditions_are_total_and_typed():
    for name, (fn, arity, desc) in PRECONDITIONS.items():
        assert isinstance(name, str) and desc
        assert arity in (1, 2)
        # callable on the right number of args
        sample = (1,) if arity == 1 else (1, 1)
        assert isinstance(fn(sample), bool)


@pytest.mark.parametrize(
    "pred,ops,expected",
    [
        ("dims_equal", (3, 3), True),
        ("dims_equal", (3, 4), False),
        ("broadcast_compat", (3, 1), True),
        ("broadcast_compat", (1, 5), True),
        ("broadcast_compat", (4, 5), False),
        ("numel_match", (6, 6), True),
        ("numel_match", (6, 25), False),
        ("index_in_range", (0, 2), True),
        ("index_in_range", (5, 2), False),
        ("index_in_range", (-1, 2), False),
        ("divisor_nonzero", (0,), False),
        ("divisor_nonzero", (3,), True),
        ("dim_nonneg", (-3,), False),
        ("dim_nonneg", (0,), True),
        ("arity_match", (2, 2), True),
        ("arity_match", (1, 2), False),
    ],
)
def test_precondition_semantics(pred, ops, expected):
    fn = PRECONDITIONS[pred][0]
    assert fn(ops) is expected


# --------------------------------------------------------------------------- #
# Certification: every forced-failure kind yields a verifiable certificate.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source,kind,predicate,operands",
    [
        (MATMUL, "matmul_dim_mismatch", "dims_equal", (3, 4)),
        (RESHAPE, "reshape_size_mismatch", "numel_match", (6, 25)),
        (DIVZERO, "division_by_zero", "divisor_nonzero", (0,)),
        (BROADCAST, "broadcast_mismatch", "broadcast_compat", (4, 5)),
        (AXIS, "axis_out_of_range", "index_in_range", (5, 2)),
        (INDEX, "rank_index_error", "index_in_range", (5, 3)),
        (CAT, "cat_shape_mismatch", "dims_equal", (3, 5)),
        (EINSUM, "einsum_dim_mismatch", "dims_equal", (3, 4)),
        (LINEAR, "layer_dim_mismatch", "feature_match", (10, 7)),
        (RETURN_ARITY, "return_arity_contract", "arity_match", (2, 1)),
        (UNPACK, "unpack_arity_mismatch", "arity_match", (2, 3)),
    ],
)
def test_certify_extracts_witness(source, kind, predicate, operands):
    cert = _one_cert(source)
    assert cert.kind == kind
    assert cert.predicate == predicate
    assert cert.operands == operands
    assert cert.version == CERTIFICATE_VERSION
    # the precondition is genuinely violated on the recovered witness
    assert PRECONDITIONS[predicate][0](operands) is False


def test_certify_claim_only_when_no_witness():
    cert = _one_cert(NEGDIM)
    assert cert.kind == "negative_dimension"
    assert cert.predicate == "dim_nonneg"
    assert cert.operands is None
    assert cert.is_claim_only


def test_certify_result_one_per_bug():
    r = analyze_source(MATMUL)
    certs = certify_result(r, "f.py")
    assert len(certs) == len(r.bugs)
    assert all(isinstance(c, BugCertificate) for c in certs)
    assert all(c.filename == "f.py" for c in certs)


# --------------------------------------------------------------------------- #
# Replay: independent re-derivation.                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source",
    [MATMUL, RESHAPE, DIVZERO, BROADCAST, AXIS, INDEX, CAT, EINSUM, LINEAR,
     RETURN_ARITY, UNPACK],
)
def test_replay_verifies_genuine_bugs(source):
    cert = _one_cert(source)
    res = replay(cert)
    assert res.status == "verified"
    assert res.ok
    assert cert.predicate in res.detail


def test_replay_claim_only_is_unchecked():
    res = replay(_one_cert(NEGDIM))
    assert res.status == "unchecked"
    assert res.ok  # unchecked is not a refutation


def test_replay_refutes_tampered_witness():
    cert = _one_cert(MATMUL)  # dims_equal (3, 4) -> violated
    tampered = dataclasses.replace(cert, operands=(5, 5))  # now satisfied
    res = replay(tampered)
    assert res.status == "refuted"
    assert not res.ok


def test_replay_unknown_predicate_is_unchecked():
    cert = BugCertificate(
        version=CERTIFICATE_VERSION, kind="x", line=1, col=0, function="",
        message="m", predicate="not_a_real_predicate", claim="", operands=(1, 2),
    )
    res = replay(cert)
    assert res.status == "unchecked"


def test_replay_all_and_all_verified():
    r = analyze_source(MATMUL)
    certs = certify_result(r)
    results = replay_all(certs)
    assert len(results) == len(certs)
    assert all_verified(results)


# --------------------------------------------------------------------------- #
# Serialization round-trip + replay straight from the wire form.               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source", [MATMUL, RESHAPE, DIVZERO, NEGDIM, LINEAR])
def test_dict_roundtrip(source):
    cert = _one_cert(source)
    back = certificate_from_dict(certificate_to_dict(cert))
    assert back == cert


def test_dumps_loads_roundtrip_and_replay():
    r = analyze_source(MATMUL)
    certs = certify_result(r, "f.py")
    text = dumps_certificates(certs)
    # deterministic / sorted
    assert dumps_certificates(certs) == text
    restored = loads_certificates(text)
    assert restored == certs
    # replay purely from the serialized form (fully independent of the engine)
    results = replay_text(text)
    assert [x.status for x in results] == ["verified"]


def test_serialized_operands_are_none_for_claim_only():
    cert = _one_cert(NEGDIM)
    d = certificate_to_dict(cert)
    assert d["operands"] is None
    assert certificate_from_dict(d).operands is None


# --------------------------------------------------------------------------- #
# SymResult convenience methods.                                               #
# --------------------------------------------------------------------------- #
def test_symresult_certificates_and_replay_methods():
    r = analyze_source(MATMUL)
    certs = r.certificates("f.py")
    assert certs and all(isinstance(c, BugCertificate) for c in certs)
    results = r.replay("f.py")
    assert all(x.status == "verified" for x in results)


def test_correct_code_yields_no_certificates():
    clean = (
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    a = torch.randn(2, 3); b = torch.randn(3, 5); c = a @ b\n"
    )
    r = analyze_source(clean)
    assert r.certificates() == []
    assert r.replay() == []
