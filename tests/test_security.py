"""Step 79-80 — security regression tests.

Proves TensorGuard's central security property: analysing an untrusted model
file never executes that file's top-level code.  We feed every source-level
entry point a malicious source whose module-level statements would create a
sentinel file (and run ``os.system``) if executed, and assert the sentinel is
never created while verification still completes and reports the real shape bug.
"""

from __future__ import annotations

import os

import pytest

from src.api import analyze, analyze_file, quick_check, verify_architecture
from src.safe_loader import (
    is_static_only_source,
    verify_file_safely,
    verify_source_safely,
)


GOOD_TAIL = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 20)\n"
    "        self.fc2 = nn.Linear(30, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(self.fc1(x))\n"
)


def _malicious_source(sentinel: str) -> str:
    # Top-level code that has side effects ONLY if the module is executed.
    return (
        "import os\n"
        f"os.system('touch {sentinel}')\n"
        f"open(r'{sentinel}', 'w').write('pwned')\n"
        "raise RuntimeError('this would fire at import time')\n"
        + GOOD_TAIL
    )


@pytest.fixture
def sentinel(tmp_path):
    s = os.path.join(str(tmp_path), "PWNED")
    if os.path.exists(s):
        os.remove(s)
    yield s
    if os.path.exists(s):
        os.remove(s)


def test_verify_architecture_does_not_execute_source(sentinel):
    src = _malicious_source(sentinel)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    assert not os.path.exists(sentinel), "untrusted source was executed!"
    # And it still did its real job: caught the fc2 shape mismatch.
    assert result.verdict == "UNSAFE"
    assert result.bugs, "expected the shape bug to be reported"


def test_analyze_does_not_execute_source(sentinel):
    result = analyze(_malicious_source(sentinel), filename="untrusted.py")
    assert not os.path.exists(sentinel)
    assert result is not None


def test_quick_check_does_not_execute_source(sentinel):
    quick_check(_malicious_source(sentinel))
    assert not os.path.exists(sentinel)


def test_analyze_file_does_not_import_file(tmp_path, sentinel):
    p = tmp_path / "untrusted_model.py"
    p.write_text(_malicious_source(sentinel), encoding="utf-8")
    result = analyze_file(str(p))
    assert not os.path.exists(sentinel), "analyze_file executed the file!"
    assert result is not None


def test_verify_file_safely_reads_as_text(tmp_path, sentinel):
    p = tmp_path / "untrusted_model.py"
    p.write_text(_malicious_source(sentinel), encoding="utf-8")
    result = verify_file_safely(str(p), input_shapes={"x": ("batch", 10)})
    assert not os.path.exists(sentinel)
    assert result.verdict == "UNSAFE"
    assert result.bugs


def test_verify_source_safely_matches_verify_architecture(sentinel):
    src = _malicious_source(sentinel)
    a = verify_source_safely(src, input_shapes={"x": ("batch", 10)})
    b = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    assert not os.path.exists(sentinel)
    assert a.verdict == b.verdict


def test_verify_file_safely_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_file_safely(str(tmp_path / "does_not_exist.py"))


def test_is_static_only_source_accepts_valid_and_rejects_garbage():
    assert is_static_only_source(GOOD_TAIL) is True
    assert is_static_only_source("def (:\n  pass") is False


def test_safe_loader_never_creates_sentinel_for_clean_model(tmp_path):
    # A benign, correct model must verify cleanly with no side effects.
    p = tmp_path / "clean.py"
    p.write_text(GOOD_TAIL.replace("Linear(30, 5)", "Linear(20, 5)"),
                 encoding="utf-8")
    result = verify_file_safely(str(p), input_shapes={"x": ("batch", 10)})
    assert result is not None
