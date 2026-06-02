"""Step 65 — keep the 5-minute Getting Started doc executable and true.

Every ```python fenced block in GETTING_STARTED.md that is annotated with a
trailing ``<!-- tg-verify: bug|safe [verify-args] -->`` marker is extracted,
written to a temp file, and run through ``verify_architecture``.  The verdict
must match the marker, so the doc cannot promise a verdict the tool no longer
produces.
"""

import os
import re

import pytest

import torch  # noqa: F401  (ensures the torch import in the snippets resolves)

from src.api import verify_architecture

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOC = os.path.join(_REPO, "GETTING_STARTED.md")

# A python fence immediately followed by an HTML verify marker.
_BLOCK_RE = re.compile(
    r"```python\n(?P<code>.*?)```\s*<!--\s*tg-verify:\s*(?P<verdict>bug|safe)"
    r"(?P<args>[^>]*?)-->",
    re.DOTALL,
)


def _parse_shapes(arg_str: str):
    """Parse ``-s name=d1,d2`` specs out of a marker's argument string."""
    shapes = {}
    for m in re.finditer(r"-s\s+(\S+)", arg_str):
        spec = m.group(1)
        if "=" not in spec:
            continue
        name, dims = spec.split("=", 1)
        parsed = []
        for d in dims.split(","):
            d = d.strip()
            try:
                parsed.append(int(d))
            except ValueError:
                parsed.append(d)
        shapes[name] = tuple(parsed)
    return shapes


def _doc_blocks():
    with open(_DOC, encoding="utf-8") as fh:
        text = fh.read()
    blocks = []
    for m in _BLOCK_RE.finditer(text):
        blocks.append((m.group("verdict"), m.group("args"), m.group("code")))
    return blocks


def test_doc_exists_and_has_annotated_blocks():
    blocks = _doc_blocks()
    # The tutorial relies on at least a couple of checked examples.
    assert len(blocks) >= 3, blocks
    verdicts = {b[0] for b in blocks}
    assert verdicts == {"bug", "safe"}


@pytest.mark.parametrize("verdict,args,code", _doc_blocks())
def test_getting_started_block_matches_marker(verdict, args, code):
    shapes = _parse_shapes(args)
    result = verify_architecture(
        code,
        input_shapes=shapes or None,
        filename="getting_started_snippet.py",
    )
    bug_count = len(getattr(result, "bugs", []) or [])
    if verdict == "bug":
        assert bug_count >= 1, (
            f"doc says this block has a bug but verifier found none:\n{code}"
        )
    else:
        assert bug_count == 0, (
            f"doc says this block is safe but verifier flagged "
            f"{bug_count} bug(s):\n{[b.message for b in result.bugs]}"
        )


def test_decorator_block_actually_runs(tmp_path):
    """The ``@tensorguard.checked`` example must import and execute cleanly.

    The decorator recovers the class source via ``inspect.getsource``, so the
    snippet is written to a real module file and imported (exec'd strings have
    no recoverable source and would make the decorator abstain).
    """
    import importlib.util
    import sys

    with open(_DOC, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(
        r"```python\nimport src as tensorguard\n(?P<code>.*?)```",
        text,
        re.DOTALL,
    )
    assert m, "decorator example block not found in GETTING_STARTED.md"
    src = "import src as tensorguard\n" + m.group("code")
    mod_path = tmp_path / "gs_decorator_snippet.py"
    mod_path.write_text(src, encoding="utf-8")

    sys.path.insert(0, _REPO)  # so `import src` resolves from the snippet
    try:
        spec = importlib.util.spec_from_file_location(
            "gs_decorator_snippet", str(mod_path)
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec so inspect.getsource can locate the class source.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
    finally:
        sys.path.remove(_REPO)

    cls = getattr(module, "SafeConvNet", None)
    assert cls is not None
    result = getattr(cls, "__tensorguard_result__", None)
    assert result is not None, "decorator did not verify the documented class"
    assert not result.bugs
