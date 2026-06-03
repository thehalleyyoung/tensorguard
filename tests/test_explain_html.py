import json
import textwrap

from src.api import verify_architecture
from src.cli.main import ReftypeCliApp
from src.inference_chain import format_explain_html


BAD_LINEAR = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(30, 5)
        def forward(self, x):
            h = self.fc1(x)
            return self.fc2(h)
"""


GOOD_LINEAR = BAD_LINEAR.replace("nn.Linear(30, 5)", "nn.Linear(20, 5)")


def test_html_report_contains_chain_graph_counterexample_badges_and_fix():
    source = textwrap.dedent(BAD_LINEAR)
    result = verify_architecture(source, input_shapes={"x": ("batch", 10)})
    html = format_explain_html(result, source=source, title="demo")

    assert "<svg" in html
    assert "TensorGuard inference-chain graph" in html
    assert "Counterexample" in html
    assert "Proof footprint" in html
    assert "Suggested fixes" in html
    assert "fc1" in html and "fc2" in html
    assert "(batch, 20)" in html
    assert "status-" in html
    assert "nn.Linear(20, 5)" in html


def test_html_report_escapes_user_source():
    source = textwrap.dedent(BAD_LINEAR).replace(
        "return self.fc2(h)",
        "return self.fc2(h)  # <script>alert('x')</script>",
    )
    result = verify_architecture(source, input_shapes={"x": ("batch", 10)})
    html = format_explain_html(result, source=source, title="<unsafe>")

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "&lt;unsafe&gt;" in html


def test_explain_cli_writes_bug_report(tmp_path):
    model = tmp_path / "bad_model.py"
    model.write_text(textwrap.dedent(BAD_LINEAR), encoding="utf-8")
    out = tmp_path / "explain.html"

    rc = ReftypeCliApp().run(
        ["explain", str(model), "-s", "x=batch,10", "-o", str(out), "--no-config"]
    )

    assert rc == 1
    html = out.read_text(encoding="utf-8")
    assert "TensorGuard explain: bad_model.py" in html
    assert "Inference-chain graph" in html
    assert "Counterexample" in html
    assert "Suggested fixes" in html
    assert "fc2" in html


def test_explain_cli_writes_safe_report(tmp_path):
    model = tmp_path / "good_model.py"
    model.write_text(textwrap.dedent(GOOD_LINEAR), encoding="utf-8")
    out = tmp_path / "safe.html"

    rc = ReftypeCliApp().run(
        ["explain", str(model), "-s", "x=batch,10", "-o", str(out), "--no-config"]
    )

    assert rc == 0
    html = out.read_text(encoding="utf-8")
    assert "Verdict:</strong> SAFE" in html
    assert "No bugs reported." in html
    assert "No inference-chain graph is available." in html


def test_explain_cli_rejects_invalid_shape(tmp_path):
    model = tmp_path / "bad_model.py"
    model.write_text(textwrap.dedent(BAD_LINEAR), encoding="utf-8")
    out = tmp_path / "bad.html"

    rc = ReftypeCliApp().run(["explain", str(model), "-s", "not-a-shape", "-o", str(out)])

    assert rc == 1
    assert not out.exists()


def test_explain_command_is_registered_in_help():
    app = ReftypeCliApp()
    assert "explain" in app.COMMANDS
    assert "HTML inference-chain" in app._command_help("explain")
