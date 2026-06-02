"""Step 42 — automatic inference of forward input specifications.

TensorGuard recovers the ``input_shapes`` mapping statically from the four
conventional places models document it: shape-typed annotations (jaxtyping /
torchtyping), docstring ``Args:`` blocks, example-input tensors, and config
dicts.  ``verify_model`` then runs with **zero** hand-written ``-s`` annotations,
while inference stays conservative (it abstains on ambiguity, so it can only
fill in otherwise-unconstrained inputs — never introduce an unsound shape).
"""
import textwrap

import pytest

from src.input_spec_inference import infer_input_specs
from src.model_checker import verify_model


def _src(body: str) -> str:
    return textwrap.dedent(body)


# --------------------------------------------------------------------------- #
# Annotations.
# --------------------------------------------------------------------------- #
def test_jaxtyping_annotation():
    s = _src("""
        import torch.nn as nn
        from jaxtyping import Float
        from torch import Tensor
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x: Float[Tensor, "batch 8"]):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": ("batch", 8)}
    assert spec.sources["x"] == "annotation"
    assert verify_model(s).safe is True
    bad = s.replace("nn.Linear(8, 4)", "nn.Linear(99, 4)")
    assert verify_model(bad).safe is False


def test_jaxtyping_multi_dim_and_modifiers():
    s = _src("""
        import torch.nn as nn
        from jaxtyping import Float
        from torch import Tensor
        class Net(nn.Module):
            def forward(self, x: Float[Tensor, "*batch 3 224 224"]):
                return x
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": ("batch", 3, 224, 224)}


def test_torchtyping_annotation():
    s = _src("""
        import torch.nn as nn
        from torchtyping import TensorType
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x: TensorType["batch", 3, 8]):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": ("batch", 3, 8)}


# --------------------------------------------------------------------------- #
# Docstrings.
# --------------------------------------------------------------------------- #
def test_docstring_args_block():
    s = _src('''
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                """Run.

                Args:
                    x: shape (batch, 8)
                """
                return self.fc(x)
    ''')
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": ("batch", 8)}
    assert spec.sources["x"] == "docstring"
    assert verify_model(s).safe is True


def test_docstring_bracket_and_type():
    s = _src('''
        import torch.nn as nn
        class Net(nn.Module):
            def forward(self, x):
                """
                x (Tensor): of size [B, 3, 224, 224]
                """
                return x
    ''')
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": ("B", 3, 224, 224)}


# --------------------------------------------------------------------------- #
# Example inputs.
# --------------------------------------------------------------------------- #
def test_class_attr_example_inputs():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            example_inputs = torch.randn(2, 8)
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": (2, 8)}
    assert spec.sources["x"] == "example_inputs"
    assert verify_model(s).safe is True
    bad = s.replace("nn.Linear(8, 4)", "nn.Linear(99, 4)")
    assert verify_model(bad).safe is False


def test_lightning_example_input_array():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            example_input_array = torch.zeros(4, 16)
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": (4, 16)}


def test_example_inputs_method():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 4)
            def example_inputs(self):
                return torch.randn(3, 10)
            def forward(self, x):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": (3, 10)}


def test_module_level_example_inputs():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(12, 4)
            def forward(self, x):
                return self.fc(x)
        example_inputs = torch.randn(5, 12)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": (5, 12)}


def test_tuple_example_inputs_multiple_args():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            example_inputs = (torch.randn(2, 8), torch.randn(2, 8))
            def forward(self, a, b):
                return a + b
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"a": (2, 8), "b": (2, 8)}


# --------------------------------------------------------------------------- #
# Config dicts.
# --------------------------------------------------------------------------- #
def test_config_dict_input_shape():
    s = _src("""
        import torch.nn as nn
        CONFIG = {"name": "net", "input_shape": (2, 8)}
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": (2, 8)}
    assert spec.sources["x"] == "config"


# --------------------------------------------------------------------------- #
# Priority & robustness.
# --------------------------------------------------------------------------- #
def test_annotation_beats_example_inputs():
    s = _src("""
        import torch
        import torch.nn as nn
        from jaxtyping import Float
        from torch import Tensor
        class Net(nn.Module):
            example_inputs = torch.randn(2, 999)
            def forward(self, x: Float[Tensor, "batch 8"]):
                return x
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {"x": ("batch", 8)}
    assert spec.sources["x"] == "annotation"


def test_scalar_param_is_skipped():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            example_inputs = torch.randn(2, 8)
            def forward(self, x, flag: bool = False):
                return x
    """)
    spec = infer_input_specs(s)
    # Only the tensor param gets the example shape; the bool is skipped.
    assert spec.shapes == {"x": (2, 8)}


def test_no_information_returns_empty():
    s = _src("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    spec = infer_input_specs(s)
    assert spec.shapes == {}
    assert not spec


def test_explicit_input_shapes_not_overridden():
    # When the caller passes input_shapes, inference must not interfere.
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            example_inputs = torch.randn(2, 999)
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    # Explicit correct shape -> safe (example_inputs 999 would have been wrong).
    assert verify_model(s, input_shapes={"x": (2, 8)}).safe is True


def test_infer_inputs_flag_disables_inference():
    s = _src("""
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            example_inputs = torch.randn(2, 8)
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    # With inference off and no shapes, behaviour matches the legacy default
    # (no inferred input constraints) — verification must still not crash.
    r = verify_model(s, infer_inputs=False)
    assert r is not None


def test_syntax_error_inference_is_safe():
    assert infer_input_specs("def (:").shapes == {}
