"""Inferred shape contracts → jaxtyping ``.pyi`` export (even_more Tier 3 #8).

The contract is *advisory/inferential*: it echoes the (annotation-derived) input
abstraction and the analysis-*derived* output abstraction.  These tests pin the
rendering and the two key properties — symbolic dims propagate and the emitted
stub is valid Python — without asserting torch behaviour.
"""

import ast

from src.symexec import (
    FunctionContract,
    TensorSpec,
    contracts_to_pyi,
    infer_contracts,
    to_pyi,
)


def _pyi(src):
    return contracts_to_pyi(src)


def test_jaxtyping_input_dtype_class_preserved():
    src = (
        "from jaxtyping import Float\n"
        "from torch import Tensor\n"
        "def f(x: Float[Tensor, \"b c\"]):\n"
        "    return x\n"
    )
    out = _pyi(src)
    assert 'x: Float[Tensor, "b c"]' in out


def test_symbolic_batch_dim_propagates_through_linear():
    src = (
        "import torch.nn as nn\n"
        "from jaxtyping import Float\n"
        "from torch import Tensor\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(3, 7)\n"
        "    def forward(self, x: Float[Tensor, \"b 3\"]):\n"
        "        return self.fc(x)\n"
    )
    out = _pyi(src)
    # the analysis carries the symbolic batch dim ``b`` and computes feature 7
    assert 'Tensor, "b 7"]' in out


def test_concrete_constructor_output_shape():
    src = "import torch\ndef make():\n    return torch.zeros(4, 5)\n"
    out = _pyi(src)
    assert 'Tensor, "4 5"]' in out


def test_scalar_param_annotation_echoed():
    src = (
        "from jaxtyping import Float\n"
        "from torch import Tensor\n"
        "def g(x: Float[Tensor, \"n\"], k: int):\n"
        "    return x\n"
    )
    out = _pyi(src)
    assert "k: int" in out


def test_unknown_rank_renders_plain_tensor():
    src = (
        "from torch import Tensor\n"
        "def h(x: Tensor):\n"
        "    return x\n"
    )
    out = _pyi(src)
    # an unannotated-rank tensor degrades honestly to a bare Tensor
    assert "x: Tensor" in out


def test_none_return_annotated():
    src = (
        "from torch import Tensor\n"
        "def side(x: Tensor):\n"
        "    return None\n"
    )
    out = _pyi(src)
    assert "-> None" in out


def test_method_emitted_in_class_block():
    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(2, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    out = _pyi(src)
    assert "class M:" in out
    assert "    def forward(self" in out


def test_emitted_pyi_is_valid_python():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "from jaxtyping import Float\n"
        "from torch import Tensor\n"
        "def scale(x: Float[Tensor, \"b c\"], k: float):\n"
        "    return x * k\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(3, 8)\n"
        "    def forward(self, x: Float[Tensor, \"b 3\"]):\n"
        "        return self.fc(x)\n"
    )
    out = _pyi(src)
    ast.parse(out)  # must not raise


def test_syntax_error_yields_empty():
    assert infer_contracts("def f(:\n    pass\n") == []
    assert to_pyi([]).startswith("# Auto-generated")


def test_contract_objects_shape():
    src = "import torch\ndef make():\n    return torch.ones(2, 3)\n"
    cs = infer_contracts(src)
    assert len(cs) == 1 and isinstance(cs[0], FunctionContract)
    assert cs[0].name == "make"
    assert isinstance(cs[0].ret, TensorSpec)
    assert cs[0].ret.axes == ("2", "3")


def test_header_present_and_imports_only_when_needed():
    # a function with no tensor contracts needs no torch import
    src = "def add(a: int, b: int):\n    return a + b\n"
    out = _pyi(src)
    assert out.startswith("# Auto-generated")
    assert "from torch import Tensor" not in out
