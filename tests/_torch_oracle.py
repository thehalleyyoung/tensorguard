"""Torch-gated **oracle** for the model→weights contract coverage metric
(roadmap step 9) and the differential harness (step 10).

This is the *only* place torch is allowed near the weights-contract tests, and
it lives strictly in the test tree — never on TensorGuard's trust path.  It
builds the *real* ``nn.Module`` from ``(source, construction)`` and reports the
authoritative ``state_dict`` ``name -> shape`` map that the symbolic contract is
measured against.

The shared :data:`FIXTURES` corpus is the single source of truth for both the
coverage dashboard/baseline test and any future differential-oracle test, so the
two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Fixture:
    """One ``(source, construction)`` model fixture with a stable ``name``."""

    name: str
    source: str
    construction: str


def state_dict_shapes(source: str, construction: str) -> Dict[str, Tuple[int, ...]]:
    """Build the real torch module and return its ``state_dict`` name→shape map.

    Imported lazily so the module can be collected even when torch is absent
    (the caller is expected to ``pytest.importorskip('torch')`` first).
    """
    import torch  # noqa: F401  (ensures a clear error if torch is missing)

    namespace: Dict[str, object] = {}
    exec(compile(source, "<fixture>", "exec"), namespace)
    model = eval(construction, namespace)  # noqa: S307 — trusted test fixtures
    return {
        name: tuple(int(d) for d in tensor.shape)
        for name, tensor in model.state_dict().items()
    }


# --------------------------------------------------------------------------- #
# Fixture corpus — one entry per layer family + partial/abstaining cases.       #
# --------------------------------------------------------------------------- #
_TINY = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

class Tiny(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.block = Block(d)
        self.head = nn.Linear(d, vocab, bias=False)
"""

_CONVNET = """
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, 2 * c, kernel_size=3, bias=False)
        self.head = nn.Linear(2 * c, 10)
"""

_BIASFREE = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 8)
        self.ln = nn.LayerNorm(8, elementwise_affine=True)
"""

# A ModuleList built from a list-comprehension over ``range(n)``.  With a
# *constant* ``n`` (the ``M(n=4, d=8)`` construction below) step-13 comprehension
# enumeration resolves every ``layers.<i>.*`` child to FULL coverage; a *symbolic*
# ``n`` would instead abstain on ``layers`` (covered by test_symexec_comprehension).
_PARTIAL = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)

class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
"""

_CONV1D = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(c, 2 * c, kernel_size=5)
        self.bn = nn.BatchNorm1d(2 * c, track_running_stats=True)
"""

# An nn.ModuleList / nn.ModuleDict built from explicit literals: every registered
# child is statically enumerable, so the contract should reach FULL coverage with
# PyTorch-faithful ``blocks.<i>.*`` / ``heads.<key>.*`` naming (roadmap step 11).
_MODULELIST = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d), Block(d), Block(d)])
        self.head = nn.Linear(d, d, bias=False)
"""

_MODULEDICT = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)

class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({"attn": Block(d), "mlp": Block(d)})
        self.norm = nn.LayerNorm(d)
"""

# A GPT-style stack built by a constant-trip ``for`` loop that ``.append``s a
# block per iteration (roadmap step 12).  Bounded precise unrolling must resolve
# all N registered children to FULL coverage with ``blocks.<i>.*`` naming.
_LOOP_BUILT = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

class GPT(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n):
            self.blocks.append(Block(d))
        self.head = nn.Linear(d, d, bias=False)
"""

# nn.Sequential built from an OrderedDict — the children carry their *declared*
# names (``net.fc1.*`` / ``net.bn.*`` / ``net.fc2.*``), not ``0/1/2`` (step 14).
_NAMED_SEQ = """
from collections import OrderedDict
import torch.nn as nn

class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d, 2 * d)),
            ("bn", nn.BatchNorm1d(2 * d)),
            ("act", nn.ReLU()),
            ("fc2", nn.Linear(2 * d, d)),
        ]))
        self.head = nn.Linear(d, d, bias=False)
"""


_CONDITIONAL = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self, d, use_proj=True, deep=False):
        super().__init__()
        self.fc = nn.Linear(d, d)
        if use_proj:
            self.proj = nn.Linear(d, 2 * d)
        if deep:
            self.block = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
"""


_INHERITED = """
import torch.nn as nn

class BaseAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
class MyAttention(BaseAttention):
    def __init__(self, d, n_head):
        super().__init__(d, n_head)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
"""


FIXTURES: Tuple[Fixture, ...] = (
    Fixture("tiny_transformer", _TINY, "Tiny(vocab=100, d=8)"),
    Fixture("convnet2d", _CONVNET, "ConvNet(c=16)"),
    Fixture("biasfree_linear", _BIASFREE, "M()"),
    Fixture("partial_modulelist", _PARTIAL, "M(n=4, d=8)"),
    Fixture("conv1d_bn", _CONV1D, "M(c=6)"),
    Fixture("modulelist_literal", _MODULELIST, "M(d=8)"),
    Fixture("moduledict_literal", _MODULEDICT, "M(d=8)"),
    Fixture("loopbuilt_modulelist", _LOOP_BUILT, "GPT(n=6, d=8)"),
    Fixture("named_sequential", _NAMED_SEQ, "M(d=8)"),
    Fixture("conditional_submodule", _CONDITIONAL, "M(d=8, use_proj=True, deep=True)"),
    Fixture("inherited_attention", _INHERITED, "MyAttention(d=8, n_head=2)"),
)


# --------------------------------------------------------------------------- #
# Extended corpus for the differential oracle harness (roadmap step 10).        #
# These deliberately exercise *soundness stressors* where a naive deriver would #
# emit a parameter torch does NOT register (or with a wrong shape):             #
#   * BatchNorm ``affine=False``        -> NO weight/bias                        #
#   * BatchNorm ``track_running_stats=False`` -> NO running_*/num_batches        #
#   * LayerNorm ``elementwise_affine=False``  -> NO params at all                #
#   * Conv ``groups=g``                 -> weight is (out, in//g, *k)            #
#   * Conv tuple / asymmetric kernel    -> weight kernel dims preserved          #
#   * Conv3d, Conv1d, BatchNorm1d/3d, tuple LayerNorm normalized_shape           #
# --------------------------------------------------------------------------- #
_STRESS = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_no_affine = nn.BatchNorm2d(8, affine=False)
        self.bn_no_stats = nn.BatchNorm2d(8, track_running_stats=False)
        self.ln_no_affine = nn.LayerNorm(8, elementwise_affine=False)
        self.ln_tuple = nn.LayerNorm((4, 8))
        self.conv_groups = nn.Conv2d(8, 8, kernel_size=3, groups=4)
        self.conv_tuple_k = nn.Conv2d(3, 6, kernel_size=(3, 5), bias=False)
"""

_CONV3D = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, 2 * c, kernel_size=(1, 3, 3))
        self.bn = nn.BatchNorm3d(2 * c)
        self.head = nn.Linear(2 * c, 4, bias=False)
"""

_DEEP_NEST = """
import torch.nn as nn

class Inner(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

class Mid(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = Inner(d)
        self.b = Inner(d)

class Outer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.tok = nn.Embedding(32, d)
        self.mid = Mid(d)
        self.proj = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
"""

_MLP_ONLY = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.Dropout(0.1), nn.Linear(d, 1)
        )
"""

# Everything in FIXTURES plus the stressors — the harness runs over ALL of these.
DIFFERENTIAL_FIXTURES: Tuple[Fixture, ...] = FIXTURES + (
    Fixture("stress_norm_conv", _STRESS, "M()"),
    Fixture("conv3d_stack", _CONV3D, "M(c=4)"),
    Fixture("deep_nested", _DEEP_NEST, "Outer(d=8)"),
    Fixture("mlp_sequential", _MLP_ONLY, "M(d=12)"),
)

