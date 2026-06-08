"""Tests for the vendored data-plane subsystem and its TensorGuard integration.

The data-plane engine (``src/dataplane``) adds refinement-type / non-interference
analysis of the deep-learning *data* layer — orthogonal to TensorGuard's native
shape/dtype (model-plane) verification.  These tests pin that (a) the engine
imports and runs inside TensorGuard, (b) each of the seven data-plane bug axes is
reachable through the unified front door, and (c) findings lower into first-class
TensorGuard :class:`Bug` objects on the public surface.
"""

import pytest

from src.api import Bug, BugCategory
from src.dataplane import (
    analyze_all,
    analyze_data_plane,
    analyze_data_plane_tree,
    analyze_source,
)


def test_value_domain_axis_native():
    src = "import torch.nn as nn\ncrit = nn.BCELoss()\nloss = crit(logits, y)\n"
    viol = [o for o in analyze_source(src, "m.py").violations]
    assert len(viol) == 1
    assert viol[0].axis == "refinement"


@pytest.mark.parametrize("axis,src", [
    ("non_interference",
     "from sklearn.preprocessing import StandardScaler\n"
     "from sklearn.model_selection import train_test_split\n"
     "s = StandardScaler()\n"
     "Xs = s.fit_transform(X)\n"
     "Xtr, Xte = train_test_split(Xs)\n"),
    ("temporal", "df['feat'] = df['price'].shift(-1)\n"),
    ("split", "train_df = df.iloc[0:800]\ntest_df = df.iloc[750:1000]\n"),
    ("group", "import pandas as pd\nfrom sklearn.model_selection import KFold\n"
              "df = pd.read_csv('m.csv')\ng = df.groupby('scaffold')\n"
              "kf = KFold(n_splits=5, shuffle=True)\n"),
    ("join", "merged = events.merge(users, on='user_id', how='left')\n"
             "tr, te = train_test_split(merged, test_size=0.2)\n"),
    ("sampling", "import numpy as np\n"
                 "from torch.utils.data import Dataset, DataLoader\n"
                 "class Aug(Dataset):\n"
                 "    def __getitem__(self, i):\n"
                 "        k = np.random.randint(0, 4)\n"
                 "        return self.data[i], k\n"
                 "    def __len__(self):\n"
                 "        return len(self.data)\n"
                 "loader = DataLoader(ds, batch_size=8, num_workers=4)\n"),
])
def test_each_axis_reachable_through_front_door(axis, src):
    viol = [o for o in analyze_all(src, "m.py").violations if o.axis == axis]
    assert len(viol) >= 1


def test_findings_lower_into_tensorguard_bugs():
    src = "import torch.nn as nn\ncrit = nn.BCELoss()\nloss = crit(logits, y)\n"
    bugs = analyze_data_plane(src, "m.py")
    assert len(bugs) == 1
    bug = bugs[0]
    assert isinstance(bug, Bug)
    assert bug.category is BugCategory.DATA_VALUE_DOMAIN
    assert bug.location.line == 3
    assert bug.severity == "error"


def test_clean_source_yields_no_data_plane_bugs():
    src = ("import torch.nn as nn\n"
           "crit = nn.BCEWithLogitsLoss()\n"   # safe: accepts logits
           "loss = crit(logits, y)\n")
    assert analyze_data_plane(src, "m.py") == []


def test_data_plane_categories_are_public_and_additive():
    # core model-plane categories must remain (additive merge, no removals)
    for name in ("NULL_DEREFERENCE", "DIVISION_BY_ZERO", "TYPE_ERROR",
                 "CEGAR_REFINED_CONTRACT"):
        assert hasattr(BugCategory, name)
    # the seven data-plane categories were added
    for name in ("DATA_VALUE_DOMAIN", "DATA_LEAKAGE", "DATA_TEMPORAL_LEAKAGE",
                 "DATA_GROUP_LEAKAGE", "DATA_JOIN_CARDINALITY",
                 "DATA_SAMPLING_DETERMINISM", "DATA_SPLIT_CONTRACT"):
        assert hasattr(BugCategory, name)


def test_tree_sweep_aggregates(tmp_path):
    (tmp_path / "a.py").write_text(
        "import torch.nn as nn\ncrit = nn.BCELoss()\nloss = crit(logits, y)\n")
    (tmp_path / "b.py").write_text(
        "train_df = df.iloc[0:800]\ntest_df = df.iloc[750:1000]\n")
    bugs = analyze_data_plane_tree(tmp_path)
    cats = {b.category for b in bugs}
    assert BugCategory.DATA_VALUE_DOMAIN in cats
    assert BugCategory.DATA_SPLIT_CONTRACT in cats


def test_top_level_lazy_export():
    import tensorguard as tg
    bugs = tg.analyze_data_plane(
        "import torch.nn as nn\ncrit = nn.BCELoss()\nloss = crit(logits, y)\n", "m.py")
    assert bugs and bugs[0].category is BugCategory.DATA_VALUE_DOMAIN
