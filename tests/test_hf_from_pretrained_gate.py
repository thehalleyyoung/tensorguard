"""Step 170 — Hugging Face ``from_pretrained`` gate, proven against real loaders.

``guarded_from_pretrained`` loads a checkpoint through the genuine
``PreTrainedModel.from_pretrained`` machinery (offline, from a directory written
by ``save_pretrained``) and verifies the *returned* model before handing it
back.  We prove end-to-end:

* a clean custom ``PreTrainedModel`` subclass is loaded, verified, and then
  genuinely **trained** by a real ``transformers.Trainer`` (3 optimizer steps);
* a buggy subclass (literal in/out mismatch) raises ``TensorGuardViolation`` —
  the misbuilt model never escapes the loader;
* ``verify_pretrained_model`` warn/ignore modes behave; a non-loader argument is
  rejected with ``TypeError``.

No network: every checkpoint is round-tripped through a temp directory.
"""

from __future__ import annotations

import tempfile
import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

transformers = pytest.importorskip("transformers")
from transformers import (  # noqa: E402
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from src.integrations.hf_hook import (  # noqa: E402
    TensorGuardViolation,
    guarded_from_pretrained,
    verify_pretrained_model,
)


class TinyTGConfig(PretrainedConfig):
    model_type = "tinytg"


class GoodTGModel(PreTrainedModel):
    config_class = TinyTGConfig

    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.post_init()

    def forward(self, x, labels=None):
        logits = self.fc2(self.fc1(x))
        loss = ((logits - labels) ** 2).mean()
        return {"loss": loss, "logits": logits}


class BadTGModel(PreTrainedModel):
    config_class = TinyTGConfig

    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)  # expects 30, gets 20 -> real shape bug
        self.post_init()

    def forward(self, x):
        return self.fc2(self.fc1(x))


_SHAPES = {"x": ("b", 10), "labels": ("b", 5)}


class _DS(Dataset):
    def __init__(self, n=16):
        self.x = torch.randn(n, 10)
        self.y = torch.randn(n, 5)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "labels": self.y[i]}


def _saved(cls):
    d = tempfile.mkdtemp()
    cls(TinyTGConfig()).save_pretrained(d)
    return d


def test_guarded_from_pretrained_loads_and_verifies_clean():
    model = guarded_from_pretrained(GoodTGModel, _saved(GoodTGModel), input_shapes=_SHAPES)
    assert isinstance(model, PreTrainedModel)
    out = model(torch.randn(4, 10), labels=torch.randn(4, 5))
    assert out["logits"].shape == (4, 5)


def test_guarded_from_pretrained_raises_on_buggy_checkpoint():
    with pytest.raises(TensorGuardViolation):
        guarded_from_pretrained(BadTGModel, _saved(BadTGModel), input_shapes={"x": ("b", 10)})


def test_non_loader_argument_rejected():
    with pytest.raises(TypeError):
        guarded_from_pretrained(object(), "ignored")


def test_verify_pretrained_model_warn_mode_does_not_raise():
    model = BadTGModel.from_pretrained(_saved(BadTGModel))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verify_pretrained_model(model, input_shapes={"x": ("b", 10)}, on_violation="warn")
    assert any("verification issue" in str(w.message) for w in caught)


def test_verify_pretrained_model_ignore_mode_returns_result():
    model = BadTGModel.from_pretrained(_saved(BadTGModel))
    result = verify_pretrained_model(model, input_shapes={"x": ("b", 10)}, on_violation="ignore")
    assert str(result.verdict).upper().endswith("UNSAFE")


def test_gated_clean_model_trains_end_to_end_with_real_trainer():
    """The model that passes the gate genuinely trains under a real Trainer."""
    model = guarded_from_pretrained(GoodTGModel, _saved(GoodTGModel), input_shapes=_SHAPES)
    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            output_dir=out,
            max_steps=3,
            per_device_train_batch_size=4,
            report_to=[],
            logging_strategy="no",
            save_strategy="no",
            use_cpu=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Trainer(model=model, args=args, train_dataset=_DS()).train()
    assert result.global_step == 3
    assert result.training_loss == result.training_loss  # finite (not NaN)
