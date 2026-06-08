"""TensorGuard interface layer — static verification of the *discrete text/token
interface* of LLM applications.

This subpackage was merged from PromptABI. TensorGuard's core verifies the
**tensor plane** of a PyTorch program (shapes, broadcasting, devices, dtypes,
training-loop numerics, axis roles). This layer verifies the complementary
**interface plane**: the tokenizer / chat-template / stop-sequence /
tool-calling / constrained-decoding boundary where *text data* crosses into and
out of an LLM. Both planes share TensorGuard's discipline — sound-or-abstain
decisions over finite abstractions (automata, SMT), CPU-only, no weights, with
replayable witnesses.

Capabilities exposed here (all self-contained; z3 used where available):

* ``torch_data_misuse`` — silent PyTorch *data-pipeline* bugs that complement the
  training-loop checks: worker-RNG duplication, ``drop_last`` on eval loaders,
  fit-before-split leakage.
* ``constrained_decoding_feasibility`` — does a guided-decoding grammar admit a
  tokenization (decoder-stall / expressivity-gap decisions).
* ``surface_ban_soundness`` — does an id-level ``bad_words``/``suppress_tokens``
  ban actually prevent a forbidden surface (product-automaton, unbounded).
* ``streaming_stop_soundness`` — can a server truncate output exactly at a stop
  string (overshoot / split-stop, unbounded).
* ``smt_tokenizer_forgery`` / ``role_separator_forgery`` /
  ``control_token_injection`` / ``normalization_confusables`` — role/control-token
  forgery at the tokenizer/template boundary.
"""

from __future__ import annotations

# Data-pipeline plane (complements src.training_loop_checks).
from .torch_data_misuse import (
    TORCH_DATA_MISUSE_VERSION,
    TorchDataFinding,
    TorchDataMisuseKind,
    TorchDataMisuseReport,
    analyze_torch_data_file,
    analyze_torch_data_source,
    analyze_torch_data_tree,
    render_torch_data_report_json,
    render_torch_data_report_text,
)

# Constrained-decoding feasibility.
from .constrained_decoding_feasibility import (
    load_vocab_surfaces_from_tokenizer_json,
    prove_decoding_feasibility,
    regex_to_dfa,
    render_decoding_feasibility_report_json,
    render_decoding_feasibility_report_text,
)

# Surface-ban soundness.
from .surface_ban_soundness import (
    SURFACE_BAN_SOUNDNESS_VERSION,
    BanSoundnessStatus,
    SurfaceBanReport,
    load_id_surfaces_from_tokenizer_json,
    naive_substring_suppression,
    prove_surface_ban,
    render_surface_ban_report_json,
    render_surface_ban_report_text,
)

# Streaming stop-sequence soundness.
from .streaming_stop_soundness import (
    STREAMING_STOP_SOUNDNESS_VERSION,
    StopHazardKind,
    StopSoundnessStatus,
    StreamingStopReport,
    prove_streaming_stop,
    render_streaming_stop_report_json,
    render_streaming_stop_report_text,
)

__all__ = [
    "TORCH_DATA_MISUSE_VERSION",
    "TorchDataFinding",
    "TorchDataMisuseKind",
    "TorchDataMisuseReport",
    "analyze_torch_data_file",
    "analyze_torch_data_source",
    "analyze_torch_data_tree",
    "render_torch_data_report_json",
    "render_torch_data_report_text",
    "load_vocab_surfaces_from_tokenizer_json",
    "prove_decoding_feasibility",
    "regex_to_dfa",
    "render_decoding_feasibility_report_json",
    "render_decoding_feasibility_report_text",
    "SURFACE_BAN_SOUNDNESS_VERSION",
    "BanSoundnessStatus",
    "SurfaceBanReport",
    "load_id_surfaces_from_tokenizer_json",
    "naive_substring_suppression",
    "prove_surface_ban",
    "render_surface_ban_report_json",
    "render_surface_ban_report_text",
    "STREAMING_STOP_SOUNDNESS_VERSION",
    "StopHazardKind",
    "StopSoundnessStatus",
    "StreamingStopReport",
    "prove_streaming_stop",
    "render_streaming_stop_report_json",
    "render_streaming_stop_report_text",
]
