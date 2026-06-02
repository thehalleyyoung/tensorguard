# RFC: TensorGuard — a sound static shape/device/dtype verifier as a PyTorch companion tool

- **Status:** Draft (for discussion)
- **Authors:** TensorGuard maintainers (see `MAINTAINERS.md`)
- **Target:** `pytorch/rfcs` / `pytorch-labs` incubation
- **Tracking:** to be assigned on submission

## Summary

We propose adopting **TensorGuard** as an official PyTorch *companion tool*
(initially under `pytorch-labs`, with a path to a supported integration). It is a
**sound** static analyzer that catches shape, device, dtype, training/eval
phase, and gradient-flow errors in `torch.nn.Module`s **before** a single tensor
is allocated — at author time, in CI, and as a `torch.compile`/`torch.export`
pre-pass. When TensorGuard reports SAFE for the declared input contract, that
result is backed by a refinement-type system discharged with an SMT solver
(Z3), not by heuristics.

## Motivation

Shape and device mismatches are among the most common and most time-wasting
PyTorch errors. Today they surface only at runtime — often deep into a training
run, on a remote accelerator, after minutes of setup. PyTorch's own tooling
(`torch.fx`, `torch.export`, `meta` tensors, `ShapeEnv`/symbolic shapes) gives
excellent *dynamic* and *tracing-time* signals, but there is no **sound,
author-time, annotation-light** static checker that:

- runs without executing the model or allocating tensors,
- works directly from source (safe on untrusted model files), and
- distinguishes "provably safe" from "could not prove" rather than guessing.

TensorGuard fills exactly that gap and composes with — rather than competes
with — `torch.compile`/`export`.

## Why this belongs with PyTorch

1. **Closes a well-known papercut** for the whole community, not a niche.
2. **Composes with the compiler stack:** it is a verification pre-pass for
   `torch.compile` and `torch.export`, and can share the symbolic-shape
   vocabulary (`ShapeEnv`, `SymInt`) over time.
3. **Soundness-first** matches PyTorch's correctness expectations: SAFE is a
   guarantee for the declared fragment, with an explicit, documented
   unsound/incomplete boundary (see the limitations section of the project).
4. **Safe on untrusted input:** analysis is purely static (AST + types + SMT);
   model files are never imported or executed (`SECURITY.md`).

## Proposed scope (and non-goals)

**In scope:** static verification of `nn.Module` architectures for shape,
device, dtype, phase, and gradient-flow properties; CLI, pytest plugin, pre-commit
hook, GitHub Action, and `torch.compile`/`export` integration; a labeled
real-world bug benchmark.

**Non-goals (initially):** full Python semantics, arbitrary dynamic control flow
that the declared fragment excludes, numerical-accuracy analysis, and replacing
runtime checks. The tool **abstains** (reports "could not prove") rather than
guessing outside its fragment.

## Design overview

TensorGuard lifts module source to a typed IR, assigns **refinement types**
(base type + predicates over symbolic dimensions/device/dtype/phase), and checks
the transfer functions of a 5-theory product domain with Z3, using
CEGAR/unsat-core refinement. Soundness of the core transfer functions is argued
on paper and, where feasible, in Lean (`lean/`). The result for each module is
SAFE (proved), UNSAFE (a concrete counterexample / bug), or ABSTAIN.

## Integration surface

- **Library API:** `verify_architecture`, `verify_file_safely`, `verify_module`,
  `guarded_compile`, `verify_exported_program` (typed, `py.typed`).
- **Tooling:** CLI (`tensorguard`), pytest plugin, pre-commit hook, GitHub
  Action, JSON/JUnit/SARIF reporters, baseline/suppression for incremental
  adoption.
- **Framework hooks:** Lightning `Callback` and Hugging Face `TrainerCallback`.

## Governance & maintenance plan

This is the crux of an incubation request, so we state it concretely:

- **Governance:** documented in `GOVERNANCE.md` — lazy-consensus for routine
  changes, two-maintainer sign-off for soundness/API-affecting changes, and a
  rule that no change may make the verifier report SAFE for an unsound program
  without an explicit, tested, opt-in boundary.
- **Maintainers & rotation:** `MAINTAINERS.md` lists maintainers and a
  **rotating lead-maintainer** role (fixed cadence, handover checklist) to
  manage bus-factor risk.
- **Release engineering:** SemVer with a published `DEPRECATION_POLICY.md`;
  reproducible wheels; conda-forge recipe; multi-OS / multi-Python (3.9–3.13) /
  multi-`torch` CI plus a non-blocking `torch`-nightly early-warning job.
- **Quality gates:** a coverage gate on the supported surface, a
  precision/recall dashboard gated against a frozen baseline, a numeric-claims
  audit for documentation, and a security boundary enforced by tests.
- **Compatibility commitment:** track stable `torch` releases within one minor
  version; the nightly job surfaces upstream breakage early.

## Compatibility & risks

- **Symbolic-shape divergence:** as `torch`'s `ShapeEnv`/`SymInt` evolve, we plan
  to align vocabulary and, where possible, consume upstream symbolic shapes.
  *Mitigation:* nightly CI + a thin adapter layer.
- **Fragment gaps (incompleteness):** the tool abstains rather than unsoundly
  claiming SAFE; recall is improved over time and tracked on a public
  leaderboard.
- **Maintenance load:** addressed by the rotation and the automated gates above.

## Adoption plan

1. Incubate under `pytorch-labs`; keep the standalone PyPI package.
2. Land a documented `torch.compile`/`export` pre-pass integration.
3. Publish the labeled bug benchmark as a citable community asset.
4. Revisit deeper integration (shared symbolic-shape layer) once the API is
   stable and the empirical study (Phase 9) is published.

## Alternatives considered

- **Runtime/`meta`-tensor checks only:** valuable but late and require execution;
  not safe on untrusted files and not author-time.
- **Heuristic linters:** fast but unsound — they cannot offer a SAFE guarantee,
  which is TensorGuard's central contribution.
- **Stay fully external:** viable, but a PyTorch-adjacent home maximizes
  community reach and compiler-stack alignment.

## Request

We ask the PyTorch maintainers to consider TensorGuard for `pytorch-labs`
incubation and to identify a sponsor to advise on aligning with the
`torch.compile`/`export` and symbolic-shape roadmaps.
