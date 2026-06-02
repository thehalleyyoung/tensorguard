# Pre-registered user study: does TensorGuard help developers localize and fix shape bugs faster?

_Status: **pre-registered design + executed localization-effort proxy.** The
human randomized controlled trial (RCT) below is specified in full and powered
from the proxy effect; it has **not** yet been run with human participants. This
document is the registration; the proxy results live in
[`../../evaluation/localization_effort.md`](../../evaluation/localization_effort.md)
and are regenerated deterministically by
[`../../evaluation/localization_effort.py`](../../evaluation/localization_effort.py)._

This separation is deliberate. Over-claiming a human study we did not run would
be dishonest; refusing to quantify the localization benefit at all would
under-sell a real, measurable effect. So we (a) **pre-register** the human study
and (b) report a transparent **proxy** that the fault-localization literature
accepts as a stand-in for time-to-localize.

---

## 1. Motivation and research questions

TensorGuard does not merely classify a module as `UNSAFE`; it emits a
counterexample and a **reported source line** (the v5 localizer,
`src/v5/localization.py`). The product hypothesis is that this pointer lets a
developer reach and fix the offending line faster than an error message alone.

- **RQ1 (localization).** Does TensorGuard reduce the number of source lines a
  developer must inspect before reaching the true bug, versus an unaided scan?
- **RQ2 (time, human).** Do developers fix shape/dtype bugs faster, and with
  higher success, when given TensorGuard's diagnostic versus a stock runtime
  traceback?
- **RQ3 (trust calibration).** Does TensorGuard change developers' (over/under)
  reliance on the tool, including on the bugs where it mis-localizes?

RQ1 is answered now by the proxy (Section 4). RQ2-RQ3 require human participants
(Section 3).

---

## 2. The localization-effort proxy (executed, Section 4)

We use the standard *lines-inspected* model from the fault-localization
literature (Parnin & Orso, *Are Automated Debugging Techniques Actually Helping
Programmers?*, ISSTA 2011; and the EXAM-score line of work). For each refuted
real bug in the frozen corpus that carries an author-placed `# BUG` marker
(ground truth independent of TensorGuard's AST walk), we measure, in the **same
unit** (source lines), the effort for two arms:

- **TensorGuard-assisted.** Start at TensorGuard's reported line and scan
  outward until the true bug line is reached:
  `effort = dist_v5 + 1`, where `dist_v5 = |reported_line - bug_line|` is the
  already-committed distance in `reproducibility/localization_marker_only_n30.json`.
- **Unaided.** With no localizer the developer inspects the module's executable
  lines in an arbitrary order; the **expected** number inspected before hitting
  the bug among `N` candidate lines is `(N + 1) / 2` (uniform random inspection
  order -- the neutral model that assumes neither a lucky top-down nor
  adversarial ordering). `N` is the count of executable lines in the repro,
  recomputed deterministically by the harness.

**Primary effect size:** Cliff's delta (non-parametric, paired with the
Mann-Whitney U test) of *unaided* vs *assisted* effort, with a seeded
percentile-bootstrap 95 percent CI. **Secondary:** Cohen's d and small-sample
corrected Hedges' g, and the median per-bug reduction factor `unaided / assisted`
with a bootstrap CI.

### Proxy threats to validity

- It assumes inspecting a line costs the same regardless of arm; real debugging
  has fixed comprehension overhead the proxy ignores (so it may *over*-state the
  ratio). The *direction* (TG points near the bug) is what the proxy soundly
  establishes.
- The uniform-order unaided model is a modelling choice; a developer who happens
  to read top-down and whose bug is early would beat it, one whose bug is late
  would do worse. It is symmetric in expectation, hence neutral.
- The corpus is the 14 refuted, marker-bearing bugs only -- small, and skewed
  toward shape bugs. We therefore use a non-parametric primary effect size and
  **report every bug, including the ones where TensorGuard mis-leads** (assisted
  greater than unaided), rather than filtering them out.

---

## 3. Pre-registered human RCT (specified, not yet run)

- **Design.** Within-subjects, counterbalanced. Each participant fixes a set of
  buggy `nn.Module`s; for each task they are randomly assigned the **Treatment**
  (TensorGuard verdict + reported line + counterexample) or **Control** (the
  stock Python/PyTorch runtime traceback). Task-to-condition assignment is a
  Latin square so every task appears under both conditions across participants.
- **Participants.** Target N = 24 developers with at least 1 year of PyTorch
  experience (see power analysis Section 3.1). Screened for familiarity with
  `nn.Module`.
- **Tasks.** Drawn from the frozen `real_benchmarks` corpus and the
  `experiments_v5/v8/real_bugs_*` repros, balanced across shape/dtype/device and
  across bugs where TensorGuard localizes well vs poorly (to probe RQ3 honestly).
- **Primary outcome.** Time-to-fix (seconds from task start to a fix that makes
  the module run and produce correctly-shaped output), capped at 15 min
  (right-censored; analyzed with a mixed-effects accelerated-failure-time model).
- **Secondary outcomes.** Fix success rate within the cap; lines opened/edited;
  self-reported confidence; reliance on the tool on mis-localized tasks.
- **Analysis.** Mixed-effects model with random intercepts per participant and
  per task; report the standardized effect (Hedges' g) with a 95 percent CI, not
  just a p-value. Pre-registered exclusion: tasks abandoned in under 10 s.
- **Stopping rule.** Fixed N; no optional stopping.

### 3.1 Power analysis (from the proxy)

The proxy yields a *large* standardized effect (Hedges' g and Cliff's delta in
Section 4). Treating g as an optimistic upper bound and deliberately powering for
a **smaller** true effect (g near 0.6, since the proxy ignores fixed
comprehension overhead), a within-subjects two-sided test at alpha = 0.05 and
power = 0.80 needs about 24 paired observations -- achievable with 24
participants times multiple tasks. The exact N is fixed in advance here so the
study cannot be re-powered post hoc.

---

## 4. Executed proxy results

The numbers below are produced and version-pinned by the harness; see the
generated [`localization_effort.md`](../../evaluation/localization_effort.md) for
the per-bug table. They are regenerated byte-identically by the reproducibility
pipeline (`reproducibility/reproduce_all.py --check`).

- **n = 14** refuted, marker-bearing real bugs.
- **Median lines inspected: 2 (TensorGuard) vs 7.5 (unaided).**
- **Median per-bug reduction factor about 3 times (95 percent bootstrap CI
  excludes 1 time).**
- **Cliff's delta large and positive (95 percent bootstrap CI excludes 0).**
- **TensorGuard reduced effort on 12 of 14 bugs and increased it on 2** -- both
  retained and reported, not hidden.

### Interpretation

On this corpus the localization pointer cuts the search space a developer must
inspect by a large, statistically distinguishable margin, while remaining honest
about the minority of bugs where TensorGuard's line is far from the marker. This
motivates -- but does not substitute for -- the human RCT in Section 3.

---

## 5. Reproducing

```bash
cd tensorguard && . .venv/bin/activate
PYTHONPATH=. python3 evaluation/localization_effort.py          # regenerate
PYTHONPATH=. python3 evaluation/localization_effort.py --check  # assert determinism
python3 -m pytest tests/test_localization_effort.py -q          # unit + property tests
```

The effect-size estimators (Cliff's delta, Cohen's d, Hedges' g, bootstrap CI)
live in `src/statistical_rigor.py` and are unit-tested against textbook values.
