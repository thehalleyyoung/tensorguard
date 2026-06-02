# Project Governance

This document describes how the TensorGuard project is run.

## Roles

* **Users** — anyone who uses TensorGuard. Feedback and bug reports are
  contributions.
* **Contributors** — anyone who has had a change merged. There is no obligation
  beyond following `CONTRIBUTING.md` and the code of conduct.
* **Maintainers** — contributors with merge rights who are responsible for
  review, releases, and the project's soundness guarantees. Listed in
  `MAINTAINERS.md`.
* **Lead maintainer** — breaks ties and owns the release/security process. The
  role rotates (see below).

## Decision making

Routine changes are merged by any maintainer after review. We seek
**lazy consensus**: a change with maintainer approval and no sustained objection
may merge. Substantive or contentious changes (API/soundness-affecting,
governance, or anything touching the security boundary) require:

1. an issue or RFC describing the change and its soundness implications,
2. approval from at least two maintainers, and
3. no unresolved objection from another maintainer.

If consensus cannot be reached, the lead maintainer decides, recording the
rationale in the issue.

## Soundness is non-negotiable

A change that would make the verifier report **SAFE** for an unsound program
will not be merged without an explicit, documented, opt-in flag and a proof or
test demonstrating the boundary. Soundness regressions are treated as the
highest-severity bugs.

## Releases

Releases follow `DEPRECATION_POLICY.md` (SemVer). The lead maintainer cuts
releases; any maintainer may propose one. Each release must pass the supported-
surface test suite, the coverage gate, and the numeric-claims audit.

## Maintainer rotation

To avoid burnout and bus-factor risk, the **lead maintainer role rotates on a
fixed cadence** (default: every two release cycles, or quarterly, whichever is
sooner). The rotation order and the current holder are tracked in
`MAINTAINERS.md`. At handover the outgoing lead:

1. transfers any release/signing credentials through the documented secure
   process,
2. confirms the incoming lead can cut a release end to end, and
3. updates `MAINTAINERS.md` with the new holder and effective date.

New maintainers are nominated by an existing maintainer and confirmed by lazy
consensus of the current maintainers. Inactive maintainers (no review activity
for two consecutive cycles) move to *emeritus* status and may be reinstated on
request.

## Changing this document

Amendments to governance follow the substantive-change process above.
