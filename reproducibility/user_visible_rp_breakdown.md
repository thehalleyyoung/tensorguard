# `user_visible_rp_breakdown.json` — assume-stripped 128/78 split

## Command

The breakdown is derived from re-running TG with the
synthesised config-envelope assume disabled
(``ConfigEnvelope.disable=True`` flag in ``src/v5/symbolic_config.py``)
on the same 488 blocks, then bucketing the new verdicts.  The
script is ``experiments_v5/v8/build_user_visible_rp.py``.

## Numbers

Of the 128 Contract-Violation refutations in the headline triple:

* **0** survive as unconditional Refuted-Proof under the no-assume regime.
* **128** collapse into Library-Warn (conservative warning, outside the
  soundness theorem).

Of the 78 Library-Warn refutations:

* **78** unchanged (LW is by definition not assume-dependent).

Of the 57 Verified blocks:

* **34** remain Verified.
* **23** become Abstain (the synthesised config-envelope assume was the
  only thing constraining ``self.config.X`` to a feasible value).

The user-visible (no-assume) triple is therefore

    34 V / 0 RP / 0 CV / 206 LW / 248 A

— a more honest line for a practitioner running TG without TG
synthesising contracts on their behalf.

## Paper claim citing this artifact

* `eval_v6.tex` "Calibration first" para already reports the user-visible
  triple; this breakdown is the per-bin decomposition the reviewer
  asked for in Q3.
* `review_response.md` Q3.
