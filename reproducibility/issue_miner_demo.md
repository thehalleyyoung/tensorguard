# Offline issue miner -> corpus candidates (human-in-the-loop)

Mined **4** frozen issue fixtures offline. The miner corroborates every claim against real PyTorch before proposing a candidate, and promotes a candidate to *accepted* only when a human has added its id to the allowlist.

- proposed (corroborated, awaiting human acceptance): **1**
- accepted (in human allowlist): **1**
- rejected (no code / not reproducible / not a bug): **2**

| issue | status | label | reason |
| --- | --- | --- | --- |
| fixture-0001 | accepted | buggy | corroborated buggy repro |
| fixture-0002 | proposed | buggy | corroborated buggy repro |
| fixture-0003 | rejected | None | no python code block |
| fixture-0004 | rejected | None | claimed error not reproduced (forward ran clean) |

**Every corroborated candidate is gated (proposed or accepted): True.** No candidate enters the corpus without a reproduced failure *and* explicit human acceptance.
