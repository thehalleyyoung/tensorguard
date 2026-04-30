# Backward-verifier verdict surface on the <=12% subset (round-8 Q5)

On six modules each constructed to exercise tied weights, torch.utils.checkpoint, or both, TG's first-order backward verifier is conservative: it returns ABSTAIN on the checkpoint and tied-weight constructs (the lattice is first-order and does not model recomputation or aliasing).  We observe 0 SILENTLY_INCORRECT_VERIFIED rows: the verdict surface on the <=12% subset is bounded above by ABSTAIN, not by silently-wrong Verified.  This is consistent with the limitation paragraph in the paper (the lattice is first-order and the construct triggers an honest Abstain), and is the strictly-stronger outcome compared to a silently incorrect Verified.

Torch version: `2.9.1`

## Counts

| Verdict bucket | Count |
|---|---|
| `ABSTAIN` | 6 |
| `VERIFIED` | 0 |
| `SILENTLY_INCORRECT_VERIFIED` | 0 |
| `BUGS_REPORTED` | 0 |

## Per-module

| Module | TG verdict | Runtime silently severed? | distinct-storage params |
|---|---|---|---|
| `tied_embedding_decoder` | `ABSTAIN` | False | 1 |
| `checkpointed_two_layer` | `ABSTAIN` | False | 4 |
| `tied_and_checkpointed` | `ABSTAIN` | False | 1 |
| `frozen_backbone_tied_head` | `ABSTAIN` | False | 1 |
| `checkpointed_attention` | `ABSTAIN` | False | 6 |
| `siamese_shared_tower` | `ABSTAIN` | False | 2 |