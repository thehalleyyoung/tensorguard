"""Per-PR overlap matrix for the N=15 unfiltered post-freeze sample.

Reads the verdict stream from real_bugs_unfiltered.md (committed) and emits
the overlap matrix the round-1 reviewer asked for: which PRs each tool
catches, by-PR (not just marginal counts).
"""
import json
from pathlib import Path

# Per-PR catches, transcribed from real_bugs_unfiltered.md (the committed,
# pre-registered post-freeze N=15 verdict stream).
ROWS = [
    # (id, pr, tg_catches_upstream, ft_refutes, pytea_refutes)
    ("rb_pf_001", "huggingface/diffusers#13494",     True,  False, None),
    ("rb_pf_002", "huggingface/transformers#45540",  False, False, None),
    ("rb_pf_003", "huggingface/peft#3165",           True,  False, True),
    ("rb_pf_004", "huggingface/transformers#45473",  True,  False, False),
    ("rb_pf_005", "huggingface/diffusers#13490",     False, False, None),
    ("rb_pf_006", "huggingface/diffusers#13441",     False, False, None),
    ("rb_uf_007", "huggingface/transformers#45602",  False, False, True),
    ("rb_uf_008", "huggingface/diffusers#13520",     True,  True,  None),
    ("rb_uf_009", "huggingface/transformers#45597",  False, True,  None),
    ("rb_uf_010", "huggingface/transformers#45611",  False, False, None),
    ("rb_uf_011", "huggingface/transformers#45624",  False, False, None),
    ("rb_uf_012", "huggingface/diffusers#13561",     True,  False, None),
    ("rb_uf_013", "huggingface/peft#3208",           False, False, False),
    ("rb_uf_014", "huggingface/transformers#45650",  False, False, None),
    ("rb_uf_015", "huggingface/diffusers#13580",     False, False, True),
]

def overlap(a, b):
    """Return |a ∩ b|, |a \\ b|, |b \\ a|, |neither| over the 15 rows."""
    A = {r[0] for r in ROWS if r[a]}
    B = {r[0] for r in ROWS if r[b]}
    return {
        "both":  sorted(A & B),
        "a_only": sorted(A - B),
        "b_only": sorted(B - A),
        "neither": sorted({r[0] for r in ROWS} - A - B),
    }

def main():
    out = {
        "n": 15,
        "tg_catches":   sorted([r[0] for r in ROWS if r[2]]),
        "ft_refutes":   sorted([r[0] for r in ROWS if r[3]]),
        "pytea_refutes":sorted([r[0] for r in ROWS if r[4] is True]),
        "tg_vs_ft":    overlap(2, 3),
        "tg_vs_pytea": overlap(2, 4),
        "ft_vs_pytea": overlap(3, 4),
    }
    here = Path(__file__).parent
    (here / "postfreeze_overlap_matrix.json").write_text(
        json.dumps(out, indent=2, default=lambda x: list(x)))
    rows_md = "\n".join(
        f"| {r[0]} | {r[1]} | {'TG' if r[2] else '.'} | {'FT' if r[3] else '.'} | "
        f"{'Pytea' if r[4] is True else ('.' if r[4] is False else 'n/a')} |"
        for r in ROWS)
    md = f"""# Per-PR overlap matrix (N=15 unfiltered post-freeze)

Round-1 reviewer asked for the joint distribution, not just the marginal
catch counts (TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15).  Below is the
per-PR matrix.  Marginal counts: TG 5/15, FakeTensorMode 2/15,
Pytea 3/15 (silent-skip-corrected).

| PR id | upstream | TG catches | FT refutes | Pytea refutes |
|---|---|:---:|:---:|:---:|
{rows_md}

## Pairwise overlap

| pair | both | TG only | other only | neither |
|---|---|---|---|---|
| TG vs FakeTensorMode | {len(out['tg_vs_ft']['both'])} | {len(out['tg_vs_ft']['a_only'])} | {len(out['tg_vs_ft']['b_only'])} | {len(out['tg_vs_ft']['neither'])} |
| TG vs Pytea           | {len(out['tg_vs_pytea']['both'])} | {len(out['tg_vs_pytea']['a_only'])} | {len(out['tg_vs_pytea']['b_only'])} | {len(out['tg_vs_pytea']['neither'])} |
| FakeTensorMode vs Pytea | {len(out['ft_vs_pytea']['both'])} | {len(out['ft_vs_pytea']['a_only'])} | {len(out['ft_vs_pytea']['b_only'])} | {len(out['ft_vs_pytea']['neither'])} |

## Member lists

* TG ∩ FakeTensorMode = {out['tg_vs_ft']['both'] or '∅'}
* TG ∩ Pytea          = {out['tg_vs_pytea']['both'] or '∅'}
* TG-only (vs FT)     = {out['tg_vs_ft']['a_only'] or '∅'}
* TG-only (vs Pytea)  = {out['tg_vs_pytea']['a_only'] or '∅'}

## Take-away

The TG catch set is **not contained in** the union of FT and Pytea catches,
and the union of FT and Pytea catches is **not contained in** the TG
catch set.  At N=15 the per-tool catch sets are largely disjoint, which
is why the marginal Fisher-exact test does not separate at α=0.05 even
though the point catch rates differ by 13--20 pp.
"""
    (here / "postfreeze_overlap_matrix.md").write_text(md)
    print("Wrote postfreeze_overlap_matrix.{json,md}")

if __name__ == "__main__":
    main()
