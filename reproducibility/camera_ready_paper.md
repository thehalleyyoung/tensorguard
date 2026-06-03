# Camera-ready paper package (Step 268)

This manifest is generated from the paper-evidence index and committed evidence artifacts. The root `tool_paper.tex` must contain the exact generated LaTeX ledger below; if any backing number drifts, the ledger changes and `python reproducibility/camera_ready_paper.py --check` fails.

- canonical TeX: `tool_paper.tex`
- canonical PDF: `tool_paper.pdf`
- claims: **8**
- all claim artifacts indexed: **True**

| claim | source artifacts | generated statement |
| --- | --- | --- |
| `paper_evidence_index` | `reproducibility/paper_evidence_index.json` | The camera-ready paper is backed by 63 indexed evidence artifacts, including 46 rendered tables. |
| `extended_corpus_score` | `reproducibility/corpus_extended_score.json` | On the extended runtime-validated corpus, sound mode reports TP=153, FP=0, TN=74, FN=0 across 227 cases. |
| `clean_fp_stress` | `reproducibility/fp_stress_eval.json` | Sound mode has 0 false alarms and 0 abstentions on 101 clean stress-test models. |
| `differential_dispatcher` | `reproducibility/differential_dispatcher.json` | Dispatcher differential testing checks 2000 generated modules with 0 false alarms and 0 soundness violations. |
| `mutation_kill_rate` | `reproducibility/mutation_clean_models.json` | Mutation testing kills 756 of 756 runtime-proven mutants in sound mode, with 0 survivors. |
| `stratified_meta_analysis` | `reproducibility/statistical_meta_analysis.json` | The meta-analysis keeps 8 distributions separate, uses 5000 suite-level bootstrap resamples, and forbids naive global pooling: True. |
| `negative_controls` | `evaluation/negative_controls.json` | Negative controls include 6 value-domain cases where TensorGuard catches 0 and the explicit finite-output runtime checker catches 6. |
| `fresh_machine_package` | `reproducibility/artifact_package.json` | The artifact package validates 3 fresh-machine modes (Docker, conda, and source), all passed: True. |

## Generated LaTeX ledger

```tex
% BEGIN GENERATED CAMERA-READY CLAIMS
\begin{center}
\begin{tabular}{p{0.25\linewidth}p{0.66\linewidth}}
\hline
\textbf{Generated claim} & \textbf{Evidence-derived statement} \\
\hline
paper\_evidence\_index & The camera-ready paper is backed by 63 indexed evidence artifacts, including 46 rendered tables. \\
extended\_corpus\_score & On the extended runtime-validated corpus, sound mode reports TP=153, FP=0, TN=74, FN=0 across 227 cases. \\
clean\_fp\_stress & Sound mode has 0 false alarms and 0 abstentions on 101 clean stress-test models. \\
differential\_dispatcher & Dispatcher differential testing checks 2000 generated modules with 0 false alarms and 0 soundness violations. \\
mutation\_kill\_rate & Mutation testing kills 756 of 756 runtime-proven mutants in sound mode, with 0 survivors. \\
stratified\_meta\_analysis & The meta-analysis keeps 8 distributions separate, uses 5000 suite-level bootstrap resamples, and forbids naive global pooling: True. \\
negative\_controls & Negative controls include 6 value-domain cases where TensorGuard catches 0 and the explicit finite-output runtime checker catches 6. \\
fresh\_machine\_package & The artifact package validates 3 fresh-machine modes (Docker, conda, and source), all passed: True. \\
\hline
\end{tabular}
\end{center}
% END GENERATED CAMERA-READY CLAIMS
```
