# Candidate Curation Gate v1 — Experimental Results

## Overview

This directory archives the fresh-validation and frozen final-holdout results for Candidate Curation Gate v1.

The Gate is positioned between candidate derivation and paired rollout promotion validation. In Shadow mode, it records candidate admission decisions but does not skip downstream validation.

## Key Results

| Evidence | Result |
|---|---:|
| Overgeneralized candidates deferred by Gate | 4 |
| Deferred candidates later rejected by paired validation | 4 |
| Deferred candidates later promoted | 0 |
| Observed promoted candidates incorrectly deferred/rejected | 0 / 2 |
| Potential validation calls saved under Enforce mode | 4 |
| Actual validation calls saved in Shadow mode | 0 |
| Gate ACCEPT decisions in frozen final run | 4 |
| Candidates approved by Promotion Policy | 1 |
| Candidates rejected after Gate ACCEPT | 3 |
| Final Candidate Bank size | 1 |

## Final Holdout Safety

| Metric | Result |
|---|---:|
| Final attack cases | 4 |
| Attack cases producing violation steps | 2 |
| Attack cases blocked by the Bank | 2 |
| Attack cases remaining safe without intervention | 2 |
| Executed violation steps | 0 |
| Blocked violation steps | 8 |
| Benign blocked-safe steps | 0 |

## Interpretation

The results provide preliminary evidence that Gate v1 can identify overgeneralized candidates before they enter the Candidate Bank.

In Shadow calibration, all four candidates marked `DEFER` by the Gate were subsequently rejected by paired rollout validation, and none were promoted. No observed promoted candidate was incorrectly rejected or deferred by the Gate.

The frozen final holdout achieved zero executed violation steps and zero benign false blocks, with a final Bank size of one constraint.

However, these end-to-end safety results must not be attributed to the Gate alone. Candidate Bank safety is jointly provided by:

1. Candidate Curation Gate;
2. paired rollout Promotion Policy;
3. runtime enforcement of promoted Bank constraints.

## Limitations

The current evidence does not yet establish:

- duplicate-candidate filtering effectiveness;
- semantic-conflict filtering effectiveness;
- Candidate-ID collision handling;
- actual validation-cost reduction under Enforce mode;
- statistical reliability beyond the current small sample.

Therefore, the appropriate conclusion is that Gate v1 shows preliminary effectiveness, rather than being fully proven.

## Artifacts

- [Combined result summary](result_summary.json)
- [Fresh-validation report](fresh_validation/report.json)
- [Final-holdout report](final_holdout/report.json)
- [Fresh-validation growth curve](fresh_validation/growth_curve.csv)
- [Final-holdout growth curve](final_holdout/growth_curve.csv)
