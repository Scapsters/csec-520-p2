# Submission Manifest — Project 2

Fill this in before submitting. The grading agent reads it first.

## Student
- Name (RIT ID):

## Course level
- [ ] 520 (undergraduate)
- [ ] 620 (graduate)

## Dataset
- Name / source / version / URL:
- How to obtain it (command or steps):
- Label column (used for EVALUATION ONLY):
- Subsample size and how you chose it:

## How to reproduce
```bash
make setup
# Download the course Drive archive and extract data/UNSW_NB15_training-set.csv
# See data/README.md; make data is an optional mirror fallback / checksum check.
make reproduce
```
Anything non-default the grader must know (data download, runtime):

## Your implementation
- Confirm `config.yaml` has `kmeans.implementation: scratch`: [ ]
- Initialization used (`random` / `kmeans++`):
- How you handle **empty clusters**:
- Convergence criterion and tolerance:
- Agreement with the reference (each run’s `reference_check` in metrics.json; same geometry):
  - Euclidean `inertia_ratio` / `ari_vs_reference`:
  - Mahalanobis `inertia_ratio` / `ari_vs_reference`:

## Choosing k
- k you report, and the evidence (elbow / silhouette):
- If the best silhouette k differs from the number of true classes, explain:

## Claimed results (must match `results/metrics.json`)
| Metric | Euclidean | Mahalanobis |
|---|---|---|
| Silhouette (common Euclidean) | | |
| Silhouette (configured geometry) | | |
| V-measure | | |
| Accuracy | | |
| Macro F1 | | |

- Primary comparison criterion declared before weight experiments:
- The diagonal C you chose, its feature order, and why:
- Same sample, k, seed and restart budget across the primary runs: [ ]
- Tradeoffs and whether any independent validation was used:

## AI-use acknowledgment
Per the syllabus policy, briefly note any substantive use of AI assistants.
