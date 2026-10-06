# Submission Manifest — Project 2

Fill this in before submitting. The grading agent reads it first.

## Student
- Name (RIT ID): Scott Happy (sdh8796)

## Course level
- [x] 520 (undergraduate)
- [ ] 620 (graduate)

## Dataset
- Name / source / version / URL: UNSW-NB15 partitioned **training** CSV
  (`UNSW_NB15_training-set.csv`, 175,341 rows × 45 cols). Source:
  Moustafa & Slay (2015). Primary copy from the course Google Drive archive
  (`https://research.unsw.edu.au/projects/unsw-nb15-dataset`).
  SHA-256 of the verified file: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.
- How to obtain it (command or steps): download the course Drive archive and
  extract `UNSW_NB15_training-set.csv` to `data/`, **or** run `make data` (mirror
  fallback, checksum-verified). See `data/README.md`.
- Label column (used for EVALUATION ONLY): `attack_cat` (10 classes). `id` and
  `label` are dropped; string fields `proto`/`service`/`state` are dropped.
- Subsample size and how you chose it: `data.subsample: 4000` cap → **3,730
  rows**, class-balanced (400/class, 130 *Worms*). Features z-scored; 39 numeric
  features remain.

## How to reproduce
```bash
make setup
# Download the course Drive archive and extract data/UNSW_NB15_training-set.csv
# (or: make data). See data/README.md.
make reproduce
```
Anything non-default the grader must know: the large UNSW CSV must be present at
`data/UNSW_NB15_training-set.csv` (git-ignored); it is NOT committed. The
runtime for the full 2-run reproduce is a few minutes on a laptop. `kmeans.n_init`
is set to 50 (vs the template default 10) to obtain reference agreement on this
multi-modal data.

## Your implementation
- Confirm `config.yaml` has `kmeans.implementation: scratch`: [x]
- Initialization used (`random` / `kmeans++`): `kmeans++`
- How you handle **empty clusters**: keep the old centroid in place (do not
  re-seed or drop it); did not occur in the converged runs but is safe if it does.
- Convergence criterion and tolerance: Frobenius norm of the centroid shift
  `< tol` (1e-4), or `max_iter` (300); `n_init=50` restarts keeping the lowest
  inertia.
- Agreement with the reference (each run's `reference_check`; same geometry):
  - Euclidean `inertia_ratio` / `ari_vs_reference`: **0.997 / 0.900**
  - Mahalanobis `inertia_ratio` / `ari_vs_reference`: **1.000 / 1.000**

## Choosing k
- k you report, and the evidence (elbow / silhouette): **k = 10**. Silhouette
  (common Euclidean space) peaks at k=10 (0.458), interior to the 2–15 sweep;
  the elbow bends around k=9–10. Matches the 10 true classes.
- If the best silhouette k differs from the number of true classes, explain:
  here they agree (k=10 = 10 classes). The Iris warm-up instead peaked at k=2
  despite 3 species — internal metrics measure geometry, not ground truth.

## Claimed results (must match `results/metrics.json`)
| Metric | Euclidean | Mahalanobis |
|---|---|---|
| Silhouette (common Euclidean) | 0.4577 | 0.4573 |
| Silhouette (configured geometry) | 0.4577 | 0.4832 |
| V-measure | 0.2807 | 0.2805 |
| Accuracy | 0.3271 | 0.3271 |
| Macro F1 | 0.2522 | 0.2513 |

- Primary comparison criterion declared before weight experiments: **silhouette
  in the common (standardized) Euclidean space**, with V-measure and macro F1 as
  label-based cross-checks; inertia is not compared across geometries.
- The diagonal C you chose, its feature order, and why: a 39-element positive
  diagonal stored in `kmeans.mahalanobis_diag` (final feature order). All weights
  are 1 except six heavy-tailed features — `stcpb` & `dtcpb` = 0.05, `rate` =
  0.10, `sload` = 0.10, `dload` = 0.20, `sbytes` = 0.30 — damped because, even
  after z-scoring, they carry the largest outliers (z-skew 1.2–27.9) and a few
  extreme flows otherwise dominate distance; a scale/heterogeneity choice, not a
  label-based one.
- Same sample, k, seed and restart budget across the primary runs: [x]
- Tradeoffs and whether any independent validation was used: on the predeclared
  common-Euclidean criterion the diagonal leaves the silhouette flat
  (0.4577 → 0.4573) and the label-based scores unchanged; the only "improvement"
  is in the self-referential configured-geometry silhouette (k-means wins its own
  metric by construction), which we do not count as a cross-geometry win. No
  independent validation was performed, so no held-out detection accuracy is
  claimed.

## AI-use acknowledgment
Per the syllabus policy, an AI assistant was used to help debug and guide myself during implementation of Lloyd's algorithm, and was used in the following sections of the report:
- Implementation
These sections did not provide a substantial ability for me to express desired learning objectives. I believe my implementation is self-documenting (and also is just plainly documented).
