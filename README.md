# CSEC 520/620 — Project 2: K-means from scratch

Start with **Python 3.10+**, `make`, and a plain virtual environment:

```bash
make setup
make reproduce
make test
```

The untouched template runs a working **Euclidean sklearn baseline on Iris**.
It writes `results/metrics.json` and figures under `results/euclidean/`.
The instructor's original Iris copy is also bundled in `data/iris/`:

```bash
make reproduce CONFIG=configs/iris-local.yaml
```

Those original files differ from sklearn's Iris at two historical records; see
[data/iris/README.md](data/iris/README.md). Use the same copy when comparing results.

## What you implement

`KMeansReference` is a provided correctness oracle. `KMeansScratch` is your work:

1. `_assign`: nearest-centroid assignment.
2. `_update`: cluster means, including an explained empty-cluster strategy.
3. `_init_centroids`: random and kmeans++ initialization.
4. `fit`: Lloyd iterations, convergence and independent restarts.
5. `src/distances.py::mahalanobis_sqdist`: positive-diagonal weighted distance.

The five pieces remain stubs. Tests for them skip until implemented. Library
clustering calls belong only in the reference, never your scratch implementation.

After implementing them, edit `config.yaml`:

```yaml
kmeans:
  implementation: scratch
  mahalanobis_diag: null  # identity weights initially; replace with your chosen positive vector
```

`mahalanobis_diag` must match the retained numeric features, in the exact order
recorded as `feature_names` in metrics. All weights must be finite and positive.
No improvement is expected from identity weights.

## Reproduce the complete comparison

With `implementation: scratch`, **`make reproduce` runs both Euclidean and
Mahalanobis** on the same prepared sample, k, seed, initialization and restart
budget. It writes:

```
results/metrics.json                         # combined record: runs.euclidean / runs.mahalanobis
results/euclidean/metrics.json               # per-run metrics + config + feature order + weights
results/euclidean/{k_selection,clusters,confusion_matrix}.png
results/mahalanobis/metrics.json
results/mahalanobis/{k_selection,clusters,confusion_matrix}.png
```

Top-level Euclidean fields are retained for compatibility; use `runs` for the
report's two-column comparison. Original input configuration is not overwritten.
A failed run removes the previous aggregate rather than leaving a stale success.

For a single exploratory configuration, use
`python -m src.cluster --config config.yaml`. Its `kmeans.distance` selects one
metric; the complete Make target explicitly runs both in scratch mode.
Requesting Mahalanobis with the sklearn baseline raises an error.

## Dataset setup

Iris is the debugging warm-up. The deliverable uses UNSW-NB15:

Download the [course dataset archive from Google Drive](https://drive.google.com/file/d/1jDsXYALnEsLzYYGCEtxgkAYygS-vKIt7/view?usp=sharing)
(about 20.6 GB). Extract `UNSW_NB15_training-set.csv` from the archive and
place it at `data/UNSW_NB15_training-set.csv`. The primary experiment uses
this partitioned training CSV; it does not require loading the entire archive.

Then set `data.source: csv`, the CSV path, and `data.target: attack_cat`.
Keep `drop_columns: [id, label]` to exclude the row ID and duplicate binary target.
See [data/README.md](data/README.md) for extraction instructions, provenance, and an optional mirror fallback.
Data acquisition is a setup step; subsequent reproduction needs no network.
Do not commit this large dataset.

The balanced sampler uses labels to preserve class coverage; this is a permitted
sampling design, not permission to use labels as input features. The 4,000 cap
produces 3,730 rows on this training file (400 per class except 130 Worms). Report
actual counts and do not interpret the balanced sample as natural traffic prevalence.
String/numeric multiclass targets are encoded for evaluation. Missing targets,
non-numeric-only data, and data without usable finite/nonconstant features are
rejected. Categorical feature columns are excluded unless you add an encoding step.

## Choosing k and interpreting results

The sweep records its scoring geometry. `selection.silhouette_geometry` selects
`euclidean` (default common space) or `configured`. If the maximum occurs at the
upper boundary, extend the sweep or justify stopping. Hold k fixed between the
primary comparison runs; additional k experiments can be reported separately.

- `silhouette_euclidean`: both partitions scored in the same standardized Euclidean space.
- `silhouette_configured`: each partition scored in its configured geometry.
- `silhouette`: compatibility alias of `silhouette_euclidean`.
- V-measure, ARI and NMI: external cluster agreement measures.
- Accuracy, precision, recall and macro F1: majority-vote mapping fitted and scored
  on the same sample. These are descriptive agreement, not held-out detection accuracy.

Declare the primary comparison criterion before trying weights. Investigate a
positive diagonal, seek improvement and explain tradeoffs using feature scales,
variance or feature groups. Do not compare inertia across distances or claim that
all scores must improve. Independent validation is needed for generalization claims.

## Correctness checks and grading

```bash
make test
make reproduce
make grade
```

With `compare_to_reference: true`, each scratch run compares against sklearn in
the **same geometry**. For Mahalanobis the reference fits `X * sqrt(diag)`;
its inertia is comparable to the scratch weighted objective. Ratios/ARI near one
show agreement. Poor agreement warrants inspecting code, convergence and restart
budget, not an automatic conclusion that the algorithm is wrong.

The grader invokes **`make reproduce` in a fresh temporary output directory**,
then validates the generated metrics, both scratch runs and their figures. It does
not accept old results. An untouched baseline is intentionally marked incomplete
for scratch implementation and comparison requirements. Heuristic checks remain
subject to instructor review; investigate reference diagnostics with evidence.

`OUTPUT_DIR` is an override used by grading, for example:
`make reproduce OUTPUT_DIR=/tmp/p2-results`. Custom Make workflows must honor it.

## Deliverables and submission

Write the report in `report/`, fill `SUBMISSION.md`, and match every reported
comparison to `results/metrics.json`. Describe sampling, k evidence, the chosen
criterion and diagonal, empty clusters, convergence, reference agreement and limits.

```bash
git add -A && git commit -m "p2 final" && git push
git tag p2-final && git push origin p2-final
```

Submit your private repository URL, tag and report through the course dropbox.
Follow the syllabus AI policy: acknowledge substantive assistance and write the
assigned scratch algorithm yourself.
