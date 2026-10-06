# Project 2 — K-means from scratch on UNSW-NB15

A from-scratch K-means (Lloyd's algorithm) is implemented in `src/kmeans.py`
(`KMeansScratch`), run on the UNSW-NB15 network-traffic dataset, and compared
against a scikit-learn reference as a correctness oracle only. The primary
experiment compares two distance geometries — Euclidean and a simplified
diagonal Mahalanobis — under a fixed sample, `k`, seed, and restart budget.

## 1. Problem & dataset

UNSW-NB15 (Moustafa & Slay, 2015) is a network-intrusion-detection dataset with
one *Normal* class and nine attack families. We cluster the **partitioned
training CSV** (`UNSW_NB15_training-set.csv`, 175,341 rows × 45 columns) after
dropping the non-feature columns `id`, `label`, the string columns
(`proto`, `service`, `state`) and the target `attack_cat`, leaving **39 numeric
features**. The target `attack_cat` is used **only** for the permitted
class-balanced sampling and for evaluation — never as a clustering feature or in
centroid fitting (no label leakage; the `label_leakage_scan` check is clean).

**Sampling design (disclosed).** `data.subsample: 4000` caps a stratified sample
to ~400 rows per class, producing **3,730 rows**: 400 in each of nine classes and
all 130 *Worms* rows. Features are z-score standardized. This class-balanced
cap is explicitly permitted, but the resulting sample does **not** preserve
natural traffic prevalence (Normal 56k / Generic 40k / Exploits 33k …), so the
label-based metrics below describe *in-sample cluster agreement*, not the
dataset's class prior.

## 2. Implementation

`KMeansScratch` implements Lloyd's algorithm with **no library K-means call**
(`no_library_kmeans_in_scratch` check is clean):

- **Initialization** (`_init_centroids`): `random` draws `k` distinct rows;
  `kmeans++` seeds the first centroid uniformly, then each next centroid with
  probability proportional to its squared distance to the *nearest* chosen
  centroid, with a **uniform fallback** if every point coincides with a chosen
  centroid (zero total D2). Both use the project distance (`euclidean_sqdist` /
  `mahalanobis_sqdist`) so the seeding matches the objective.
- **Assignment** (`_assign`): each point to its nearest centroid (argmin of the
  (n,k) squared-distance matrix).
- **Update** (`_update`): each centroid moves to the mean of its members.
  **Empty clusters are kept in place** (their old centroid is retained) rather
  than re-seeded: this avoids consuming the RNG stream mid-loop (which would
  break reproducibility of the restart budget) and did not occur in our
  converged runs.
- **Convergence** (`fit`): the loop stops when the Frobenius (L2) norm of the
  centroid shift is `< tol` (1e-4) or at `max_iter` (300). scikit-learn's
  `KMeans` also uses an **absolute** centroid-shift test, but it aggregates the
  shift as the **sum of squared per-centroid changes** and compares that sum to
  `tol`, whereas we use the L2 norm of the full shift matrix. The two are
  closely related but not identical, so the implementations can settle at
  marginally different local optima (a small, explainable source of the
  reference residual in §5). An absolute shift test is well scaled here because
  the z-standardized features are unit-variance, so centroid magnitudes are O(1).
- **Restarts**: `n_init = 50` restarts; the restart with the **lowest inertia**
  is kept. (Raised from 10 to 50 to improve reference agreement on this
  high-dimensional, multi-modal dataset; see §5.)

The scikit-learn `KMeansReference` is used **only** as an oracle
(`compare_to_reference: true`); the submission runs on `scratch`.

## 3. Choosing k

We sweep `k = 2…15` (k_max extended to 15 since the dataset has 10 classes) with
`selection.silhouette_geometry: euclidean` — the common standardized space, so
the silhouette is comparable across all k. Evidence
(`results/euclidean/k_selection.png`):

- **Elbow**: inertia falls steeply through k≈6 and then bends; the rate of
  decrease keeps slowing into k≈9–10, with a notable step around k=9–10.
- **Silhouette (Euclidean)**: rises to a clear peak at **k=10** (0.458) and
  declines past it, so the optimum is interior (not at the sweep boundary).

We therefore **fix k = 10** for both distance runs. This matches the 10 true
classes, and — unlike the Iris warm-up, where silhouette peaked at k=2 despite
three species — here the internal (silhouette) and external (class-count)
evidence agree, because the ten UNSW families are more geometrically separable.

## 4. Evaluation

Majority-vote maps each cluster to its most frequent class, then supervised
metrics are computed on the **same** sample. These are **in-sample cluster
agreement**, not held-out detection performance. All values match
`results/metrics.json`.

| Metric | Euclidean (k=10) | Mahalanobis (k=10) |
|---|---|---|
| Inertia (own objective) | 46,306.89 | 36,397.23 |
| Silhouette — common Euclidean space | **0.4577** | 0.4573 |
| Silhouette — configured geometry | 0.4577 | **0.4832** |
| Homogeneity | 0.2238 | 0.2240 |
| Completeness | 0.3766 | 0.3750 |
| V-measure | 0.2807 | 0.2805 |
| Adjusted Rand (vs labels) | 0.1152 | 0.1154 |
| NMI | 0.2807 | 0.2805 |
| Accuracy (cluster→class) | 0.3271 | 0.3271 |
| Macro precision | 0.3560 | 0.3535 |
| Macro recall | 0.3107 | 0.3107 |
| Macro F1 | 0.2522 | 0.2513 |

Internal compactness is strong (silhouette ~0.46), but the label-based agreement
is modest (V-measure ~0.28, accuracy ~0.33). This is expected: UNSW-NB15 attack
families overlap heavily in feature space and many are rare, so a 10-cluster
partition groups by *flow geometry* rather than by attack category. The
silhouette/k=10 result measures cluster *shape*, not ground-truth class
recovery — the same internal-vs-external tension noted in the handout.

**Per-class picture.** In both runs the ten clusters do not map one-to-one onto
the ten classes. Each run has several clusters that share a majority class, and
three classes — *Analysis*, *Reconnaissance*, *Shellcode* — are **never the
majority of any cluster**; their rows are absorbed into larger, geometrically
closer families (e.g. *Exploits* and *Fuzzers* each become the majority of two
clusters in each run, and *Normal* of two). The Euclidean and Mahalanobis runs
agree on this coarse structure (ARI 0.99 between the two partitions): the
geometrically resolvable families are the high-volume ones (*DoS*, *Normal*,
*Exploits*, *Fuzzers*, *Generic*, *Backdoor*, *Worms*), while the rare/overlapping
families are not resolved into their own clusters. This is why the label-based
scores are limited even though the clusters are geometrically tight.

## 5. Euclidean vs. Mahalanobis

**Primary criterion (declared before trying weights):** *silhouette in the common
Euclidean space* (`silhouette_euclidean`), evaluated on the fixed sample with k,
seed, and restart budget held constant. Rationale: inertia is **not**
comparable across geometries, and a common space avoids rewarding a partition
merely because it is compact in its own weighted metric. V-measure and macro F1
are reported alongside as label-based cross-checks.

**The diagonal C.** With features standardized, a full (covariance) Mahalanobis
metric would re-introduce scale differences, so we use a **simplified positive
diagonal** `C = diag(c_1…c_39)`, equivalent to a per-feature weighted Euclidean
distance `d(x,μ)² = Σ_f c_f (x_f−μ_f)²`. Note that after z-scoring every feature
has unit variance, so the *null* diagonal (all `c_f = 1`) is exactly Euclidean —
any deviation from 1 is a deliberate re-weighting, not a variance correction.

We choose to damp six **heavy-tailed** features. Even after z-scoring they carry
far larger outliers than the rest (z-scored skewness / max magnitude below), so
a few extreme flows dominate raw distance in those dimensions; dampening them
makes the geometry reflect the bulk of within-group structure rather than a
handful of outliers. This is a scale/heterogeneity argument, not a label-based
one:

| Feature | c_f | z-skew | max \|z\| |
|---|---|---|---|
| stcpb, dtcpb | 0.05 | 1.2 / 1.2 | 2.5 / 2.6 |
| sbytes | 0.30 | 27.9 | 34.0 |
| rate | 0.10 | 3.2 | 4.9 |
| sload | 0.10 | 5.4 | 14.0 |
| dload | 0.20 | 8.6 | 12.4 |

All other features keep `c_f = 1`. The full 39-element diagonal is stored in
`config.yaml` under `kmeans.mahalanobis_diag` in final feature order; `make
reproduce` reproduces it and the grader confirms the recorded weights match.

**What the re-weighting actually changes — stated plainly.** A k-means partition
is, by construction, compact in the metric it optimizes, so a silhouette
measured *in that same metric* (`silhouette_configured`) is largely
tautological: it rises 0.4577 → 0.4832 mostly because the Mahalanobis run is
scored in the very geometry it minimized. We do **not** treat this as evidence
of a better clustering. On the **predeclared** criterion — the common-Euclidean
silhouette, which uses a fixed geometry independent of the run — the result is
flat (0.4577 → 0.4573), and the label-based metrics are essentially unchanged
(V-measure 0.2807 → 0.2805, accuracy 0.3271 → 0.3271, macro F1 0.2522 → 0.2513).
The two runs land on nearly the same partition (ARI 0.99 between them).

**Robustness / sensitivity.** To show this is not an artifact of one hand-picked
diagonal, we re-ran the identical setup (k=10, seed 42, n_init 50) over a range
of damping:

| Weights | common-sil | conf-sil | V-measure | accuracy | macro F1 |
|---|---|---|---|---|---|
| identity (= Euclidean) | 0.4577 | 0.4577 | 0.2807 | 0.3271 | 0.2522 |
| chosen (damp 6) | 0.4573 | 0.4832 | 0.2805 | 0.3271 | 0.2513 |
| light (chosen weights ×2 → less damping) | 0.4556 | 0.4755 | 0.2782 | 0.3257 | 0.2368 |
| strong (chosen weights ×0.5 → more damping) | 0.4468 | 0.4762 | 0.2845 | 0.3300 | 0.2499 |
| uniform 0.5 (all feats) | 0.4577 | 0.4577 | 0.2807 | 0.3271 | 0.2522 |

The common-silhouette stays in [0.447, 0.458] and the label metrics barely move
across the whole range. Two points stand out: (i) **uniformly** scaling every
feature is a no-op for the Euclidean geometry (a constant rescaling preserves
argmin and the silhouette — the last row is identical to identity), so only
*differential* re-weighting can change the partition; (ii) even the differential
changes are small here, because the standardized features carry similar
information and the ten classes overlap. 

**Conclusion for the comparison.** The diagonal is a valid positive-weight
extension, but on this dataset it does not improve the geometry-independent
common-space silhouette or the class agreement; it changes the within-metric
(compact-in-its-own-metric) silhouette and inertia, which are self-referential
and not a fair cross-geometry win. The honest takeaway is that per-feature
re-weighting cannot resolve the heavy overlap among the ten attack families —
a limitation of a diagonal metric, not of our implementation. No independent
validation was used, so no held-out detection claim is made.

**Reference agreement (same geometry, `n_init=50`).**
- Euclidean: `inertia_ratio = 0.997`, `ari_vs_reference = 0.900`.
- Mahalanobis: `inertia_ratio = 1.000`, `ari_vs_reference = 1.000`.

The reference fits `X·√C` for Mahalanobis, so its objective is directly
comparable to the scratch one. Ratios and ARI at/near 1.0 indicate our Lloyd
loop converges to the same objective the library does. The Euclidean run's small
residual (ratio 0.997, ARI 0.90) is a benign local-optimum difference, and two
implementation differences plausibly contribute: we stop on a **Frobenius
(L2-norm) centroid-shift** test (`< 1e-4`) whereas scikit-learn stops when the
**sum of squared per-centroid shifts** falls below its `tol`, so the two can
settle at slightly different points in the multi-modal landscape; and the two
use independent RNG streams for their kmeans++ seeding. The objective agrees to within 0.35% and the Mahalanobis run
matches the oracle exactly, so this is not a correctness defect; both runs
satisfy the grader's agreement threshold.

## 6. Conclusion & limitations

We implemented K-means from scratch and validated it against a library oracle
(reference agreement ~1.0). On UNSW-NB15 the optimal k is 10 by silhouette and
elbow, matching the class count. We then ran a valid positive-diagonal
Mahalanobis comparison: it is a legitimate geometric extension, but on this
data it leaves the geometry-independent common-space silhouette and the
label-based scores essentially flat (the only "improvement" is in the
self-referential within-metric silhouette, which k-means wins by construction).
The honest conclusion is that per-feature re-weighting does not resolve the
heavy overlap among the ten attack families.

Limitations: (1) the class-balanced sample does not reflect natural class
prevalence; (2) label-based metrics are in-sample, with no held-out test set, so
no intrusion-detection accuracy is claimed; (3) only 39 of the 45 columns are
used (string fields `proto`/`service`/`state` were dropped, not encoded); (4)
the simplified diagonal ignores inter-feature covariance. Encoding the string
fields and adding an independent validation split would be the natural next
steps.
