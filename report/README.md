# Report

Put your IEEE-format report here (`.tex` source and the built `.pdf`).

Required content for Project 2:

1. **Problem & dataset** — what you clustered and why.
2. **Implementation** — your K-means: initialization, assignment, update,
   convergence, restarts, empty-cluster handling.
3. **Choosing k** — elbow and/or silhouette evidence (`results/<distance>/k_selection.png`).
4. **Evaluation** — cluster→class confusion matrix and metrics
   (`results/<distance>/confusion_matrix.png`), plus internal measures. State explicitly
   that labels were used only for the permitted sampling design and evaluation, not clustering.
5. **Euclidean vs. Mahalanobis** — declare the primary criterion, specify C and feature order,
   hold sample/k/seed/restarts fixed, and explain improvements or tradeoffs in terms of scale/variance.
   ⚠️ Inertia is **not** comparable across different distance metrics — compare
   silhouette or label-based measures instead.
6. **Conclusion & limitations.**

Every comparison number must match `results/metrics.json` under `runs.euclidean` or
`runs.mahalanobis`. Label the silhouette geometry. Majority-vote scores are in-sample
cluster agreement, not held-out detection accuracy. A sound comparison need not improve every metric.
