"""Evaluating clusters.

Two families of measures (see L07):

* **Internal** — no labels: inertia, silhouette. These say whether the clusters
  are compact and well separated.
* **External** — labels used ONLY to score, never to cluster: homogeneity,
  completeness, V-measure, ARI, NMI, plus a cluster->class confusion matrix
  with accuracy / precision / recall / F1.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics as skm

NAVY, ACCENT, ORANGE, GREEN = "#1F3A5F", "#2E6DA4", "#D9821B", "#4CA36A"


def internal_metrics(X: np.ndarray, labels: np.ndarray, inertia: float, configured_X=None) -> dict:
    out = {"inertia": float(inertia)}
    valid = 1 < len(np.unique(labels)) < len(X)
    common = float(skm.silhouette_score(X, labels)) if valid else None
    configured = (common if configured_X is None else
                  float(skm.silhouette_score(configured_X, labels)) if valid else None)
    out.update(silhouette=common, silhouette_euclidean=common,
               silhouette_configured=configured)
    if valid:
        out["calinski_harabasz"] = float(skm.calinski_harabasz_score(X, labels))
        out["davies_bouldin"] = float(skm.davies_bouldin_score(X, labels))

    return out


def external_metrics(y_true: np.ndarray, labels: np.ndarray) -> dict:
    h, c, v = skm.homogeneity_completeness_v_measure(y_true, labels)
    return {
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "adjusted_rand": float(skm.adjusted_rand_score(y_true, labels)),
        "nmi": float(skm.normalized_mutual_info_score(y_true, labels)),
    }


def map_clusters_to_classes(y_true: np.ndarray, labels: np.ndarray):
    """Majority-vote map each cluster to a class; return (mapping, y_pred).

    This is how an unsupervised partition gets scored with supervised metrics:
    every point in cluster j is *predicted* to be j's most common true class.
    """
    mapping = {}
    for j in np.unique(labels):
        members = y_true[labels == j]
        mapping[int(j)] = int(np.bincount(members).argmax()) if len(members) else -1
    y_pred = np.array([mapping[int(j)] for j in labels], dtype="int64")
    return mapping, y_pred


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, _ = skm.precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    pw, rw, fw, _ = skm.precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "precision_weighted": float(pw), "recall_weighted": float(rw),
        "f1_weighted": float(fw),
    }


def k_selection_sweep(X, build_fn, k_min: int, k_max: int, score_X=None):
    """Fit for each k in [k_min, k_max]; return (ks, inertias, silhouettes)."""
    if not 2 <= k_min <= k_max < len(X):
        raise ValueError("k sweep must satisfy 2 <= k_min <= k_max < number of usable rows")
    score_X = X if score_X is None else score_X
    ks, inertias, sils = [], [], []
    for k in range(k_min, k_max + 1):
        est = build_fn(k).fit(X)
        ks.append(k)
        inertias.append(float(est.inertia_))
        sils.append(float(skm.silhouette_score(score_X, est.labels_))
                    if 1 < len(np.unique(est.labels_)) < len(X) else None)
    return ks, inertias, sils


# ---------------------------------------------------------------- figures --
def save_k_selection_plot(ks, inertias, sils, path, chosen_k=None, geometry="euclidean"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ks, inertias, "o-", color=ACCENT)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia (within-cluster SSE)")
    axes[0].set_title("Elbow method", color=NAVY)
    axes[1].plot(ks, sils, "o-", color=GREEN)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("mean silhouette")
    axes[1].set_title(f"Silhouette ({geometry} geometry)", color=NAVY)
    for ax in axes:
        ax.grid(alpha=.3)
        if chosen_k is not None:
            ax.axvline(chosen_k, color="red", ls="--", lw=1, alpha=.7)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_cluster_scatter(X, labels, centroids, path, title="Clusters (PCA projection)"):
    """Project to 2-D with PCA so any dimensionality can be visualized."""
    from sklearn.decomposition import PCA

    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=0).fit(X)
        P, Cc = pca.transform(X), pca.transform(centroids)
        xl, yl = "PC1", "PC2"
    elif X.shape[1] == 1:
        P = np.column_stack([X[:, 0], np.zeros(len(X))])
        Cc = np.column_stack([centroids[:, 0], np.zeros(len(centroids))])
        xl, yl = "feature 1", ""
    else:
        P, Cc, xl, yl = X, centroids, "feature 1", "feature 2"

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cmap = plt.get_cmap("tab10")
    for j in np.unique(labels):
        m = labels == j
        ax.scatter(P[m, 0], P[m, 1], s=18, alpha=.78, color=cmap(int(j) % 10),
                   label=f"cluster {j}", edgecolors="none")
    ax.scatter(Cc[:, 0], Cc[:, 1], s=220, marker="X", c="black",
               edgecolors="white", linewidths=1.5, zorder=5, label="centroids")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, color=NAVY)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_confusion_matrix(y_true, y_pred, path, class_names=None):
    labels = range(len(class_names)) if class_names is not None else None
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted (cluster -> majority class)")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix", color=NAVY)
    n = cm.shape[0]
    ticks = range(n)
    names = [str(c) for c in (class_names or ticks)][:n]
    ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=8)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return cm.tolist()


def write_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
