"""KNIGHT's three uncertainty components and fixed-ranking evaluation helpers."""

import numpy as np

COMPONENTS = ("u_prob", "u_pair", "u_topo")


def probability_entropy(mean_probabilities) -> float:
    p = np.asarray(mean_probabilities, dtype=np.float64)
    if p.ndim != 1 or p.size < 2 or (p < 0).any() or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("Expected a nonnegative probability vector with positive mass.")
    eps = 1e-12
    p = p / (p.sum() + eps)
    return float(-(p * np.log(p + eps)).sum() / (np.log(len(p)) + eps))


def pairwise_entropy(rankings) -> float:
    """Average binary entropy over all unordered candidate pairs."""
    ranks = np.asarray(rankings)
    if ranks.ndim != 2 or not ranks.size or ranks.shape[1] < 2:
        raise ValueError("Expected complete rankings of the same candidate set.")
    positions = np.argsort(ranks, axis=1)
    a, b = np.triu_indices(ranks.shape[1], 1)
    p = np.clip((positions[:, a] < positions[:, b]).mean(axis=0), 1e-12, 1 - 1e-12)
    return float(np.mean(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))))


def semantic_dispersion(top1_indices, embeddings) -> float:
    """Mean pairwise cosine distance across the Top-1 multiset, retaining repeats."""
    top1 = np.asarray(top1_indices, dtype=int)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError("Expected finite candidate title embeddings.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Candidate title embeddings must be nonzero.")
    if len(set(top1.tolist())) <= 1:
        return 0.0
    e = (arr / norms)[top1]
    a, b = np.triu_indices(len(e), 1)
    return float(np.clip(np.mean(1 - np.clip(e @ e.T, -1, 1)[a, b]), 0, 2))


def ndcg_at(rank, ground_truth, k: int) -> float:
    """Single-positive, top-k NDCG on the final base ranking."""
    for i, item in enumerate(rank[:k]):
        if item == ground_truth:
            return float(1.0 / np.log2(i + 2))
    return 0.0
