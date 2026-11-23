"""Validation-only, nonnegative fusion of the three uncertainty components."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

from .uncertainty import COMPONENTS


@dataclass(frozen=True)
class Fusion:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        for name in ("mean", "std", "weights"):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if v.shape != (len(COMPONENTS),) or not np.isfinite(v).all():
                raise ValueError(f"Invalid fusion {name}.")
            object.__setattr__(self, name, v)
        if (self.std <= 0).any() or (self.weights < 0).any() or not np.isclose(self.weights.sum(), 1):
            raise ValueError("Fusion requires positive stds and simplex weights.")

    def score(self, components):
        values = np.asarray(components, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != len(COMPONENTS) or not np.isfinite(values).all():
            raise ValueError("Expected finite u_prob, u_pair and u_topo values in that order.")
        return ((values - self.mean) / self.std) @ self.weights

    def to_dict(self):
        return {"components": list(COMPONENTS), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, value):
        if tuple(value["components"]) != COMPONENTS:
            raise ValueError("Unexpected component ordering in fusion JSON.")
        return cls(value["mean"], value["std"], value["weights"])


def concordance(uncertainty, ndcg) -> float:
    """Exact C-index with uncertainty and NDCG ties excluded, as in StageB."""
    risk = np.asarray(uncertainty)
    target = np.asarray(ndcg)
    by_risk = np.argsort(risk, kind="stable")
    risk = risk[by_risk]
    metrics, ranks = np.unique(target[by_risk], return_inverse=True)
    tree = [0] * (len(metrics) + 1)

    def prefix(i):
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    concordant = comparable = 0
    cuts = np.r_[0, np.flatnonzero(risk[1:] != risk[:-1]) + 1, risk.size]
    for a, b in zip(cuts[:-1], cuts[1:]):
        for ri in ranks[a:b]:
            r = int(ri)
            left, right = prefix(r), prefix(r + 1)
            concordant += a - right
            comparable += a - (right - left)
        for ri in ranks[a:b]:
            j = int(ri) + 1
            while j < len(tree):
                tree[j] += 1
                j += j & -j
    return float(concordant / comparable) if comparable else float("nan")


def fit_fusion(validation_components, validation_ndcg, grid_step=0.05) -> tuple[Fusion, float]:
    """Pooled validation-only z-score and StageB's avgAll simplex search.

    Run independently for each frozen backbone using its combined validation
    cases across the desired domains. NDCG columns are @5 and @20, in that order.
    """
    x = np.asarray(validation_components, dtype=np.float64)
    y = np.asarray(validation_ndcg, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2 or y.shape != (len(x), 2):
        raise ValueError("Expected validation component [n,3] and NDCG [n,2] matrices.")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()):
        raise ValueError("Invalid validation values or NDCG outside [0,1].")
    if not np.isfinite(grid_step) or not 0 < grid_step <= 1:
        raise ValueError("grid_step must be in (0,1].")
    m = round(1.0 / grid_step)
    if not np.isclose(m * grid_step, 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("grid_step must divide 1.0.")
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std
    best = None
    best_value = -np.inf
    for i in range(m + 1):
        for j in range(m - i + 1):
            w = np.array([i, j, m - i - j], dtype=np.float64) / m
            risk = z @ w
            c = [concordance(risk, col) for col in y.T]
            tau = [kendalltau(-risk, col, variant="b").statistic for col in y.T]
            value = (c[0] + c[1] + (tau[0] + 1) / 2 + (tau[1] + 1) / 2) / 4
            if np.isfinite(value) and value > best_value:
                best, best_value = w, float(value)
    if best is None:
        raise ValueError("No defined validation objective. Check NDCG and component variability.")
    return Fusion(mean, std, best), best_value
