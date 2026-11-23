"""Fit and apply centered candidate-label position bias."""

import numpy as np


def fit_label_bias(validation_logits) -> np.ndarray:
    """Use raw, validation-only base logits. No relevance labels are required."""
    z = np.asarray(validation_logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("Expected a finite matrix of validation base logits.")
    return (z - z.mean(axis=-1, keepdims=True)).mean(axis=0)


def calibrate_logits(logits, label_bias) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    b = np.asarray(label_bias, dtype=np.float32)
    if z.ndim < 1 or b.shape != (z.shape[-1],) or not (
        np.isfinite(z).all() and np.isfinite(b).all()
    ):
        raise ValueError("Logits and label bias must be finite and have the same label dimension.")
    return z - z.mean(axis=-1, keepdims=True) - b


def softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    out = np.exp(z - z.max(axis=-1, keepdims=True))
    return out / out.sum(axis=-1, keepdims=True)
