"""Apply the candidate-aligned probe direction to diagnostic key views."""

import numpy as np


def normalize_probe_scores(scores) -> np.ndarray:
    q = np.asarray(scores, dtype=np.float32)
    if q.ndim != 1 or q.size < 2 or not np.isfinite(q).all():
        raise ValueError("Expected one finite probe score per candidate (at least two candidates).")
    # StageB uses torch.std's sample standard deviation, not population std.
    return (q - q.mean()) / (q.std(ddof=1) + 1e-6)


def controlled_logits(key_logits, normalized_scores, candidate_indices, strength):
    key_logits = np.asarray(key_logits, dtype=np.float32)
    normalized_scores = np.asarray(normalized_scores, dtype=np.float32)
    candidate_indices = np.asarray(candidate_indices)
    if (key_logits.shape != candidate_indices.shape or key_logits.ndim != 2
        or not np.issubdtype(candidate_indices.dtype, np.integer)
        or normalized_scores.shape != (key_logits.shape[1],)
        or not np.array_equal(np.sort(candidate_indices, axis=1),
                               np.broadcast_to(np.arange(key_logits.shape[1]), key_logits.shape))):
        raise ValueError("Key-view orders must be permutations of the shared candidate set.")
    if not (np.isfinite(strength) and strength >= 0):
        raise ValueError("CD strength must be finite and nonnegative.")
    return key_logits + float(strength) * normalized_scores[candidate_indices]
