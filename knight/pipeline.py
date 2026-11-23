"""Core pipeline: calibrated base ranking, controlled views, and reliability signals."""

import hashlib
from dataclasses import dataclass

import numpy as np

from .calibration import calibrate_logits, softmax
from .controlled_decoding import controlled_logits, normalize_probe_scores
from .fusion import Fusion
from .prompts import DatasetConfig, build_prompt
from .uncertainty import COMPONENTS, pairwise_entropy, probability_entropy, semantic_dispersion


@dataclass(frozen=True)
class Case:
    case_id: str | int
    history_titles: tuple[str, ...]
    candidate_ids: tuple[str | int, ...]
    candidate_titles: tuple[str, ...]
    base_orders: tuple[tuple, ...] | None = None
    key_orders: tuple[tuple, ...] | None = None

    @classmethod
    def from_dict(cls, row):
        def orders(key):
            return tuple(tuple(o) for o in row[key]) if key in row else None

        return cls(row["case_id"], tuple(row["history_titles"]), tuple(row["candidate_ids"]),
                   tuple(row["candidate_titles"]), orders("base_orders"), orders("key_orders"))

    def __post_init__(self):
        if (not isinstance(self.case_id, (str, int)) or
            not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in self.candidate_ids)):
            raise ValueError("Case and candidate IDs must be strings or integers.")
        if len(self.candidate_titles) != len(self.candidate_ids) or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Provide exactly one title per distinct candidate ID.")
        if not self.history_titles or not all(isinstance(t, str) and t.strip()
                                            for t in (*self.history_titles, *self.candidate_titles)):
            raise ValueError("Provide nonempty chronological history and candidate titles.")


def _user_seed(case_id) -> int:
    try:
        return abs(int(case_id))
    except (TypeError, ValueError):
        return int(hashlib.sha1(str(case_id).encode("utf-8")).hexdigest()[:12], 16)


def view_indices(case: Case, config: DatasetConfig, source: str) -> np.ndarray:
    if len(case.candidate_ids) != config.candidate_size:
        raise ValueError("Case candidate count does not match config.")
    if source == "base":
        count, orders, multiplier = config.base_views, case.base_orders, 1729
    elif source == "key":
        count, orders, multiplier = len(config.keys), case.key_orders, 8191
    else:
        raise ValueError("source must be base or key.")
    if orders is None:
        rng = np.random.default_rng(config.seed + multiplier * _user_seed(case.case_id))
        return np.stack([rng.permutation(config.candidate_size) for _ in range(count)])
    lookup = {item: i for i, item in enumerate(case.candidate_ids)}
    if len(orders) != count or any(
        len(order) != len(lookup) or set(order) != lookup.keys() for order in orders
    ):
        raise ValueError(f"{source}_orders must contain {count} complete candidate permutations.")
    return np.asarray([[lookup[item] for item in order] for order in orders])


def view_prompts(case: Case, config: DatasetConfig, source: str) -> list[str]:
    indices = view_indices(case, config, source)
    keys = (None,) * len(indices) if source == "base" else tuple(config.keys)
    return [build_prompt(config, case.history_titles, [case.candidate_titles[i] for i in order], key)
            for order, key in zip(indices, keys)]


@dataclass(frozen=True)
class KnightResult:
    final_ranking: tuple
    base_probabilities: np.ndarray  # aligned to the original candidate_ids
    base_rankings: tuple[tuple, ...]
    controlled_rankings: tuple[tuple, ...]
    components: tuple[float, float, float]
    uncertainty: float | None

    def to_dict(self) -> dict:
        row = {
            "final_ranking": list(self.final_ranking),
            "base_probabilities": self.base_probabilities.tolist(),
            "base_rankings": [list(rank) for rank in self.base_rankings],
            "controlled_rankings": [list(rank) for rank in self.controlled_rankings],
            **dict(zip(COMPONENTS, self.components)),
        }
        if self.uncertainty is not None:
            row["u_fusion"] = self.uncertainty
        return row


def score_case(case: Case, config: DatasetConfig, *, base_logits, key_logits, probe_scores,
               title_embeddings, label_bias, fusion: Fusion | None = None) -> KnightResult:
    """Score cached or freshly computed arrays without modifying the base ranking.

    Logit rows follow each view's candidate order; probe scores and embeddings
    follow case.candidate_ids. Raw input logits are position-debiased here.
    """
    bi, ki = view_indices(case, config, "base"), view_indices(case, config, "key")
    if np.shape(base_logits) != bi.shape or np.shape(key_logits) != ki.shape:
        raise ValueError(f"Expected raw base and key logits of shapes {bi.shape} and {ki.shape}.")
    emb = np.asarray(title_embeddings)
    if emb.ndim != 2 or len(emb) != config.candidate_size:
        raise ValueError("Title embeddings must be aligned to candidates.")
    base = softmax(calibrate_logits(base_logits, label_bias))
    aligned = np.empty_like(base)
    np.put_along_axis(aligned, bi, base, axis=1)
    p_mean = aligned.mean(axis=0)
    final = np.argsort(-p_mean)
    base_ranks = np.take_along_axis(bi, np.argsort(-base, axis=1), axis=1)

    q = normalize_probe_scores(probe_scores)
    key = calibrate_logits(key_logits, label_bias)
    cd = softmax(controlled_logits(key, q, ki, config.cd_strength))
    cd_ranks = np.take_along_axis(ki, np.argsort(-cd, axis=1), axis=1)
    all_ranks = np.concatenate([base_ranks, cd_ranks], axis=0)
    components = (
        probability_entropy(p_mean),
        pairwise_entropy(all_ranks),
        semantic_dispersion(all_ranks[:, 0], emb),
    )

    def items(rank):
        return tuple(case.candidate_ids[i] for i in rank)

    return KnightResult(items(final), p_mean, tuple(map(items, base_ranks)),
                        tuple(map(items, cd_ranks)), components,
                        float(fusion.score(components)) if fusion is not None else None)
