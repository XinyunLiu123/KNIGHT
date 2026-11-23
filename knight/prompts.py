"""Paper domain prompts, ordered recommendation keys and label token checks."""

from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    domain_singular: str
    domain_plural: str
    action: str
    history_length: int
    candidate_size: int
    base_views: int
    cd_strength: float
    seed: int
    keys: dict[str, str]

    def __post_init__(self):
        if not 2 <= self.candidate_size <= 26 or self.base_views < 1 or self.history_length < 1:
            raise ValueError("Need 2–26 candidates, at least one base view and a positive history length.")
        if self.action not in {"watch", "buy", "play"}:
            raise ValueError("Unknown recommendation action.")
        if not isinstance(self.keys, dict) or not self.keys or not all(
            isinstance(k, str) and k and isinstance(v, str) and v for k, v in self.keys.items()
        ):
            raise ValueError("keys must be a nonempty ordered mapping of names to prompt instructions.")
        if not np.isfinite(self.cd_strength) or self.cd_strength < 0 or self.seed < 0:
            raise ValueError("Use finite nonnegative CD strength and a nonnegative seed.")


def load_config(path: str | Path) -> DatasetConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return DatasetConfig(**yaml.safe_load(stream))


def candidate_labels(n: int) -> tuple[str, ...]:
    if not 2 <= n <= 26:
        raise ValueError("Candidate count must be between 2 and 26.")
    return tuple(ascii_uppercase[:n])


def label_token_ids(tokenizer, labels) -> tuple[int, ...]:
    """Match the paper's bare candidate-letter scoring convention."""
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("This tokenizer does not encode every bare candidate label as one token.")
    ids = tuple(tokens[0] for tokens in token_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate label token IDs must be distinct.")
    return ids


def build_prompt(config: DatasetConfig, history_titles, candidate_titles, key: str | None = None) -> str:
    past = {"watch": "watched", "buy": "bought", "play": "played"}[config.action]
    candidate_titles = tuple(candidate_titles)
    if len(candidate_titles) != config.candidate_size:
        raise ValueError("Prompt candidates do not match config.candidate_size.")
    labels = candidate_labels(len(candidate_titles))
    history = "\n".join(f'- "{title}"' for title in tuple(history_titles)[-config.history_length:])
    candidates = "\n".join(f'Candidate {lab}: "{title}"' for lab, title in zip(labels, candidate_titles))
    condition = f"\nAdditional recommendation criterion: {config.keys[key]}\n" if key else ""
    return (
        f"You are a {config.domain_singular} recommender system.\n"
        f"I have {past} the following {config.domain_plural} in the past in order:\n{history}\n\n"
        f"The {len(candidate_titles)} candidate {config.domain_plural} are as follows:\n{candidates}\n{condition}\n"
        f"Which {config.domain_singular} would I like to {config.action} next most?\n"
        f"Only output one candidate index from this set: {', '.join(labels)}.\n"
        "Answer: "
    )
