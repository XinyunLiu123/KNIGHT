"""The title-embedding CD probe and its listwise training loss."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
PROBE_HISTORY_LENGTH = 10


class CDProbe(nn.Module):
    """Compatible with the original 384-dim title probe state dict."""

    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, item):
        return self.mlp(torch.cat([state, item], dim=-1)).squeeze(-1)


def history_text(titles) -> str:
    return " ".join(str(t).strip() for t in titles[-PROBE_HISTORY_LENGTH:])


def listwise_loss(probe: CDProbe, states, positives, negatives, temperature: float):
    """First column is the positive; other columns are training-only negatives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    batch, n_neg, dim = negatives.shape
    pos = probe(states, positives).reshape(batch, 1)
    repeated = states.unsqueeze(1).expand(-1, n_neg, -1).reshape(batch * n_neg, dim)
    neg = probe(repeated, negatives.reshape(batch * n_neg, dim)).reshape(batch, n_neg)
    logits = torch.cat((pos, neg), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(batch, dtype=torch.long, device=logits.device))


def load_probe(path, embedding_dim, device) -> CDProbe:
    state = torch.load(path, map_location=device, weights_only=True)
    for key in ("probe", "state_dict", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    probe = CDProbe(embedding_dim).to(device)
    probe.load_state_dict(state, strict=True)
    return probe.eval()


@torch.inference_mode()
def score_titles(probe: CDProbe, encoder, history_titles, candidate_titles, batch_size=512):
    """Return probe scores and normalized title embeddings in candidate-ID order."""
    embeddings = encoder.encode([history_text(history_titles), *candidate_titles],
                                batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    x = torch.as_tensor(embeddings, device=next(probe.parameters()).device)
    scores = probe(x[0].expand_as(x[1:]), x[1:]).cpu().numpy()
    return scores, embeddings[1:]
