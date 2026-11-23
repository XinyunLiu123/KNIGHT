"""Train the CD probe on embeddings prepared from training interactions only."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from knight.probe import CDProbe, listwise_loss


def training_embeddings(path):
    with np.load(path, allow_pickle=False) as archive:
        if archive["split"].item() != "train":
            raise ValueError("Probe inputs must declare split='train'.")
        s, p, n = (torch.as_tensor(archive[key], dtype=torch.float32)
                   for key in ("states", "positives", "negatives"))
    if s.ndim != 2 or not len(s) or p.shape != s.shape or n.shape != (len(s), 5, s.shape[1]):
        raise ValueError("Expected states and positives [N, D] and five negatives [N, 5, D].")
    return TensorDataset(s, p, n)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with split, states, positives, negatives")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    torch.manual_seed(42)
    data = training_embeddings(args.input)
    batches = DataLoader(data, batch_size=256, shuffle=True,
                         generator=torch.Generator().manual_seed(42))
    probe = CDProbe(data.tensors[0].shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4)
    for _ in range(10):
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = listwise_loss(probe, *(v.to(args.device) for v in batch), temperature=0.1)
            loss.backward()
            optimizer.step()

    with args.output.open("xb") as stream:
        torch.save(probe.state_dict(), stream)


if __name__ == "__main__":
    main()
