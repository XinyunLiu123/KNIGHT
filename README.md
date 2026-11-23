# KNIGHT

Core implementation of **KNIGHT: Key-attended Controlled Decoding with Topological
Enhancement for Uncertainty Quantification in LLM-based Recommendation**.

KNIGHT estimates the uncertainty of a frozen LLM recommendation ranking. It
combines calibrated probability entropy, pairwise ranking instability, and semantic
dispersion across views. Controlled decoding does not change the final ranking.

## Layout

```text
KNIGHT/
├── knight/   # prompts, calibration, probe, controlled decoding, uncertainty, fusion
├── configs/  # MovieLens 1M, Amazon Grocery, and Steam
├── scripts/  # reference entry points for probe training and scoring
└── tests/
```

## Installation

```bash
pip install -r requirements.txt
```

## Inputs

`scripts/run_knight.py` scores prepared JSONL cases using base and key-view logits,
candidate orders, probe scores, and title embeddings. Fit label bias per
backbone–dataset pair and fusion per backbone, using validation data only.

`scripts/train_probe.py` trains the probe from a training-only NPZ of state,
positive-item, and negative-item embeddings. Datasets, model weights, caches, and
benchmark runners are not included.

## Citation

If you find KNIGHT useful in your research, please consider citing our paper:
**KNIGHT: Key-attended Controlled Decoding with Topological Enhancement for
Uncertainty Quantification in LLM-based Recommendation**. See [CITATION.cff](CITATION.cff).

## License

[Apache-2.0](LICENSE).
