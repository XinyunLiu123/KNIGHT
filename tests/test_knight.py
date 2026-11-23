import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from knight.calibration import fit_label_bias
from knight.controlled_decoding import controlled_logits, normalize_probe_scores
from knight.fusion import Fusion, concordance, fit_fusion
from knight.pipeline import Case, score_case, view_indices, view_prompts
from knight.prompts import label_token_ids, load_config
from knight.uncertainty import COMPONENTS, pairwise_entropy, probability_entropy, semantic_dispersion
from scripts.run_knight import main as run_knight
from scripts.train_probe import training_embeddings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ml1m.yaml"


def test_domain_keys_and_prompt():
    key_sets = {
        "ml1m": ["genre", "popularity", "novelty", "recency", "similarity"],
        "grocery": ["category", "brand", "price", "popularity", "similarity"],
        "steam": ["genre", "popularity", "novelty", "developer", "similarity"],
    }
    for dataset, keys in key_sets.items():
        cfg = load_config(ROOT / f"configs/{dataset}.yaml")
        assert list(cfg.keys) == keys
        assert (cfg.base_views, cfg.candidate_size, cfg.cd_strength) == (5, 20, 3.0)
        ids = tuple(range(20))
        case = Case(42, ("History",), ids, tuple(map(str, ids)))
        base = view_prompts(case, cfg, "base")
        key = view_prompts(case, cfg, "key")
        assert len(base) == len(key) == 5
        assert base[0].startswith(f"You are a {cfg.domain_singular} recommender system.")
        assert "Additional recommendation criterion:" not in base[0]
        assert f"Additional recommendation criterion: {cfg.keys[keys[0]]}" in key[0]
        assert "Candidate T:" in base[0] and base[0].endswith("Answer: ")
        np.testing.assert_array_equal(view_indices(case, cfg, "key"), view_indices(case, cfg, "key"))


def test_bias_and_item_aligned_control():
    bias = fit_label_bias([[1, 3, 8], [5, 7, 12]])
    np.testing.assert_allclose(bias, np.array([1, 3, 8]) - 4)
    q = normalize_probe_scores([1, 2, 3])
    np.testing.assert_allclose(q, [-1, 0, 1], atol=2e-6)
    key = [[0, 2, 1], [2, 0, 1]]
    orders = [[0, 2, 1], [2, 1, 0]]
    np.testing.assert_allclose(controlled_logits(key, q, orders, 3),
                               np.asarray(key) + 3 * q[orders])
    with pytest.raises(ValueError, match="permutations"):
        controlled_logits(key, q, [[0, 0, 0], [2, 1, 0]], 3)


def test_components_and_repeated_top1s():
    assert probability_entropy([0.5, 0.5]) == pytest.approx(1)
    assert pairwise_entropy([[0, 1], [1, 0]]) == pytest.approx(1)
    assert pairwise_entropy([[0, 1], [0, 1]]) < 1e-8
    assert semantic_dispersion([0, 0, 1], [[1, 0], [0, 2]]) == pytest.approx(2 / 3)
    assert semantic_dispersion([1, 1, 1], [[1, 0], [0, 2]]) == 0.0


def test_cd_never_changes_the_recommendation():
    cfg = replace(load_config(CONFIG), candidate_size=3, base_views=2, keys={"genre": "Test"})
    case = Case("case", ("history",), (10, 20, 30), ("a", "b", "c"),
                base_orders=((10, 20, 30), (20, 30, 10)), key_orders=((30, 20, 10),))
    inputs = dict(base_logits=[[3, 1, 0], [1, 0, 3]], key_logits=[[0, 1, 3]],
                  label_bias=[0, 0, 0], probe_scores=[0.0, 0.1, 3.0], title_embeddings=np.eye(3),
                  fusion=Fusion([0, 0, 0], [1, 1, 1], [0.15, 0.25, 0.6]))
    a = score_case(case, replace(cfg, cd_strength=0), **inputs)
    b = score_case(case, cfg, **inputs)
    assert a.final_ranking == b.final_ranking == (10, 20, 30)
    np.testing.assert_array_equal(a.base_probabilities, b.base_probabilities)
    assert a.components[0] == b.components[0]
    assert a.controlled_rankings != b.controlled_rankings
    assert a.uncertainty < b.uncertainty


def test_concordance_excludes_ties():
    risk, target = [0.1, 0.1, 0.2, 0.3, 0.4], [0.9, 0.5, 0.2, 0.8, 0.2]
    pairs = [(i, j) for i, j in combinations(range(5), 2)
             if risk[i] != risk[j] and target[i] != target[j]]
    assert concordance(risk, target) == pytest.approx(
        sum((risk[i] - risk[j]) * (target[i] - target[j]) < 0 for i, j in pairs) / len(pairs)
    )


def test_scoring_entry_point_with_frozen_validation_fits(tmp_path):
    rng = np.random.default_rng(19)

    def case(i):
        return dict(case_id=i, gt_item=i % 20, history_titles=["history"],
                    candidate_ids=list(range(20)), candidate_titles=list(map(str, range(20))),
                    base_logits=rng.normal(size=(5, 20)).tolist(),
                    key_logits=rng.normal(size=(5, 20)).tolist(),
                    title_embeddings=rng.normal(size=(20, 6)).tolist(),
                    probe_scores=rng.normal(size=20).tolist())

    val = [case(i) for i in range(12)]
    bias = fit_label_bias(np.concatenate([row["base_logits"] for row in val], axis=0))
    bias_file = tmp_path / "bias.json"
    bias_file.write_text(json.dumps(dict(split="validation", label_bias=bias.tolist())))

    def score(rows, tag, extra=()):
        src, out = (tmp_path / f"{tag}_{suffix}.jsonl" for suffix in ("in", "out"))
        src.write_text("".join(json.dumps(row) + "\n" for row in rows))
        run_knight(["--config", str(CONFIG), "--input", str(src), "--output", str(out),
                    "--label-bias", str(bias_file), *extra])
        return [json.loads(line) for line in out.read_text().splitlines()]

    calibrated = score(val, "val")
    fusion, _ = fit_fusion([[r[k] for k in COMPONENTS] for r in calibrated],
                           [[r["ndcg5"], r["ndcg20"]] for r in calibrated], grid_step=0.5)
    fusion_file = tmp_path / "fusion.json"
    fusion_file.write_text(json.dumps(dict(split="validation", **fusion.to_dict())))
    held_out = [case(i) for i in range(20, 24)]
    raw = score(held_out, "test_raw")
    frozen = score(held_out, "test_frozen", ["--fusion", str(fusion_file)])
    assert len(raw) == len(frozen) == len(held_out)
    assert all(a["final_ranking"] == b["final_ranking"] and np.isfinite(b["u_fusion"])
               for a, b in zip(raw, frozen))
    fusion_file.write_text(json.dumps(dict(split="test", **fusion.to_dict())))
    with pytest.raises(ValueError, match="validation"):
        score(held_out, "invalid", ["--fusion", str(fusion_file)])


def test_probe_training_checkpoint_and_split(tmp_path):
    import torch

    from knight.probe import CDProbe, listwise_loss, load_probe

    torch.manual_seed(42)
    model = CDProbe(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = listwise_loss(model, torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 5, 4), 0.1)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.parameters())
    opt.step()
    path = tmp_path / "probe.pth"
    torch.save(model.state_dict(), path)
    assert not load_probe(path, 4, "cpu").training
    path = tmp_path / "embeddings.npz"
    values = dict(states=np.ones((3, 4)), positives=np.ones((3, 4)), negatives=np.ones((3, 5, 4)))
    np.savez(path, split="train", **values)
    assert len(training_embeddings(path)) == 3
    np.savez(path, split="validation", **values)
    with pytest.raises(ValueError, match="train"):
        training_embeddings(path)


def test_candidate_labels_require_distinct_single_tokens():
    class Tokenizer:
        def encode(self, label, add_special_tokens):
            assert not add_special_tokens
            return [ord(label)] if label != "C" else [1, 2]

    assert label_token_ids(Tokenizer(), "AB") == (65, 66)
    with pytest.raises(ValueError, match="one token"):
        label_token_ids(Tokenizer(), "ABC")
