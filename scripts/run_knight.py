"""Score prepared KNIGHT cases; logits are supplied, not generated here."""

import argparse
import json
from pathlib import Path

from knight.fusion import Fusion
from knight.pipeline import Case, score_case
from knight.prompts import load_config
from knight.uncertainty import ndcg_at


def validation_artifact(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("split") != "validation":
        raise ValueError("Calibration and fusion must be fitted on validation data.")
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, type=Path, help="JSONL cases with cached arrays")
    parser.add_argument("--label-bias", required=True, help="Validation JSON with a label_bias vector")
    parser.add_argument("--fusion", help="Optional validation JSON with components, mean, std, weights")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    bias = validation_artifact(args.label_bias)["label_bias"]
    fusion = Fusion.from_dict(validation_artifact(args.fusion)) if args.fusion else None
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            case = Case.from_dict(row)
            scores = score_case(
                case, config, base_logits=row["base_logits"], key_logits=row["key_logits"],
                probe_scores=row["probe_scores"], title_embeddings=row["title_embeddings"],
                label_bias=bias, fusion=fusion,
            )
            result = {"case_id": case.case_id, **scores.to_dict()}
            if "gt_item" in row:
                gt = row["gt_item"]
                if gt not in case.candidate_ids:
                    raise ValueError("gt_item must belong to the fixed candidate set.")
                result.update({f"ndcg{k}": ndcg_at(scores.final_ranking, gt, k) for k in (5, 20)})
            out.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
