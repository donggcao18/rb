from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_config
from .evaluate import evaluate
from .mine_hard_negatives import mine_from_checkpoint
from .prepare_data import prepare_dataset
from .train_ar2 import train as train_ar2
from .train_ranker import train as train_ranker
from .train_retriever import train as train_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete TheVault multi-positive AR2 workflow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_dataset(
            train_path=data["raw_train"],
            test_path=data.get("raw_test"),
            output_dir=data["prepared_dir"],
            dev_fraction=float(data.get("dev_fraction", 0.1)),
            seed=int(config.get("runtime", {}).get("seed", 13)),
            global_qrels=bool(data.get("global_qrels", False)),
            allow_missing_documents=bool(data.get("allow_missing_documents", False)),
        )
    if args.prepare_only:
        return

    retriever_warm = train_retriever(config)
    initial_negatives = output_dir / "hard-negatives-initial.jsonl"
    mine_from_checkpoint(config, retriever_warm, "train", initial_negatives)
    ranker_warm = train_ranker(config, initial_negatives)
    retriever_final, ranker_final = train_ar2(config, retriever_warm, ranker_warm, initial_negatives)

    if (Path(data["prepared_dir"]) / "queries.dev.jsonl").stat().st_size:
        evaluate(config, retriever_final, "dev", output_dir / "metrics-dev.json", ranker_final)
    if data.get("raw_test") and (Path(data["prepared_dir"]) / "queries.test.jsonl").stat().st_size:
        evaluate(config, retriever_final, "test", output_dir / "metrics-test.json", ranker_final)


if __name__ == "__main__":
    main()
