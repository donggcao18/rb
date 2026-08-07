# Multi-positive AR2 for TheVault Ruby

This package adapts AR2 to natural-language-to-Ruby-code retrieval. It uses only
the original `Query:` and `Code:` rows. Pseudo-query augmentation files are not
read by any stage.

## Data contract

- `text_id` is the code-document label and is always preserved as a string.
- `numeric_id` identifies the source query row.
- Rows beginning with `Code:` form the searchable corpus.
- Rows beginning with `Query:` form query groups.
- Identical normalized queries are deduplicated and their `text_id` values are
  combined into `positive_doc_ids`.

The converter accepts a JSON list or JSONL regardless of the filename suffix.
By default, train and test relevance labels are grouped independently to avoid
test-label leakage. `--global-qrels` exists only to reproduce the behavior of
the older `build_multilable.py` utility.

## Environment

Use Python 3.10 or newer on the GPU machine. Install a CUDA-compatible PyTorch
build, then install `requirements-code-search.txt`. The A100 configuration uses
BF16. FAISS CPU is sufficient for a corpus of roughly 40,000 Ruby methods; the
transformer encoding still runs on the GPU.

## Prepare and audit data

Run commands from `AR2/AR2`:

```bash
python -m code_search.prepare_data \
  --train ../../theVault-multilabel/Ruby_train_r32.0.json \
  --test /path/to/Ruby_test_r32.0.json \
  --output-dir outputs/vault-ruby/data
```

Review `outputs/vault-ruby/data/audit.json` before training. A missing positive
document is fatal. `--allow-missing-documents` is intended only for the partial
sample committed to this repository.

Validate already prepared artifacts independently with:

```bash
python -m code_search.validate_data --prepared-dir outputs/vault-ruby/data
```

## Full workflow

Edit `configs/vault_ruby_a100.yaml`, especially `data.raw_test`, then run:

```bash
python -m code_search.run_pipeline --config configs/vault_ruby_a100.yaml
```

The workflow performs:

1. Canonical conversion and group-aware train/dev split.
2. Multi-positive retriever warm-up.
3. Top-100 retrieval and positive-safe hard-negative mining.
4. Multi-positive cross-encoder warm-up.
5. Alternating ranker/retriever AR2 rounds with index refresh.
6. Retriever and reranked dev/test evaluation.

Individual entry points are also available:

```bash
python -m code_search.train_retriever --config configs/vault_ruby_a100.yaml
python -m code_search.mine_hard_negatives --config configs/vault_ruby_a100.yaml \
  --checkpoint outputs/vault-ruby/checkpoints/retriever-warm.pt \
  --split train --output outputs/vault-ruby/checkpoints/hard-negatives-initial.jsonl
python -m code_search.train_ranker --config configs/vault_ruby_a100.yaml \
  --hard-negatives outputs/vault-ruby/checkpoints/hard-negatives-initial.jsonl
python -m code_search.train_ar2 --config configs/vault_ruby_a100.yaml \
  --retriever-checkpoint outputs/vault-ruby/checkpoints/retriever-warm.pt \
  --ranker-checkpoint outputs/vault-ruby/checkpoints/ranker-warm.pt \
  --hard-negatives outputs/vault-ruby/checkpoints/hard-negatives-initial.jsonl
```

To resume after round `N`, pass the round-`N` retriever and ranker checkpoints,
the matching hard-negative file, and `--start-round N+1`. Optimizer state is
restored from the supplied checkpoints.

`train_ar2` currently targets one process/one A100 and uses gradient
accumulation to reproduce the original effective batch. Retriever warm-up can
be launched with `torchrun` for multiple GPUs.

## Metrics

Evaluation reports Hit, multi-label Recall, MRR, MAP, and nDCG at configured
cutoffs. Relevance is based only on document IDs; answer-string matching from
the Wikipedia implementation is not used.
