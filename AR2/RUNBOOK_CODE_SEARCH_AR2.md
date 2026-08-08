# Runbook: Multi-positive AR2 for TheVault Ruby

This runbook describes how to prepare, train, resume, and evaluate the
multi-positive AR2 code-search pipeline on one NVIDIA A100 80 GB GPU.

The pipeline uses only original `Query:` and `Code:` records. It does not read
`Ruby_q10.jsonl` or `Ruby_ready_to_feed_numeric.jsonl`.

## Correct execution order

Use this order on the Linux server:

1. Enter `AR2`.
2. Create and activate the Conda environment.
3. Verify that the Conda PyTorch installation detects the A100.
4. Configure the full train and test paths.
5. Run the regression tests.
6. Prepare and validate the dataset.
7. Start the training pipeline.

Do not run the pipeline command before completing installation, configuration,
and validation.

## 1. Repository layout and entry point

The active implementation is directly under `retrieval-baseline/AR2`:

```text
retrieval-baseline/
`-- AR2/                         # upstream repository root
    |-- code_search/             # TheVault adaptation implemented here
    |-- configs/
    |-- tests/
    |-- environment-code-search.yml
    |-- requirements-code-search.txt
    `-- RUNBOOK_CODE_SEARCH_AR2.md
```

All commands in this runbook run from the `AR2/` directory:

```bash
cd /path/to/retrieval-baseline/AR2
```

The eventual pipeline entry point is:

```bash
python -m code_search.run_pipeline --config configs/vault_ruby_a100.yaml
```

Do not execute it yet. Complete Sections 2 through 6 first. Do not use the
legacy `requirement.txt` or the scripts under `wiki/` for this dataset.

## 2. Create the Conda environment and install dependencies

Check that Conda is available:

```bash
conda --version
```

Create and activate the complete environment:

```bash
conda env create -f environment-code-search.yml
conda activate ar2
```

This creates the complete environment, including PyTorch 2.5.1 with its CUDA
12.1 runtime. PyTorch comes from the official `pytorch` and `nvidia` Conda
channels; the remaining Conda packages come from `defaults`.

`faiss-cpu` and Transformers are installed by pip inside the same Conda
environment. Keeping FAISS out of Conda's dependency solve avoids the
`pytorch-gpu`/`faiss-cpu` conflict from the previous environment file. CPU
FAISS is intentional: model encoding runs on the A100, while exact search over
the expected Ruby corpus is small enough to run on CPU.

Confirm the active environment:

```bash
which python
python --version
conda list
```

PyTorch 2.5.1 is intentionally pinned because it is the last release for which
PyTorch published official Conda packages. The AR2 adaptation does not require
PyTorch 2.6 or newer.

CUDA 12.1 is a broadly compatible choice for an A100. PyTorch ships the CUDA
runtime in the environment, so a separate system CUDA toolkit is not required.
The NVIDIA driver must support CUDA 12.1. If the server has an older driver,
change `pytorch-cuda=12.1` to `pytorch-cuda=11.8` in
`environment-code-search.yml` before creating the environment.

The first training run downloads `microsoft/codebert-base` from Hugging Face.
For an offline machine, cache the model in advance and set both
`retriever.model_name` and `ranker.model_name` to the local model directory.
The current co-training collator requires both models to use the same
tokenizer.

## 3. Verify the GPU environment

Check the driver and GPU:

```bash
nvidia-smi
```

Check PyTorch:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("BF16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

Required results:

```text
CUDA available: True
GPU: NVIDIA A100 ... 80GB
BF16 supported: True
```

Verify imports:

```bash
python - <<'PY'
import faiss
import numpy
import torch
import transformers
import yaml

print("Dependency imports succeeded")
PY
```

## 4. Required input data

Provide the complete files:

```text
Ruby_train_r32.0.json
Ruby_test_r32.0.json
```

The train file must contain:

- `Query:` rows used for training.
- `Code:` rows forming the searchable code corpus.
- A `text_id` on every query and code row.
- Corresponding code documents for every training-query `text_id`.

The test file may contain query rows only, but each query must reference a
`text_id` present in the code corpus built from the train file.

The files can be JSONL even though their suffix is `.json`.

## 5. Configure paths

Edit `configs/vault_ruby_a100.yaml`:

```bash
nano configs/vault_ruby_a100.yaml
```

Set the Linux paths:

```yaml
data:
  raw_train: /absolute/path/to/Ruby_train_r32.0.json
  raw_test: /absolute/path/to/Ruby_test_r32.0.json
  prepared_dir: outputs/vault-ruby/data
  dev_fraction: 0.1
  global_qrels: false
  allow_missing_documents: false

output:
  dir: outputs/vault-ruby/checkpoints
```

Keep `global_qrels: false` unless the benchmark protocol explicitly allows
test relevance labels to influence training. Setting it to `true` reproduces
the older `build_multilable.py` behavior but can leak test labels.

Never enable `allow_missing_documents` for a real training run. That option is
only for auditing the incomplete sample committed to this repository.

## 6. Run regression tests

```bash
python -m unittest discover -s tests -v
```

All tests should pass. On the A100 environment, the tensor-loss test should run
instead of being skipped.

## 7. Prepare the dataset

Run conversion without starting training:

```bash
python -m code_search.run_pipeline \
  --config configs/vault_ruby_a100.yaml \
  --prepare-only
```

This creates:

```text
outputs/vault-ruby/data/
├── audit.json
├── corpus.jsonl
├── queries.train.jsonl
├── queries.dev.jsonl
├── queries.test.jsonl
├── qrels.train.tsv
├── qrels.dev.tsv
└── qrels.test.tsv
```

Inspect the audit:

```bash
python -m json.tool outputs/vault-ruby/data/audit.json | less
```

Important fields:

- `corpus_documents` must be greater than zero.
- `query_groups.train` must be greater than zero.
- `missing_positive_references` must be zero.
- `multi_positive_groups` should reflect the known duplicated-query cases.
- Review `train_test_normalized_query_overlap` before training.
- Review query and code whitespace-token length percentiles.

Run the independent validator:

```bash
python -m code_search.validate_data \
  --prepared-dir outputs/vault-ruby/data \
  --output outputs/vault-ruby/data/validation.json
```

Do not continue unless it exits successfully and reports:

```text
'valid': True
```

## 8. Run the complete pipeline

After successful preparation and validation:

```bash
mkdir -p logs

python -m code_search.run_pipeline \
  --config configs/vault_ruby_a100.yaml \
  --skip-prepare \
  2>&1 | tee logs/vault-ruby-ar2.log
```

The pipeline executes:

1. Multi-positive retriever warm-up.
2. Full-corpus encoding and top-100 retrieval.
3. Positive-safe hard-negative mining.
4. Multi-positive cross-encoder ranker warm-up.
5. Alternating AR2 ranker and retriever updates.
6. Corpus-index and hard-negative refresh after every AR2 round.
7. Retriever and retrieve-then-rerank evaluation.

Keep the terminal session alive with `tmux`, `screen`, or the cluster scheduler.

## 9. Optional Slurm job

Adapt partition, account, CPU, memory, and time limits to the cluster:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=vault-ruby-ar2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/vault-ruby-ar2-%j.log

set -euo pipefail

cd /path/to/retrieval-baseline/AR2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ar2

python -m code_search.run_pipeline \
  --config configs/vault_ruby_a100.yaml \
  --skip-prepare
```

Submit it with:

```bash
sbatch run_vault_ar2.sbatch
```

## 10. Monitor training

GPU utilization:

```bash
watch -n 5 nvidia-smi
```

Training log:

```bash
tail -f logs/vault-ruby-ar2.log
```

Healthy behavior:

- GPU utilization remains high during model training and corpus encoding.
- CPU utilization can rise during tokenization and FAISS search.
- Retriever and ranker losses remain finite.
- A new checkpoint and hard-negative file appear after every AR2 round.

## 11. Output files

Expected files under `outputs/vault-ruby/checkpoints/`:

```text
retriever-warm.pt
ranker-warm.pt
hard-negatives-initial.jsonl
retriever-ar2-round-1.pt
ranker-ar2-round-1.pt
hard-negatives-round-1.jsonl
...
retriever-ar2-round-5.pt
ranker-ar2-round-5.pt
hard-negatives-round-5.jsonl
metrics-dev.json
metrics-test.json
```

The highest completed round is the final model. Evaluation reports Hit,
multi-label Recall, MRR, MAP, and nDCG at the configured cutoffs.

## 12. Run stages manually

### 12.1 Warm retriever

```bash
python -m code_search.train_retriever \
  --config configs/vault_ruby_a100.yaml
```

### 12.2 Mine initial hard negatives

```bash
python -m code_search.mine_hard_negatives \
  --config configs/vault_ruby_a100.yaml \
  --checkpoint outputs/vault-ruby/checkpoints/retriever-warm.pt \
  --split train \
  --output outputs/vault-ruby/checkpoints/hard-negatives-initial.jsonl
```

### 12.3 Warm ranker

```bash
python -m code_search.train_ranker \
  --config configs/vault_ruby_a100.yaml \
  --hard-negatives outputs/vault-ruby/checkpoints/hard-negatives-initial.jsonl
```

### 12.4 Run AR2 co-training

```bash
python -m code_search.train_ar2 \
  --config configs/vault_ruby_a100.yaml \
  --retriever-checkpoint outputs/vault-ruby/checkpoints/retriever-warm.pt \
  --ranker-checkpoint outputs/vault-ruby/checkpoints/ranker-warm.pt \
  --hard-negatives outputs/vault-ruby/checkpoints/hard-negatives-initial.jsonl
```

### 12.5 Evaluate a checkpoint

```bash
python -m code_search.evaluate \
  --config configs/vault_ruby_a100.yaml \
  --retriever-checkpoint outputs/vault-ruby/checkpoints/retriever-ar2-round-5.pt \
  --ranker-checkpoint outputs/vault-ruby/checkpoints/ranker-ar2-round-5.pt \
  --split test \
  --output outputs/vault-ruby/checkpoints/metrics-test.json
```

## 13. Resume an interrupted AR2 run

If round 3 completed successfully, resume at round 4 with the matching model
and hard-negative checkpoints:

```bash
python -m code_search.train_ar2 \
  --config configs/vault_ruby_a100.yaml \
  --retriever-checkpoint outputs/vault-ruby/checkpoints/retriever-ar2-round-3.pt \
  --ranker-checkpoint outputs/vault-ruby/checkpoints/ranker-ar2-round-3.pt \
  --hard-negatives outputs/vault-ruby/checkpoints/hard-negatives-round-3.jsonl \
  --start-round 4
```

The supplied retriever checkpoint, ranker checkpoint, and hard-negative file
must come from the same round. Optimizer states are restored from the model
checkpoints.

## 14. Configuration adjustments

### Out of GPU memory

Reduce these values in order:

```yaml
ranker:
  per_device_batch_size: 4

ar2:
  per_device_batch_size: 4
```

Increase gradient accumulation proportionally to preserve effective batch
size. If necessary, reduce:

```yaml
ranker:
  max_length: 256

retriever:
  code_max_length: 192
```

### Too much code truncation

Increase `retriever.code_max_length` and `ranker.max_length` only after checking
the audit length distribution. Increasing pair length substantially increases
ranker memory and runtime.

### Quick pilot run

For a pipeline validation run before the final experiment:

```yaml
retriever:
  warmup_epochs: 1

ranker:
  warmup_epochs: 1

ar2:
  rounds: 1
  ranker_steps_per_round: 20
  retriever_steps_per_round: 50
```

Restore the production values before reporting final metrics.

## 15. Troubleshooting

### Conda reports `Found conflicts` and fails

The earlier environment combined the `pytorch-gpu` and `faiss-cpu` Conda
packages from different channels. Do not keep retrying that file. Sync the
updated `environment-code-search.yml`, then run:

```bash
cd /path/to/retrieval-baseline/AR2
conda env list

# Run this only if "ar2" appears in the environment list.
conda env remove -n ar2 -y

conda env create -f environment-code-search.yml
conda activate ar2
```

A failed solve normally does not create the environment, so the removal step
is often unnecessary. If a recent Conda installation still spends a long time
using the classic solver, use its libmamba solver:

```bash
conda env create -f environment-code-search.yml --solver=libmamba
```

Warnings stating that `.*` is superfluous come from dependency records in the
Conda index. They are deprecation warnings and are not the cause of the failed
solve.

### Missing positive documents

Symptom:

```text
positive document references are missing from the code corpus
```

Actions:

1. Confirm the complete train file is being used, not the repository sample.
2. Confirm the train file contains all `Code:` rows.
3. Confirm query and code records use matching `text_id` values.
4. Do not bypass this error for production training.

### CUDA is unavailable

Actions:

1. Check `nvidia-smi`.
2. Confirm the job was allocated a GPU.
3. If the driver is too old for CUDA 12.1, change the environment file to
   `pytorch-cuda=11.8` and recreate the environment.
4. Re-run the PyTorch GPU verification command.

### FAISS cannot be imported

```bash
conda activate ar2
python -m pip install --upgrade faiss-cpu
```

For the expected Ruby corpus size, CPU FAISS is sufficient. The expensive
transformer encoding still runs on the A100.

### Ranker reports no hard negatives

Actions:

1. Confirm `hard-negatives-initial.jsonl` is not empty.
2. Confirm the corpus contains documents other than the query's positives.
3. Increase `mining.top_k` if too many top results are correct documents.
4. Confirm all positive IDs are filtered rather than being treated as negatives.

### Hugging Face download fails

Actions:

1. Confirm network access and authentication settings.
2. Download `microsoft/codebert-base` on a connected machine.
3. Copy the model directory to the server.
4. Set both model names in the YAML file to the local directory.

### Retriever and ranker tokenizer mismatch

Use the same `model_name` for both models unless the co-training collator is
extended to tokenize retriever and ranker inputs independently.

### Loss becomes NaN or infinite

Actions:

1. Confirm `runtime.precision` is `bf16` on the A100.
2. Reduce learning rates.
3. Confirm each training query has at least one positive and one negative.
4. Inspect the latest hard-negative file for empty candidate lists.

## 16. Completion checklist

- [ ] Complete train and test files are configured.
- [ ] CUDA-enabled PyTorch detects the A100.
- [ ] All regression tests pass.
- [ ] Data preparation completes.
- [ ] `missing_positive_references` is zero.
- [ ] Prepared-data validation reports `valid: True`.
- [ ] Warm retriever checkpoint exists.
- [ ] Initial hard-negative file exists and is non-empty.
- [ ] Warm ranker checkpoint exists.
- [ ] All configured AR2 rounds complete.
- [ ] Final dev and test metrics are written.
- [ ] Configuration, logs, checkpoints, and metrics are retained for reproducibility.
