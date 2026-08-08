# TheVault multi-label DPR

This integration trains DPR directly from original `Query:` rows and retrieves
`Code:` rows. Pseudo-query files are not used. The commands below target a
Linux server with one NVIDIA A100 and should be run from the `DPR` directory.

## 1. Install the dependencies

Use a new environment rather than reusing an environment that contains a
damaged CUDA library. Python 3.9 is recommended for this legacy DPR codebase:

```bash
conda create -n dpr python=3.9 pip -y
conda activate dpr
cd /path/to/retrieval-baseline/DPR

python -m pip install --upgrade pip setuptools wheel
```

The A100 with NVIDIA driver 580.95.05 can run an older CUDA 11.8 application.
The CUDA 13.0 value printed by `nvidia-smi` is the maximum version supported by
the driver; PyTorch does not need to use CUDA 13. Install the official PyTorch
2.0.1 CUDA 11.8 wheel:

```bash
python -m pip install \
  torch==2.0.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Verify PyTorch immediately, before installing the other packages:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

The expected values are PyTorch `2.0.1+cu118`, CUDA runtime `11.8`, and an
`NVIDIA A100-SXM4-80GB` GPU. If this import reports `invalid ELF header`, stop:
the environment or package extraction is corrupted, or the filesystem is out
of space/quota.

On clusters, installation is commonly performed on a login/head node where no
GPU is exposed. `CUDA available: False` is expected there. Repeat the PyTorch
check inside an allocated GPU job; training must also run inside that allocation.

Install the remaining pinned dependencies:

```bash
python -m pip install \
  "transformers==4.30.2" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "numpy<2" \
  "faiss-cpu==1.7.4" \
  filelock \
  regex \
  tqdm \
  wget \
  jsonlines \
  soundfile \
  editdistance

python -m pip install -e . --no-deps --no-build-isolation
```

spaCy and `en_core_web_sm` are not required for the TheVault pipeline. They are
loaded lazily only by DPR's legacy table and answer-tokenization workflows.

Verify the complete environment:

```bash
python - <<'PY'
import faiss
import hydra
import jsonlines
import torch
import transformers

print("Environment OK")
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Run installation on the login node if compute nodes cannot access PyPI. The
supplied training configuration uses FP32, so NVIDIA Apex is not required. The
legacy `fp16=True` path still requires Apex.

## 2. Prepare the data

The complete input files should be available at:

```text
theVault-multilabel/Ruby_train_r32.0.json
theVault-multilabel/Ruby_test_r32.0.json
```

From the `DPR` directory, run:

```bash
python scripts/prepare_thevault.py \
  --train /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_train_r32.0.json \
  --test /home/users/congthanh_le/scratch/east/CodeGR/data/original_indexed_data_RQ_8_16_decoder_start/Ruby_test_r32.0.json \
  --output-dir data/thevault/ruby \
  --dev-ratio 0.05 \
  --test-relevance-scope global
```

This creates:

```text
data/thevault/ruby/
├── corpus.tsv
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── manifest.json
```

Inspect the integrity report before training:

```bash
cat data/thevault/ruby/manifest.json
```

`missing_positive_ids` must be `0`. The preparation command fails by default if
a query's positive `text_id` is absent from the code corpus.

The `global` relevance policy reproduces the existing `build_multilable.py`
behavior: an identical normalized query receives the union of labels from train
and test. Use `--test-relevance-scope test` to derive test labels only from test
rows.

## 3. Run the tests

```bash
python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

The data-preparation and metric tests use only the Python standard library. The
multi-positive loss tests require PyTorch and the DPR dependencies.

## 4. Train DPR

The initial baseline uses original queries, multi-positive in-batch contrastive
learning, and no mined hard negatives:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
  datasets=thevault_ruby \
  train=biencoder_thevault_a100 \
  'train_datasets=[ruby_train]' \
  'dev_datasets=[ruby_dev]' \
  output_dir=outputs/thevault_ruby \
  fp16=False
```

The A100 profile starts with batch size 64, up to eight positive documents per
query, ten epochs, and FP32. It retains the current best checkpoint plus the two
most recent checkpoint files.

Training logs the best checkpoint path. The final checkpoint will normally be:

```text
outputs/thevault_ruby/dpr_biencoder.9
```

Set the checkpoint to use for embedding and retrieval. Replace the value with
the best checkpoint reported by training when appropriate:

```bash
THEVAULT_MODEL=outputs/thevault_ruby/dpr_biencoder.9
```

## 5. Encode the corpus

For a single embedding shard:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  model_file="$THEVAULT_MODEL" \
  out_file=outputs/thevault_ruby/embeddings/ruby \
  batch_size=128 \
  shard_id=0 \
  num_shards=1
```

The resulting file is:

```text
outputs/thevault_ruby/embeddings/ruby_0
```

For a large corpus, run multiple processes with a shared `num_shards` value and
a different zero-based `shard_id` for each process. For example, the first of
four shards is:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  model_file="$THEVAULT_MODEL" \
  out_file=outputs/thevault_ruby/embeddings/ruby \
  batch_size=128 \
  shard_id=0 \
  num_shards=4
```

Repeat with `shard_id=1`, `2`, and `3`.

## 6. Retrieve and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
  datasets=thevault_ruby \
  ctx_sources=thevault \
  qa_dataset=ruby_test \
  'ctx_datatsets=[ruby_code]' \
  'encoded_ctx_files=[outputs/thevault_ruby/embeddings/ruby_*]' \
  validation_mode=document_ids \
  n_docs=100 \
  out_file=outputs/thevault_ruby/test_results.json
```

`ctx_datatsets` is intentionally misspelled because that is the parameter name
used by the original DPR repository.

The result file reports Hit, Precision, Recall, MRR, MAP, and nDCG at the
configured cutoffs. Each unique normalized query is evaluated once against its
complete set of relevant `text_id` values.

Print the aggregate metrics:

```bash
python -c "import json; result=json.load(open('outputs/thevault_ruby/test_results.json')); print(json.dumps(result['aggregate_metrics'], indent=2))"
```
