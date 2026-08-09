# TheVault multi-label DPR

This integration trains DPR directly from original `Query:` rows and retrieves
`Code:` rows. Pseudo-query files are not used. The commands below target a
Linux server with one NVIDIA A100 and should be run from the `DPR` directory.

## 1. Create the environment

Python 3.9 is recommended for this legacy DPR codebase:

```bash
conda create -n thevault-dpr python=3.9 -y
conda activate thevault-dpr
cd /path/to/retrieval-baseline/DPR
```

Install a CUDA-enabled PyTorch build compatible with the server. For a server
whose NVIDIA driver supports CUDA 11.8:

```bash
conda install pytorch==2.0.1 pytorch-cuda=11.8 \
  -c pytorch -c nvidia -y
```

Install the remaining dependencies with versions compatible with this DPR
implementation:

```bash
pip install \
  "transformers==4.30.2" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "numpy<2" \
  faiss-cpu \
  filelock \
  regex \
  tqdm \
  wget \
  jsonlines \
  soundfile \
  editdistance \
  "thinc==8.2.5" \
  "spacy==3.7.5"

python -m pip install -e . --no-deps --no-build-isolation
```

Verify that PyTorch can see the A100:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The supplied training configuration uses FP32, so NVIDIA Apex is not required.
The legacy `fp16=True` path still requires Apex.

## 2. Download encoder models for offline use

### BERT

This DPR pipeline uses `bert-base-uncased` for both the question encoder and
the context encoder. It does not use `t5-base`. On a login node or another
Linux machine that can access Hugging Face, download and save the complete
model and tokenizer:

```bash
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
mkdir -p "$THEVAULT_BERT_DIR"

python - <<'PY'
import os
from transformers import BertModel, BertTokenizer

model_name = "bert-base-uncased"
output_dir = os.environ["THEVAULT_BERT_DIR"]

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir, safe_serialization=False)

print(f"Saved {model_name} to {output_dir}")
PY
```

If the cluster has no internet access at all, run the block above on an
internet-connected machine with a suitable local output path. Copy the saved
directory to the cluster:

```bash
rsync -av /local/path/bert-base-uncased/ \
  USER@CLUSTER:/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased/
```

Replace `USER@CLUSTER` with your cluster SSH destination. The copied directory
must include `config.json`, `pytorch_model.bin`, and `vocab.txt`.

Verify that the model can be loaded without network access:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import os
from transformers import BertModel, BertTokenizer

model_dir = os.environ["THEVAULT_BERT_DIR"]
BertTokenizer.from_pretrained(model_dir, local_files_only=True)
BertModel.from_pretrained(model_dir, local_files_only=True)
print(f"Offline BERT check passed: {model_dir}")
PY
```

Export `THEVAULT_BERT_DIR`, `HF_HUB_OFFLINE`, and `TRANSFORMERS_OFFLINE` in
every new shell or batch job. Offline mode prevents Hugging Face from retrying
internet requests when a local file is missing.

### CodeBERT

CodeBERT uses the Hugging Face RoBERTa architecture and tokenizer, not BERT's
WordPiece tokenizer. Download `microsoft/codebert-base` on a machine with
internet access:

```bash
export THEVAULT_CODEBERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/codebert-base
mkdir -p "$THEVAULT_CODEBERT_DIR"

python - <<'PY'
import os
from transformers import RobertaModel, RobertaTokenizer

model_name = "microsoft/codebert-base"
output_dir = os.environ["THEVAULT_CODEBERT_DIR"]

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir, safe_serialization=False)

print(f"Saved {model_name} to {output_dir}")
PY
```

If the GPU cluster cannot access Hugging Face, copy the complete directory from
the internet-connected machine:

```bash
rsync -av /local/path/codebert-base/ \
  USER@CLUSTER:/mnt/beegfs/scratch/congthanh_le/east/baseline/models/codebert-base/
```

The directory must include `config.json`, `pytorch_model.bin`, `vocab.json`,
and `merges.txt`. Verify the local copy without network access:

```bash
export THEVAULT_CODEBERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/codebert-base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import os
from transformers import RobertaModel, RobertaTokenizer

model_dir = os.environ["THEVAULT_CODEBERT_DIR"]
RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
RobertaModel.from_pretrained(model_dir, local_files_only=True)
print(f"Offline CodeBERT check passed: {model_dir}")
PY
```

CodeBERT needs no additional dependency beyond the pinned PyTorch and
Transformers versions from Section 1. Do not point `encoder=hf_bert` at this
directory; use the `encoder=hf_codebert` profile shown below.

## 3. Prepare the data

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

## 4. Run the tests

```bash
python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

The data-preparation and metric tests use only the Python standard library. The
multi-positive loss tests require PyTorch and the DPR dependencies.

## 5. Train DPR

### Version A: BERT with multi-positive loss

The initial baseline uses original queries, multi-positive in-batch contrastive
learning, and no mined hard negatives:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
  datasets=thevault_ruby \
  train=biencoder_thevault_a100 \
  'train_datasets=[ruby_train]' \
  'dev_datasets=[ruby_dev]' \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  "output_dir=$THEVAULT_DPR_ROOT/outputs/thevault_ruby" \
  fp16=False \
  hydra.job.chdir=False
```

`hydra.job.chdir=False` prevents Hydra from changing into an
`outputs/YYYY-MM-DD/HH-MM-SS` directory before resolving relative paths. The
absolute application output path above keeps checkpoints directly under
`$THEVAULT_DPR_ROOT/outputs/thevault_ruby`.

The A100 profile starts with batch size 64, up to eight positive documents per
query, ten epochs, and FP32. With `keep_last_n: 2`, it retains at most two
checkpoint files, normally the best validation checkpoint and the latest one.

Training logs the best checkpoint path. The final checkpoint will normally be:

```text
outputs/thevault_ruby/dpr_biencoder.9
```

Set the checkpoint to use for embedding and retrieval. Replace the value with
the best checkpoint reported by training when appropriate:

```bash
export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby/dpr_biencoder.9"
test -f "$THEVAULT_MODEL" && echo "Checkpoint found: $THEVAULT_MODEL"
```

If embedding or retrieval is started from a new shell or a separate batch job,
export `THEVAULT_DPR_ROOT`, `THEVAULT_BERT_DIR`, `HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, and `THEVAULT_MODEL` again in that job.

### Version B: BERT with legacy DPR loss and gold in-batch negatives

This alternative keeps the original single-positive `BiEncoderNllLoss`. For
each query, the selected gold document is the target and gold documents paired
with other questions in the same mini-batch act as negatives. Before computing
the softmax, the code masks any in-batch document whose ID occurs in that
query's complete multi-label `positive_ids`, preventing a known positive from
being trained as a negative:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
  datasets=thevault_ruby \
  train=biencoder_thevault_legacy_inbatch_a100 \
  'train_datasets=[ruby_train]' \
  'dev_datasets=[ruby_dev]' \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  "output_dir=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch" \
  fp16=False \
  hydra.job.chdir=False

export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/dpr_biencoder.9"
test -f "$THEVAULT_MODEL" && echo "Checkpoint found: $THEVAULT_MODEL"
```

Replace `.9` with the best checkpoint reported by training when appropriate.
Version B uses no separate mining pass and no `train_hard_negatives.jsonl`.
Its configuration sets `hard_negatives: 0`, `multi_positive: false`, and
`mask_multilabel_false_negatives: true`. Version A remains unchanged.

### Version C: CodeBERT with legacy DPR loss and gold in-batch negatives

This is the recommended first CodeBERT comparison. It uses the same Version B
loss and multi-label false-negative mask, while initializing both independent
encoders from `microsoft/codebert-base`. CodeBERT uses `<s>` as its vector
position and does not lowercase the query or code:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_CODEBERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/codebert-base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
  datasets=thevault_ruby \
  encoder=hf_codebert \
  train=biencoder_thevault_legacy_inbatch_a100 \
  'train_datasets=[ruby_train]' \
  'dev_datasets=[ruby_dev]' \
  "encoder.pretrained_model_cfg=$THEVAULT_CODEBERT_DIR" \
  "output_dir=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch" \
  do_lower_case=False \
  fp16=False \
  hydra.job.chdir=False

export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/dpr_biencoder.9"
test -f "$THEVAULT_MODEL" && echo "Checkpoint found: $THEVAULT_MODEL"
```

Replace `.9` with the best checkpoint path printed by training. If batch size
64 does not fit, append `train.batch_size=32 train.dev_batch_size=64`; the loss
and negative masking do not change.

To compare CodeBERT with Version A's multi-positive loss, change only these
overrides and keep the remaining CodeBERT command unchanged:

```bash
train=biencoder_thevault_a100
"output_dir=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_multi_positive"
```

BERT and CodeBERT checkpoints are not interchangeable. A CodeBERT checkpoint
contains both the trained CodeBERT query encoder and context encoder in the
same `dpr_biencoder.N` file.

## 6. Encode the corpus

### Version A: BERT multi-positive checkpoint

For a single embedding shard:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$THEVAULT_DPR_ROOT/outputs/thevault_ruby/embeddings"

CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby/embeddings/ruby" \
  batch_size=128 \
  shard_id=0 \
  num_shards=1 \
  hydra.job.chdir=False
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
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby/embeddings/ruby" \
  batch_size=128 \
  shard_id=0 \
  num_shards=4 \
  hydra.job.chdir=False
```

Repeat with `shard_id=1`, `2`, and `3`.

### Version B: BERT legacy-loss gold in-batch checkpoint

The final context encoder differs from Version A, so write its embeddings to a
separate directory:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/dpr_biencoder.9"

mkdir -p "$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/embeddings"

CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/embeddings/ruby" \
  batch_size=128 \
  shard_id=0 \
  num_shards=1 \
  hydra.job.chdir=False
```

The Version B embedding file is
`outputs/thevault_ruby_legacy_inbatch/embeddings/ruby_0`.

### Version C: CodeBERT legacy-loss checkpoint

CodeBERT has a separately trained context encoder, so it requires a fresh
corpus embedding file. Do not reuse either BERT embedding directory:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_CODEBERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/codebert-base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/dpr_biencoder.9"

mkdir -p "$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/embeddings"

CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
  encoder=hf_codebert \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  "encoder.pretrained_model_cfg=$THEVAULT_CODEBERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/embeddings/ruby" \
  do_lower_case=False \
  batch_size=128 \
  shard_id=0 \
  num_shards=1 \
  hydra.job.chdir=False
```

The resulting file is
`outputs/thevault_ruby_codebert_legacy_inbatch/embeddings/ruby_0`. If you used
the CodeBERT multi-positive output instead, change both `THEVAULT_MODEL` and
the embedding output directory to `thevault_ruby_codebert_multi_positive`.

## 7. Retrieve and evaluate

### Version A: BERT multi-positive checkpoint

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -z "${THEVAULT_MODEL:-}" ] || [ ! -f "$THEVAULT_MODEL" ]; then
  echo "Set THEVAULT_MODEL to an existing DPR checkpoint before retrieval"
  exit 1
fi

echo "Using checkpoint: $THEVAULT_MODEL"

CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
  datasets=thevault_ruby \
  ctx_sources=thevault \
  qa_dataset=ruby_test \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  'ctx_datatsets=[ruby_code]' \
  "encoded_ctx_files=[$THEVAULT_DPR_ROOT/outputs/thevault_ruby/embeddings/ruby_*]" \
  validation_mode=document_ids \
  n_docs=100 \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby/test_results.json" \
  hydra.job.chdir=False
```

### Version B: BERT legacy-loss gold in-batch checkpoint

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/dpr_biencoder.9"

CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
  datasets=thevault_ruby \
  ctx_sources=thevault \
  qa_dataset=ruby_test \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  'ctx_datatsets=[ruby_code]' \
  "encoded_ctx_files=[$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/embeddings/ruby_*]" \
  validation_mode=document_ids \
  n_docs=100 \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_inbatch/test_results.json" \
  hydra.job.chdir=False
```

### Version C: CodeBERT legacy-loss checkpoint

Use the query encoder from the same CodeBERT checkpoint that produced the
CodeBERT corpus embeddings:

```bash
export THEVAULT_DPR_ROOT="$(pwd -P)"
export THEVAULT_CODEBERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/codebert-base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THEVAULT_MODEL="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/dpr_biencoder.9"

CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
  datasets=thevault_ruby \
  encoder=hf_codebert \
  ctx_sources=thevault \
  qa_dataset=ruby_test \
  "encoder.pretrained_model_cfg=$THEVAULT_CODEBERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  'ctx_datatsets=[ruby_code]' \
  "encoded_ctx_files=[$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/embeddings/ruby_*]" \
  validation_mode=document_ids \
  do_lower_case=False \
  n_docs=100 \
  "out_file=$THEVAULT_DPR_ROOT/outputs/thevault_ruby_codebert_legacy_inbatch/test_results.json" \
  hydra.job.chdir=False
```

Export `THEVAULT_MODEL` in the current shell before running this block. The
checkpoint must be the same one used to generate the `ruby_*` embeddings. The
explicit `model_file` override is required: exporting `THEVAULT_MODEL` alone
does not automatically populate Hydra's `model_file` setting.

`ctx_datatsets` is intentionally misspelled because that is the parameter name
used by the original DPR repository.

The result file reports Hit, Precision, Recall, MRR, MAP, and nDCG at the
configured cutoffs. Each unique normalized query is evaluated once against its
complete set of relevant `text_id` values.

Print Version A aggregate metrics:

```bash
python -c "import json; result=json.load(open('outputs/thevault_ruby/test_results.json')); print(json.dumps(result['aggregate_metrics'], indent=2))"
```

Print Version B aggregate metrics:

```bash
python -c "import json; result=json.load(open('outputs/thevault_ruby_legacy_inbatch/test_results.json')); print(json.dumps(result['aggregate_metrics'], indent=2))"
```

Print Version C aggregate metrics:

```bash
python -c "import json; result=json.load(open('outputs/thevault_ruby_codebert_legacy_inbatch/test_results.json')); print(json.dumps(result['aggregate_metrics'], indent=2))"
```
