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
  editdistance

  "thinc==8.2.5" \
  "spacy==3.7.5" \

pip install -e . --no-deps
```

Verify that PyTorch can see the A100:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The supplied training configuration uses FP32, so NVIDIA Apex is not required.
The legacy `fp16=True` path still requires Apex.

## 2. Download BERT for offline use

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

The initial baseline uses original queries, multi-positive in-batch contrastive
learning, and no mined hard negatives:

```bash
export THEVAULT_BERT_DIR=/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
  datasets=thevault_ruby \
  train=biencoder_thevault_a100 \
  'train_datasets=[ruby_train]' \
  'dev_datasets=[ruby_dev]' \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
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

## 6. Encode the corpus

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

## 7. Retrieve and evaluate

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
