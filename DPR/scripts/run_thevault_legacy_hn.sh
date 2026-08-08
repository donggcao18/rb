#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. The conda environment must already be activated.
THEVAULT_DPR_ROOT="${THEVAULT_DPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
THEVAULT_BERT_DIR="${THEVAULT_BERT_DIR:-/mnt/beegfs/scratch/congthanh_le/east/baseline/models/bert-base-uncased}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HARD_NEGATIVE_POOL_SIZE="${HARD_NEGATIVE_POOL_SIZE:-20}"
MINING_TOP_K="${MINING_TOP_K:-100}"

export THEVAULT_DPR_ROOT
export THEVAULT_BERT_DIR
export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASELINE_DIR="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_baseline"
FINAL_DIR="$THEVAULT_DPR_ROOT/outputs/thevault_ruby_legacy_hn"
BASELINE_LOG="$BASELINE_DIR/train.log"
FINAL_LOG="$FINAL_DIR/train.log"
MINING_EMBED_PREFIX="$BASELINE_DIR/embeddings/ruby"
TRAIN_RETRIEVAL="$BASELINE_DIR/train_retrieval.json"
MINED_TRAIN_FILE="$THEVAULT_DPR_ROOT/data/thevault/ruby/train_hard_negatives.jsonl"
FINAL_EMBED_PREFIX="$FINAL_DIR/embeddings/ruby"
TEST_RESULTS="$FINAL_DIR/test_results.json"

required_files=(
  "$THEVAULT_BERT_DIR/config.json"
  "$THEVAULT_BERT_DIR/vocab.txt"
  "$THEVAULT_DPR_ROOT/data/thevault/ruby/corpus.tsv"
  "$THEVAULT_DPR_ROOT/data/thevault/ruby/train.jsonl"
  "$THEVAULT_DPR_ROOT/data/thevault/ruby/dev.jsonl"
  "$THEVAULT_DPR_ROOT/data/thevault/ruby/test.jsonl"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

if [[ ! -f "$THEVAULT_BERT_DIR/pytorch_model.bin" && ! -f "$THEVAULT_BERT_DIR/model.safetensors" ]]; then
  echo "No BERT weight file found under $THEVAULT_BERT_DIR" >&2
  exit 1
fi

mkdir -p "$BASELINE_DIR/embeddings" "$FINAL_DIR/embeddings"
cd "$THEVAULT_DPR_ROOT"

select_best_checkpoint() {
  local log_file="$1"
  local checkpoint_dir="$2"
  local checkpoint

  checkpoint="$(sed -n 's/.*Training finished\. Best validation checkpoint //p' "$log_file" | tail -n 1)"
  if [[ -z "$checkpoint" || "$checkpoint" == "None" ]]; then
    checkpoint="$checkpoint_dir/dpr_biencoder.9"
  elif [[ "$checkpoint" != /* ]]; then
    checkpoint="$THEVAULT_DPR_ROOT/$checkpoint"
  fi

  if [[ ! -f "$checkpoint" ]]; then
    echo "Selected checkpoint does not exist: $checkpoint" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint"
}

echo "[1/7] Training legacy-loss baseline"
python train_dense_encoder.py \
  datasets=thevault_ruby \
  train=biencoder_thevault_legacy_hn_a100 \
  'train_datasets=[ruby_train]' \
  'dev_datasets=[ruby_dev]' \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  "output_dir=$BASELINE_DIR" \
  fp16=False \
  hydra.job.chdir=False \
  2>&1 | tee "$BASELINE_LOG"

THEVAULT_BASELINE_MODEL="$(select_best_checkpoint "$BASELINE_LOG" "$BASELINE_DIR")"
export THEVAULT_BASELINE_MODEL
echo "Using baseline checkpoint: $THEVAULT_BASELINE_MODEL"

echo "[2/7] Encoding corpus for hard-negative mining"
python generate_dense_embeddings.py \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_BASELINE_MODEL" \
  "out_file=$MINING_EMBED_PREFIX" \
  batch_size=128 \
  shard_id=0 \
  num_shards=1 \
  hydra.job.chdir=False

if [[ ! -s "${MINING_EMBED_PREFIX}_0" ]]; then
  echo "Mining embedding file was not created: ${MINING_EMBED_PREFIX}_0" >&2
  exit 1
fi

echo "[3/7] Retrieving candidates for every training query"
python dense_retriever.py \
  datasets=thevault_ruby \
  ctx_sources=thevault \
  qa_dataset=ruby_train_for_mining \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_BASELINE_MODEL" \
  'ctx_datatsets=[ruby_code]' \
  "encoded_ctx_files=[${MINING_EMBED_PREFIX}_*]" \
  validation_mode=document_ids \
  n_docs="$MINING_TOP_K" \
  "out_file=$TRAIN_RETRIEVAL" \
  hydra.job.chdir=False

echo "[4/7] Filtering every known positive and writing hard negatives"
python scripts/mine_thevault_hard_negatives.py \
  --train "$THEVAULT_DPR_ROOT/data/thevault/ruby/train.jsonl" \
  --retrieval-results "$TRAIN_RETRIEVAL" \
  --output "$MINED_TRAIN_FILE" \
  --num-hard-negatives "$HARD_NEGATIVE_POOL_SIZE"

if [[ ! -s "$MINED_TRAIN_FILE" ]]; then
  echo "Mined training file was not created: $MINED_TRAIN_FILE" >&2
  exit 1
fi

echo "[5/7] Training legacy DPR with mined hard negatives"
python train_dense_encoder.py \
  datasets=thevault_ruby \
  train=biencoder_thevault_legacy_hn_a100 \
  'train_datasets=[ruby_train_hn]' \
  'dev_datasets=[ruby_dev]' \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_BASELINE_MODEL" \
  "output_dir=$FINAL_DIR" \
  ignore_checkpoint_offset=true \
  ignore_checkpoint_optimizer=true \
  ignore_checkpoint_lr=true \
  fp16=False \
  hydra.job.chdir=False \
  2>&1 | tee "$FINAL_LOG"

THEVAULT_MODEL="$(select_best_checkpoint "$FINAL_LOG" "$FINAL_DIR")"
export THEVAULT_MODEL
printf '%s\n' "$THEVAULT_MODEL" > "$FINAL_DIR/best_checkpoint.txt"
echo "Using final checkpoint: $THEVAULT_MODEL"

if [[ "${STOP_AFTER_FINAL_TRAIN:-0}" == "1" ]]; then
  echo "Stopped after final hard-negative training as requested"
  echo "Final checkpoint: $THEVAULT_MODEL"
  exit 0
fi

echo "[6/7] Encoding corpus with the final context encoder"
python generate_dense_embeddings.py \
  ctx_sources=thevault \
  ctx_src=ruby_code \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  "out_file=$FINAL_EMBED_PREFIX" \
  batch_size=128 \
  shard_id=0 \
  num_shards=1 \
  hydra.job.chdir=False

if [[ ! -s "${FINAL_EMBED_PREFIX}_0" ]]; then
  echo "Final embedding file was not created: ${FINAL_EMBED_PREFIX}_0" >&2
  exit 1
fi

echo "[7/7] Retrieving and evaluating multi-label test queries"
python dense_retriever.py \
  datasets=thevault_ruby \
  ctx_sources=thevault \
  qa_dataset=ruby_test \
  "encoder.pretrained_model_cfg=$THEVAULT_BERT_DIR" \
  model_file="$THEVAULT_MODEL" \
  'ctx_datatsets=[ruby_code]' \
  "encoded_ctx_files=[${FINAL_EMBED_PREFIX}_*]" \
  validation_mode=document_ids \
  n_docs=100 \
  "out_file=$TEST_RESULTS" \
  hydra.job.chdir=False

export TEST_RESULTS
python - <<'PY'
import json
import os

with open(os.environ["TEST_RESULTS"], encoding="utf-8") as stream:
    results = json.load(stream)
print(json.dumps(results["aggregate_metrics"], indent=2, sort_keys=True))
PY

echo "Pipeline completed"
echo "Final checkpoint: $THEVAULT_MODEL"
echo "Final embeddings: ${FINAL_EMBED_PREFIX}_0"
echo "Test results: $TEST_RESULTS"
