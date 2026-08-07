# TheVault multi-label DPR

This integration trains DPR directly from original `Query:` rows and retrieves
`Code:` rows. Pseudo-query files are not read.

## 1. Prepare the data

Place the complete train and test files outside generated output directories,
then run from the `DPR` directory:

```powershell
python scripts/prepare_thevault.py `
  --train ../theVault-multilabel/Ruby_train_r32.0.json `
  --test ../theVault-multilabel/Ruby_test_r32.0.json `
  --output-dir data/thevault/ruby `
  --dev-ratio 0.05 `
  --test-relevance-scope global
```

The command creates `corpus.tsv`, `train.jsonl`, `dev.jsonl`, `test.jsonl`,
and `manifest.json`. It fails if a query's positive `text_id` is absent from
the code corpus. The manifest records duplicate-query and missing-ID statistics.

`global` test relevance reproduces the existing `build_multilable.py` behavior:
an identical normalized query receives the union of labels from train and test.
Use `--test-relevance-scope test` to derive test labels from test rows only.

## 2. Train

The initial baseline uses original queries, multi-positive in-batch contrastive
learning, and no mined hard negatives:

```powershell
python train_dense_encoder.py `
  datasets=thevault_ruby `
  train=biencoder_thevault_a100 `
  train_datasets=[ruby_train] `
  dev_datasets=[ruby_dev] `
  output_dir=outputs/thevault_ruby
```

The A100 profile starts with batch size 64, up to eight positive documents per
query, ten epochs, and FP32. The old `fp16=True` option still requires Apex;
mixed precision modernization is deliberately separate from data correctness.
The profile retains the current best checkpoint plus the two most recent files.

## 3. Encode the corpus

```powershell
python generate_dense_embeddings.py `
  ctx_sources=thevault `
  ctx_src=ruby_code `
  model_file=outputs/thevault_ruby/dpr_biencoder.9 `
  out_file=outputs/thevault_ruby/embeddings/ruby
```

Use `shard_id` and `num_shards` to split a large corpus across workers.

## 4. Retrieve and evaluate

```powershell
python dense_retriever.py `
  datasets=thevault_ruby `
  ctx_sources=thevault `
  qa_dataset=ruby_test `
  ctx_datatsets=[ruby_code] `
  encoded_ctx_files=[outputs/thevault_ruby/embeddings/ruby_*] `
  validation_mode=document_ids `
  n_docs=100 `
  out_file=outputs/thevault_ruby/test_results.json
```

The output reports Hit, Precision, Recall, MRR, MAP, and nDCG at the configured
cutoffs. Each unique normalized query is evaluated once against its full set of
relevant `text_id` values.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The data-preparation and metric tests use only the Python standard library.
Loss tests run when PyTorch and the DPR dependencies are installed.
