# TEAL: Efficient Structure-Guided Hierarchical Analytics via Inter-Column Dependency Modeling

We propose **TEAL** (**ST**ructure-Guid**E**d Hier**A**rchica**L** Analytics), a framework that integrates a learned causal-inspired inter-column dependency structure into structured-data predictive modeling through two complementary phases: the learned structure acts as a soft structural prior to modulate column-wise self-attention, and the derived target-relative dependency hierarchy provides distance-based constraints on the aggregation of column representations for the final prediction.

The inter-column dependency structure — a target-conditioned column-level dependency profile — is learned through a self-supervised reconstruction component, regularized by sparsity and acyclicity constraints. An alternating optimization scheme coordinates dependency-structure learning and structure-guided prediction, and a lazy-update strategy reduces the overhead of dependency-structure updates.

> **Naming note**: the codebase refers to the model as `causal_attention_msk_model` / "CAIM" (an earlier working name). In the paper, the method is renamed **TEAL**. The two names refer to the same model; only the paper text uses "TEAL".

## Project Structure

```
VLDB2027-CAIM-master/
├── models/                                # TEAL model files
│   ├── causal_attention_msk_model.py      # Main model
│   ├── embedding_layers.py                # Embedding layers
│   ├── mask_generators.py                 # Causal mask generators
│   │
│   ├── # ---- design-choice variants (Sec. V-J) ----
│   ├── causal_attention_msk_model_add_mask_design.py            # Additive mask
│   ├── causal_attention_msk_model_inner_softmax_mask_design.py  # Softmax-internal mask
│   ├── causal_attention_msk_model_parents_predictor_design.py   # Flattening aggregation
│   ├── causal_attention_msk_model_parallel_recon_design.py      # Parallel reconstruction
│   ├── causal_attention_msk_model_direct_parents_design.py      # Direct-parents-only aggregation
│   ├── causal_attention_msk_model_gnn_design.py                 # Implicit GNN message passing
│   │
│   └── # ---- ablation variants (Sec. V-I) ----
│   ├── causal_attention_msk_model_wo_mask_attention.py          # w/o causal mask
│   ├── causal_attention_msk_model_wo_structure_injection.py     # w/o structure injection
│   └── causal_attention_msk_model_wo_pred_weight.py             # w/o learnable weights (mean pooling)
│
├── datasets/                              # PyTorch dataset classes
│   ├── tabular_datasets.py                # Base ReconstructDataset / IndexedReconstructDataset
│   ├── creditcard_dataset.py / cardio_dataset.py / adult_dataset.py
│   ├── diamonds_dataset.py / elevator_dataset.py / housesale_dataset.py
│   ├── crime_dataset.py / meps_dataset.py
│   ├── mslr_dataset.py                    # MSLR-WEB10K (binary classification)
│   └── ood_shift_datasets.py              # Distribution-shift datasets
│
├── utils/                                 # Training / evaluation functions
│   ├── train_msk_utils.py                 # Main training + evaluation
│   ├── train_msk_utils_wo_dag_loss.py     # w/o acyclicity loss
│   ├── train_msk_utils_wo_prediction_reg.py
│   ├── train_msk_utils_wo_reconstruction_loss.py
│   ├── train_msk_utils_wo_sparse_loss.py  # w/o sparsity loss
│   └── split_utils.py                     # Deterministic train/val/test split
│
├── baseline/                              # Baseline methods (16 in total)
│   ├── baseline_datasets.py / train_baseline_utils.py / mslr_loader.py
│   │
│   ├── # machine learning
│   ├── xgboost/      (run_xgboost.py, run_mslr_cls.py)
│   ├── automl/       (AutoGluon/AutoSklearn: autog_*.py, autos_*.py, test_automl_*.py, autog_mslr_cls.py)
│   ├── tabm/         (test_tabm_baseline.py, test_tabm_mslr_cls.py)
│   │
│   ├── # attention-based
│   ├── test_ft_transformer_baseline.py    (+ test_ft_transformer_mslr_cls.py)
│   ├── test_tabtransformer_baseline.py    (+ test_tabtransformer_mslr_cls.py)
│   ├── saint/        (train_with_custom_data.py, data_adapter.py)
│   ├── orion_bix/    (test_orion_bix_baseline.py)
│   │
│   ├── # graph-based
│   ├── tabgnn/       (train.py, preprocess_data.py, models/, experiments/, data/)
│   │
│   ├── # pre-trained
│   ├── tabpfn/       (test_tabpfn.py; TabPFN v3.5)
│   ├── xtab/         (test_xtab.py, test_xtab_mslr_cls.py, checkpoints/)
│   │
│   └── # causal-based
│   ├── castle/       (CASTLE.py, CASTLE_CF.py, main.py, main_cf.py, main_cf_mslr.py, utils.py)
│   ├── test_linear_baseline.py            # LogCause (+ test_linear_mslr_cls.py)
│   ├── llm_cd/       (test_llm_cd_baseline.py, llm_cd_baseline.py, llm_cd_prompts.py)
│   ├── notears/      (export_mask.py)     # NOTEARS → fixed mask + TEAL
│   ├── tane/         (export_mask.py)     # TANE   → fixed mask + TEAL
│   └── hyfd/         (export_mask.py)     # HyFD   → fixed mask + TEAL
│
├── tests/                                 # TEAL experiments
│   ├── test_causal_attention_msk_model.py # Main prediction + design + ablation
│   ├── test_causal_attention_msk_shift.py # Distribution-shift experiments
│   ├── test_mslr_cls.py                   # MSLR-WEB10K (binary classification)
│   └── test_runtime_scaling.py            # Runtime vs. feature-count scaling
│
├── raw_data/                              # Datasets (see Datasets section)
│   ├── creditcard.csv / cardio.csv / adult.csv
│   ├── diamonds.csv / elevator.csv / housesale.csv
│   ├── crime.csv / meps.csv
│   ├── mslr/                              # MSLR-WEB10K (parquet + source ARFF + metadata)
│   ├── diamonds_ood_experiments/          # Pre-generated OOD splits + design docs
│   └── generate_ood_splits.py
│
├── run.sh                                 # One-shot runner for all paper experiments
└── README.md
```

## Requirements

- pytorch=2.2.2 (pytorch-cuda=12.1), python=3.9
- tab-transformer-pytorch==0.4.2 (FT-Transformer / Tab-Transformer baselines)
- tensorflow (CASTLE baseline)
- tabm + rtdl_num_embeddings (TabM baseline)
- orion-bix (Orion-BiX baseline)
- causal-learn (NOTEARS, LogCause, LLM-CD baselines)
- openai (LLM-CD baseline, LLM mode only)
- dgl (TabGNN baseline)
- autogluon (AutoML and XTab baselines)
- tabpfn>=9.0 (TabPFN baseline; uses the v3.5 model)
- desbordante (HyFD and TANE baselines)

## Datasets

All datasets are publicly available on Kaggle, the UCI ML Repository, or OpenML. Data must be processed into CSV (or parquet for MSLR) format and placed in `raw_data/`.

| Dataset | Task | Notes |
|---|---|---|
| CreditCard | Binary classification | None |
| Cardiovascular (Cardio) | Binary classification | height/weight/ap_hi/ap_lo discretized (interval 10) |
| Adult | Binary classification | None |
| MSLR-WEB10K | Binary classification | relevance (0–4) binarized as `relevance > 0`; 136 features |
| Diamonds | Regression | price moved to last column |
| Elevator | Regression | vibration moved to last column |
| Housesale | Regression | 6 CSVs concatenated, shuffled (seed 42) |
| Crime | Regression | 119 continuous + 3 one-hot categorical |
| MEPS | Regression | 4 continuous + 134 one-hot categorical |

### MSLR-WEB10K

- **Source**: [Microsoft Learning to Rank (OpenML 45579)](https://openml.org/d/45579)
- The grouped (query → documents) parquet files are flattened into one row per document by `datasets/mslr_dataset.py` / `baseline/mslr_loader.py`; features are standardized column-wise.
- Data files live in `raw_data/mslr/` (`train.parquet`, `validation.parquet`, `test.parquet`, plus the source `Microsoft.arff` and `openml_45579.json` metadata).
- Per project convention, MSLR-WEB10K is evaluated as a **binary classification** benchmark only (via the `*_mslr_cls.py` scripts).

## Data Preparation

1. **Real-world datasets**: download the 8 CSV datasets (CreditCard, Cardio, Adult, Diamonds, Elevator, Housesale, Crime, MEPS) and place them in `raw_data/`.
2. **MSLR-WEB10K**: place the flattened parquet files (and optionally the source ARFF + metadata) under `raw_data/mslr/`.
3. **Pretrained checkpoints** (offline use):
   - XTab cross-table backbones (`iter_0.ckpt`, `iter_1k.ckpt`, `iter_2k.ckpt`) are in `baseline/xtab/checkpoints/`.
   - TabPFN v3.5 weights are downloaded automatically by the `tabpfn` library on first use (`create_default_for_version(ModelVersion.V3_5)`).
4. **Distribution-shift splits** (Diamonds): `cd raw_data && python generate_ood_splits.py`.

## Training models

* **our model (TEAL):**
  * `cd tests`
  * `python test_causal_attention_msk_model.py --dataset creditcard --d_model 64 --num_heads 4 --num_layers 2 --batch_size 128 --learning_rate 1e-4 --num_epochs 100 --gpu_id 0`

* **baselines:**
  * **XGBoost**: `cd baseline/xgboost && python run_xgboost.py --dataset diamonds --n_folds 5 --random_state 42 --test_size 0.2`
  * **AutoML**: `cd baseline/automl && python test_automl_classification.py --dataset creditcard --time_limit 480` (or `test_automl_regression.py --dataset diamonds --time_limit 480`)
  * **TabM**: `cd baseline/tabm && python test_tabm_baseline.py --dataset creditcard --lr 2e-3 --epochs 30`
  * **TabPFN (v3.5)**: `cd baseline/tabpfn && python test_tabpfn.py --dataset creditcard --max_samples 10000 --gpu_id 0`
  * **XTab**: `cd baseline/xtab && python test_xtab.py --dataset creditcard`
  * **FT-Transformer**: `cd baseline && python test_ft_transformer_baseline.py --dataset creditcard --d_token 64 --n_heads 8 --n_blocks 2 --lr 0.001`
  * **Tab-Transformer**: `cd baseline && python test_tabtransformer_baseline.py --dataset creditcard --d_token 64 --n_heads 8 --n_blocks 6 --lr 0.001`
  * **SAINT**: `cd baseline/saint && python train_with_custom_data.py --dataset creditcard --embedding_size 64 --transformer_depth 4 --attention_heads 4 --lr 0.0001`
  * **Orion-BiX**: `cd baseline/orion_bix && python test_orion_bix_baseline.py --dataset creditcard --n_estimators 32` (regression: `--n_bins 10`)
  * **TabGNN**: `cd baseline/tabgnn && python preprocess_data.py && python train.py --dataset adult --epochs 10 --batch_size 32`
  * **CASTLE**: `cd baseline/castle && python main_cf.py --csv ../../raw_data/creditcard.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension creditcard` (regression: `main.py`)
  * **LogCause**: `cd baseline && python test_linear_baseline.py --dataset creditcard --lr 0.0001`
  * **LLM-CD**: `cd baseline/llm_cd && python test_llm_cd_baseline.py --dataset creditcard --sample_size 200 --predictor rf` (LLM mode: set `LLMCD_API_KEY` / `LLMCD_BASE_URL` / `LLMCD_MODEL` and add `--use_llm`)
  * **NOTEARS → fixed mask + TEAL**: `cd baseline/notears && python export_mask.py --dataset adult --lambda1 0.1`
  * **TANE → fixed mask + TEAL**: `cd baseline/tane && python export_mask.py --dataset adult`
  * **HyFD → fixed mask + TEAL**: `cd baseline/hyfd && python export_mask.py --dataset adult`

## Running Experiments

To run all paper experiments at once:

```bash
bash run.sh
```

The script runs, in order: TEAL main (9 datasets) → XGBoost/AutoML/TabM → FT-Transformer/Tab-Transformer/SAINT/Orion-BiX → TabGNN → TabPFN/XTab → CASTLE/LogCause/LLM-CD → two-stage baselines (NOTEARS/TANE/HyFD) → distribution shift → design choices → ablation → scalability.

## MSLR Large-Scale Benchmark

Binary classification on MSLR-WEB10K (~723k training documents × 136 features):

* **TEAL**: `python tests/test_mslr_cls.py --max_samples 100000 --gpu_id 0`
* **XGBoost**: `cd baseline/xgboost && python run_mslr_cls.py --max_samples 0`
* **AutoGluon**: `cd baseline/automl && python autog_mslr_cls.py --max_samples 100000 --time_limit 600`
* **TabM**: `cd baseline/tabm && python test_tabm_mslr_cls.py --max_samples 100000`
* **FT-Transformer**: `cd baseline && python test_ft_transformer_mslr_cls.py --max_samples 100000`
* **Tab-Transformer**: `cd baseline && python test_tabtransformer_mslr_cls.py --max_samples 100000`
* **LogCause**: `cd baseline && python test_linear_mslr_cls.py --max_samples 100000`
* **XTab**: `cd baseline/xtab && python test_xtab_mslr_cls.py --max_samples 100000`
* **SAINT**: `cd baseline/saint && python train_with_custom_data.py --dataset mslr --max_samples 100000`
* **CASTLE**: `cd baseline/castle && python main_cf_mslr.py --max_samples 10000`

> Per the paper (Table I), TabPFN, Orion-BiX, TabGNN and LLM-CD do not scale to MSLR-WEB10K and are reported as "--".

## Two-Stage Baselines (Sec. V-F)

Dependency graphs discovered offline (NOTEARS / TANE / HyFD) are frozen as a static structural prior fed into TEAL's predictor:

```bash
cd baseline/notears && python export_mask.py --dataset adult --lambda1 0.1
cd baseline/tane    && python export_mask.py --dataset adult
cd baseline/hyfd    && python export_mask.py --dataset adult
```

These offline methods are run on the four low-dimensional datasets where they complete (adult, cardio, creditcard, diamonds).

## Distribution Shift Experiments (Sec. V-H)

OOD robustness on Diamonds, with two settings (feature-level shift and Simpson's Paradox):

```bash
cd raw_data && python generate_ood_splits.py && cd ..
cd tests
python test_causal_attention_msk_shift.py --dataset shift1   # feature-level (Color)
python test_causal_attention_msk_shift.py --dataset shift2   # Simpson's Paradox
python test_causal_attention_msk_shift.py --dataset shift2 --model_source models.causal_attention_msk_model_wo_mask_attention
```

## Design Choices (Sec. V-J)

On Adult and Diamonds (substitute `--dataset`/`--prefix` with `adult` or `diamonds`):

* **Additive mask**: `--model_source models.causal_attention_msk_model_add_mask_design`
* **Softmax mask**: `--model_source models.causal_attention_msk_model_inner_softmax_mask_design`
* **Flattening**: `--model_source models.causal_attention_msk_model_parents_predictor_design`
* **Direct parents only**: `--model_source models.causal_attention_msk_model_direct_parents_design`
* **GNN predictor**: `--model_source models.causal_attention_msk_model_gnn_design`
* **Mean pooling**: `--model_source models.causal_attention_msk_model_wo_pred_weight`
* **Parallel reconstruction**: `--model_source models.causal_attention_msk_model_parallel_recon_design`

Example:

```bash
python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult \
    --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 \
    --model_source models.causal_attention_msk_model_add_mask_design
```

## Ablation Study (Sec. V-I)

Seven ablation variants on Adult and Diamonds:

* **w/o pred reg**: `--model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg`
* **w/o acyclicity loss**: `--model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss`
* **w/o sparsity loss**: `--model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss`
* **w/o reconstruction loss**: `--model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss`
* **w/o pred weight**: `--model_source models.causal_attention_msk_model_wo_pred_weight`
* **w/o causal mask**: `--model_source models.causal_attention_msk_model_wo_mask_attention`
* **w/o structure injection**: `--model_source models.causal_attention_msk_model_wo_structure_injection`

## Scalability (Sec. V-L)

Runtime vs. feature-count scaling on CreditCard, Crime, MEPS, and MSLR-WEB10K:

```bash
python tests/test_runtime_scaling.py --dataset meps --num_samples 5000 --num_epochs 5 --gpu_id 0
```
