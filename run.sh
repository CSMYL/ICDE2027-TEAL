#!/bin/bash

# TEAL (Structure-Guided Hierarchical Analytics): Experiment Runner
# Runs all experiments reported in the paper. Data must be prepared beforehand.
# Every experiment is repeated with random seeds 42 / 123 / 456 / 789 / 1024
# for mean + std (see the individual scripts for seed handling).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================="
echo "TEAL Experiment Runner"
echo "========================================="

# Dataset groups (paper Sec. V-B: 4 classification + 5 regression)
CLS_DATASETS=("adult" "cardio" "creditcard")
REG_DATASETS=("diamonds" "elevator" "housesale" "crime" "meps")
ALL8=("adult" "cardio" "creditcard" "diamonds" "elevator" "housesale" "crime" "meps")

# ============================================
# 1. TEAL main experiments (9 datasets)
# ============================================
echo "===== Section 1: TEAL main ====="
cd tests
for dataset in "${ALL8[@]}"; do
    python test_causal_attention_msk_model.py --dataset $dataset --d_model 64 --num_heads 4 --num_layers 2 --batch_size 128 --learning_rate 1e-4 --num_epochs 100 --dropout 0.1 --weight_decay 1e-5 --gpu_id 0
done
# MSLR-WEB10K (binary classification)
python test_mslr_cls.py --max_samples 100000 --gpu_id 0
cd ..

# ============================================
# 2. Machine-learning baselines: XGBoost / AutoML / TabM
# ============================================
echo "===== Section 2: XGBoost ====="
cd baseline/xgboost
for dataset in "${ALL8[@]}"; do
    python run_xgboost.py --dataset $dataset --n_folds 5 --random_state 42 --test_size 0.2
done
python run_mslr_cls.py --max_samples 0
cd ../..

echo "===== Section 3: AutoML ====="
cd baseline/automl
for dataset in "${CLS_DATASETS[@]}"; do
    python test_automl_classification.py --dataset $dataset --time_limit 480
done
for dataset in "${REG_DATASETS[@]}"; do
    python test_automl_regression.py --dataset $dataset --time_limit 480
done
python autog_mslr_cls.py --max_samples 100000 --time_limit 600
cd ../..

echo "===== Section 4: TabM ====="
cd baseline/tabm
for dataset in "${ALL8[@]}"; do
    python test_tabm_baseline.py --dataset $dataset --k 32 --n_blocks 3 --d_block 512 --batch_size 256 --lr 2e-3 --epochs 30 --gpu_id 0
done
python test_tabm_mslr_cls.py --max_samples 100000
cd ../..

# ============================================
# 3. Attention-based baselines
# ============================================
echo "===== Section 5: FT-Transformer ====="
cd baseline
for dataset in "${ALL8[@]}"; do
    python test_ft_transformer_baseline.py --dataset $dataset --d_token 64 --n_heads 8 --n_blocks 2 --batch_size 128 --lr 0.001 --epochs 30 --gpu_id 0
done
python test_ft_transformer_mslr_cls.py --max_samples 100000
cd ..

echo "===== Section 6: Tab-Transformer ====="
cd baseline
for dataset in "${ALL8[@]}"; do
    python test_tabtransformer_baseline.py --dataset $dataset --d_token 32 --n_heads 4 --n_blocks 6 --batch_size 128 --lr 0.001 --epochs 30 --gpu_id 0
done
python test_tabtransformer_mslr_cls.py --max_samples 100000
cd ..

echo "===== Section 7: SAINT ====="
cd baseline/saint
for dataset in "${ALL8[@]}"; do
    python train_with_custom_data.py --dataset $dataset --embedding_size 64 --batchsize 128 --transformer_depth 4 --attention_heads 4 --lr 0.0001 --epochs 50
done
python train_with_custom_data.py --dataset mslr --embedding_size 64 --batchsize 128 --transformer_depth 4 --attention_heads 4 --lr 0.0001 --epochs 50 --max_samples 100000
cd ../..

echo "===== Section 8: Orion-BiX ====="
cd baseline/orion_bix
for dataset in "${CLS_DATASETS[@]}"; do
    python test_orion_bix_baseline.py --dataset $dataset --n_estimators 32 --batch_size 8 --gpu_id 0
done
for dataset in "${REG_DATASETS[@]}"; do
    python test_orion_bix_baseline.py --dataset $dataset --n_bins 10 --batch_size 8 --gpu_id 0
done
cd ../..

# ============================================
# 4. Graph-based baseline: TabGNN
# ============================================
echo "===== Section 9: TabGNN ====="
cd baseline/tabgnn
python preprocess_data.py
for dataset in "${ALL8[@]}"; do
    python train.py --dataset $dataset --hidden_dim 64 --n_layers 3 --batch_size 32 --lr 0.001 --epochs 10
done
cd ../..

# ============================================
# 5. Pre-trained baselines: TabPFN / XTab
# ============================================
echo "===== Section 10: TabPFN (v3.5) ====="
cd baseline/tabpfn
for dataset in "${ALL8[@]}"; do
    python test_tabpfn.py --dataset $dataset --max_samples 10000 --gpu_id 0
done
cd ../..

echo "===== Section 11: XTab ====="
cd baseline/xtab
for dataset in "${ALL8[@]}"; do
    python test_xtab.py --dataset $dataset
done
python test_xtab_mslr_cls.py --max_samples 100000
cd ../..

# ============================================
# 6. Causal-based baselines: CASTLE / LogCause / LLM-CD
# ============================================
echo "===== Section 12: CASTLE ====="
cd baseline/castle
for dataset in "${CLS_DATASETS[@]}"; do
    python main_cf.py --csv ../../raw_data/${dataset}.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension $dataset
done
for dataset in "${REG_DATASETS[@]}"; do
    python main.py --csv ../../raw_data/${dataset}.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension $dataset
done
python main_cf_mslr.py --max_samples 10000 --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0
cd ../..

echo "===== Section 13: LogCause ====="
cd baseline
for dataset in "${ALL8[@]}"; do
    python test_linear_baseline.py --dataset $dataset --batch_size 64 --lr 0.00001 --epochs 200
done
python test_linear_mslr_cls.py --max_samples 100000
cd ..

echo "===== Section 14: LLM-CD ====="
cd baseline/llm_cd
for dataset in "${ALL8[@]}"; do
    python test_llm_cd_baseline.py --dataset $dataset --sample_size 200 --predictor rf
done
cd ../..

# ============================================
# 7. Two-stage baselines (paper Sec. V-F)
#    NOTEARS / TANE / HyFD dependency graph -> frozen mask -> TEAL predictor
#    (run on low-dimensional datasets where the offline discovery completes)
# ============================================
echo "===== Section 15: Two-stage baselines ====="
cd baseline/notears
for dataset in adult cardio creditcard diamonds; do
    python export_mask.py --dataset $dataset --lambda1 0.1
done
cd ..

cd baseline/tane
for dataset in adult cardio creditcard diamonds; do
    python export_mask.py --dataset $dataset
done
cd ..

cd baseline/hyfd
for dataset in adult cardio creditcard diamonds; do
    python export_mask.py --dataset $dataset
done
cd ../..

# ============================================
# 8. Distribution-shift experiments (paper Sec. V-H)
# ============================================
echo "===== Section 16: Distribution shift ====="
cd raw_data
python generate_ood_splits.py
cd ..
cd tests
python test_causal_attention_msk_shift.py --dataset shift1
python test_causal_attention_msk_shift.py --dataset shift2
python test_causal_attention_msk_shift.py --dataset shift2 --model_source models.causal_attention_msk_model_wo_mask_attention
cd ..

# ============================================
# 9. Design choices (paper Sec. V-J)
# ============================================
echo "===== Section 17: Design choices ====="
cd tests
DESIGN="--num_heads 4 --num_layers 2 --gpu_id 0 --patience 50 --d_model 64 --learning_rate 0.0001"
for dataset in adult diamonds; do
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_add_mask_design
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_inner_softmax_mask_design
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_parents_predictor_design
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_direct_parents_design
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_gnn_design
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_wo_pred_weight
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $DESIGN --model_source models.causal_attention_msk_model_parallel_recon_design
done
cd ..

# ============================================
# 10. Ablation study (paper Sec. V-I)
# ============================================
echo "===== Section 18: Ablation ====="
cd tests
ABLATION="--num_heads 4 --num_layers 2 --gpu_id 0 --patience 50 --d_model 64 --learning_rate 0.0001"
for dataset in adult diamonds; do
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model_wo_pred_weight
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model_wo_mask_attention
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ABLATION --model_source models.causal_attention_msk_model_wo_structure_injection
done
cd ..

# ============================================
# 11. Scalability (paper Sec. V-L)
# ============================================
echo "===== Section 19: Scalability ====="
cd tests
for dataset in creditcard crime meps mslr; do
    python test_runtime_scaling.py --dataset $dataset --num_samples 5000 --num_epochs 5 --gpu_id 0
done
cd ..

echo ""
echo "========================================="
echo "All experiments completed."
echo "========================================="
