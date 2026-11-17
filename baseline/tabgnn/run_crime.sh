#!/bin/bash
# TabGNN GPU training script
# Uses conda environment tabgnn_clean

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabgnn_clean

export LD_LIBRARY_PATH="/usr/local/cuda-12.2/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

# ============================================
# Hyperparameters (modify as needed)
# ============================================
DATASET="crime"
EPOCHS=50
BATCH_SIZE=256
HIDDEN_DIM=128
N_LAYERS=3
DROPOUT=0.25
LR=1e-5
TRAIN_SPLIT=0.8
DEVICE="cuda:1"
# ============================================

CMD="python train.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --lr $LR \
    --train_split $TRAIN_SPLIT \
    --device $DEVICE"

if [ $# -gt 0 ]; then
    CMD="python train.py $@"
fi

echo "=========================================="
echo "TabGNN GPU Training"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Hidden dim: $HIDDEN_DIM"
echo "GNN layers: $N_LAYERS"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

$CMD
