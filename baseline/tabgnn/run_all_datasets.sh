#!/bin/bash
# TabGNN batch runner: creditcard, cardio, adult, diamonds, elevator, housesale, crime, meps
# conda env: tabgnn_clean

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabgnn_clean
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

EPOCHS=50
BATCH_SIZE=256
HIDDEN_DIM=128
N_LAYERS=3
DROPOUT=0.25
LR=1e-4
TRAIN_SPLIT=0.8
DEVICE="${DEVICE:-cuda:0}"

run_one() {
    local ds="$1"
    echo "=========================================="
    echo "TabGNN Running dataset: $ds"
    echo "=========================================="
    python train.py \
        --dataset "$ds" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --n_layers $N_LAYERS \
        --dropout $DROPOUT \
        --lr $LR \
        --train_split $TRAIN_SPLIT \
        --device $DEVICE
    echo ""
}

for ds in creditcard cardio adult diamonds elevator housesale crime meps; do
    run_one "$ds"
done

echo "All datasets completed."
