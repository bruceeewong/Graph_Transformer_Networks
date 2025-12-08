#!/bin/bash
# Train GTN on Freebase dataset
# This script exposes GTN's limitation with many edge types (36 vs typical 4-5)

# Navigate to project root
cd "$(dirname "$0")/.."

# GTN on Freebase (36 edge types)
echo "=========================================="
echo "Training GTN on Freebase (36 edge types)"
echo "=========================================="

python3 main.py \
    --model GTN \
    --dataset Freebase \
    --data_path ./extension \
    --epoch 100 \
    --num_layers 2 \
    --num_channels 2 \
    --node_dim 64 \
    --lr 0.01 \
    --weight_decay 0.001 \
    --runs 5 \
    --output_path ./output/freebase

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Compare with ACM (4 edge types) baseline:"
echo "  python3 main.py --model GTN --dataset ACM --epoch 100 --num_layers 2 --runs 5"
