export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
python3 train.py --experiment fmnist --load last --seed 0 --sparsity 1 --epochs 0 --pruning-steps 0 --pruning-logic random;
