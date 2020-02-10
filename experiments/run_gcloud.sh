export PYTHONPATH="/home/anconam/Shapley_Pruning"
#export CUDA_VISIBLE_DEVICES="0,1"

# Full model
#python3 train.py --experiment cub200 --save --seed 1 --sparsity 1 --epochs 50 --pruning-steps 0 --pruning-logic random;


python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss-2std#5;
python3 train.py --experiment cub200 --load last --seed 2 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss-2std#5;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss-abs#5;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss#5;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic weight;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic taylor-abs;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic random;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic count;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 2  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic grad;


python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss-2std#5;
python3 train.py --experiment cub200 --load last --seed 2 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss-2std#5;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss-abs#5;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic sv-loss#5;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic weight;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic taylor-abs;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic random;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic count;
python3 train.py --experiment cub200 --load last --seed 1 --sparsity 0 --max-loss-gap 10  --epochs 25 --pruning-steps 14 --pruning-start 1 --pruning-logic grad;