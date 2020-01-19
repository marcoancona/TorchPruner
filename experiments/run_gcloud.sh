export PYTHONPATH="/home/anconam/Shapley_Pruning"
#export CUDA_VISIBLE_DEVICES="0,1"

# Full model
for i in {1..1}; do python3 train.py --experiment cifar10 --epochs 150 --save --sparsity 1.0  --pruning-steps 0 --pruning-start 0 --pruning-logic random; done


for i in {1..1}; do python3 train.py --experiment cifar10 --epochs 150 --sparsity 0 --max-loss-gap 1 --pruning-steps 50 --pruning-start 1 --pruning-logic sv-abs-smallest; done
for i in {1..1}; do python3 train.py --experiment cifar10 --epochs 150 --sparsity 0 --max-loss-gap 1 --pruning-steps 50 --pruning-start 1 --pruning-logic count-smallest; done
for i in {1..1}; do python3 train.py --experiment cifar10 --epochs 150 --sparsity 0 --max-loss-gap 1 --pruning-steps 50 --pruning-start 1 --pruning-logic grad-abs-smallest; done
for i in {1..1}; do python3 train.py --experiment cifar10 --epochs 150 --sparsity 0 --max-loss-gap 1 --pruning-steps 50 --pruning-start 1 --pruning-logic taylor-abs-smallest; done
for i in {1..1}; do python3 train.py --experiment cifar10 --epochs 150 --sparsity 0 --max-loss-gap 1 --pruning-steps 50 --pruning-start 1 --pruning-logic random; done

