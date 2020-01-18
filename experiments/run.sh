#export PYTHONPATH="${PYTHONPATH}:/home/anconam/projects/ExplainRobustness"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Test one-shot pruning, on different sparsity ratios, after full training
# -- > Run full training, saving best model
#for i in {1..1}; do python3 train.py --experiment fmnist --save --sparsity 1 --epochs 50 --pruning-steps 0 --pruning-logic random; done

## -- > Load models, prune and fine tune
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.3 --pruning-steps 1 --pruning-start 0 --pruning-logic sv-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.3 --pruning-steps 1 --pruning-start 0 --pruning-logic grad-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.3 --pruning-steps 1 --pruning-start 0 --pruning-logic count-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.3 --pruning-steps 1 --pruning-start 0 --pruning-logic random; done
#
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.2 --pruning-steps 1 --pruning-start 0 --pruning-logic sv-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.2 --pruning-steps 1 --pruning-start 0 --pruning-logic grad-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.2 --pruning-steps 1 --pruning-start 0 --pruning-logic count-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.2 --pruning-steps 1 --pruning-start 0 --pruning-logic random; done
#
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic sv-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic grad-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic count-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic random; done



# Dynamic pruning with max-loss-gap
for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1 --pruning-logic sv-abs-smallest; done
for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1 --pruning-logic count-smallest; done
for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1  --pruning-logic grad-abs-smallest; done
for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1  --pruning-logic random; done

# -- > Run full training, saving best model
for i in {1..1}; do python3 train.py --experiment fmnist --save --sparsity 1 --epochs 50 --pruning-steps 0 --pruning-logic random; done

# Dynamic pruning with max-loss-gap
for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1 --pruning-logic sv-abs-smallest; done
for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1 --pruning-logic count-smallest; done
for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1  --pruning-logic grad-abs-smallest; done
for i in {1..3}; do python3 train.py --experiment fmnist --load last --sparsity 0 --max-loss-gap 1 --pruning-steps 10 --pruning-start 1  --pruning-logic random; done

# One shot pruning
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.3 --pruning-steps 1 --pruning-start 25 --pruning-logic sv-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.3 --pruning-steps 1 --pruning-start 25 --pruning-logic grad-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.3 --pruning-steps 1 --pruning-start 25 --pruning-logic count-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.3 --pruning-steps 1 --pruning-start 25 --pruning-logic random; done
#
# Fixed pruning ratio
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.20 --pruning-steps 8 --pruning-start 2 --pruning-logic grad-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.20 --pruning-steps 8 --pruning-start 2 --pruning-logic count-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.20 --pruning-steps 8 --pruning-start 2 --pruning-logic sv-abs-smallest; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.20 --pruning-steps 8 --pruning-start 2 --pruning-logic random; done
#
## Non pruning
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 1 --pruning-steps 0 --pruning-logic random; done

#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 30 --pruning-logic random; done
#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 30 --pruning-logic activation_count; done
#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 30 --pruning-logic abs_smallest; done


#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.50 --pruning-steps 20 --pruning-logic sv; done
#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 5 --pruning-logic sv; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity -1 --pruning-steps 1 --pruning-logic sv; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity -1 --pruning-steps 1 --pruning-logic sv; done
#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 1 --pruning-logic random; done
#for i in {1..2}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 5 --pruning-logic random; done
#
#
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 1 --pruning-logic random; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 1 --pruning-logic activation_count; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 1 --pruning-logic abs_smallest; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 1 --pruning-logic largest; done
#
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 10 --pruning-logic random; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 10 --pruning-logic activation_count; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 10 --pruning-logic abs_smallest; done
#for i in {1..3}; do python3 fmnist_train.py --sparsity 0.15 --pruning-steps 10 --pruning-logic largest; done



