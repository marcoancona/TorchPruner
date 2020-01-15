#export PYTHONPATH="${PYTHONPATH}:/home/anconam/projects/ExplainRobustness"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."




# Dynamic pruning ratio
for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0 --pruning-steps 45 --pruning-logic sv-zeros; done
for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0 --pruning-steps 45 --pruning-logic sv-nonpositive; done

# Fixed pruning ratio
for i in {1..3}; do python3 train.py --sparsity 0.15 --pruning-steps 45 --pruning-logic grad-abs-smallest; done
for i in {1..3}; do python3 train.py --sparsity 0.15 --pruning-steps 45 --pruning-logic count-smallest; done
for i in {1..3}; do python3 train.py --sparsity 0.15 --pruning-steps 45 --pruning-logic sv-abs-smallest; done
for i in {1..3}; do python3 train.py --sparsity 0.15 --pruning-steps 45 --pruning-logic random; done

# Non pruning
for i in {1..2}; do python3 fmnist_train.py --sparsity 1 --pruning-steps 0 --pruning-logic random; done

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



