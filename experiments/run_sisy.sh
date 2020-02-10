export PYTHONPATH="/home/anconam/Shapley_Pruning"
export LD_LIBRARY_PATH="/home/anconam/cudnn/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="0"

# Test one-shot pruning, on different sparsity ratios, after full training

# Activations test: first train a model, then load it and run test
#for i in {1..3}; do
#  python3 train.py --experiment fmnist --save --seed $i --sparsity 1 --epochs 50 --pruning-steps 0 --pruning-logic random;
#done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --seed 1 --activations-test --sparsity 1 --epochs 0 --pruning-steps 0 --pruning-logic random; done

# Given the pretrained model we saved above, we try a single shot pruning, with finetuning for 5 epochs
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.0 --max-loss-gap 1 --epochs 5 --pruning-start 0 --pruning-steps 10 --pruning-logic sv-loss-abs; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic sv-loss-abs; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic random; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic weight; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic count; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic grad; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic taylor; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic taylor-abs; done
#for i in {1..1}; do python3 train.py --experiment fmnist --load last --sparsity 0.1 --epochs 5 --pruning-start 0 --pruning-steps 1 --pruning-logic occlusion; done


#for i in {1..3};
#do
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic sv-loss;
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic sv-loss-abs;
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic random;
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic weight;
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic count;
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic grad;
#  python3 train.py --experiment fmnist --seed $i --sparsity 0 --max-loss-gap 1 --epochs 50 --pruning-start 1 --pruning-steps 25 --pruning-logic taylor-abs;
#done

########## CIFAR
for i in {1..3}; do
  python3 train.py --experiment cifar10 --save --seed $i --sparsity 1 --epochs 160 --pruning-steps 0 --pruning-logic random;
done

#for i in {1..1}; do python3 train.py --experiment cifar10 --load last --seed 1 --activations-test --sparsity 1 --epochs 0 --pruning-steps 0 --pruning-logic random; done

for i in {1..3};
do
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic sv-loss-abs;
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic sv-loss;
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic random;
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic weight;
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic count;
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic grad;
  python3 train.py --experiment cifar10 --seed $i --sparsity 0 --max-loss-gap 1 --epochs 160 --pruning-start 1 --pruning-steps 80 --pruning-logic taylor-abs;
done

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
#for i in {1..1}; do
#  python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 1 --pruning-start 0 --pruning-logic random;
#done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 1 --pruning-start 0 --pruning-logic sv-abs-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 0  --pruning-logic grad-abs-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 0  --pruning-logic taylor-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 0  --pruning-logic weight-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 0  --pruning-logic count-smallest; done
#
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 1 --pruning-logic sv-abs-smallest; done
#fpython3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 1 --pruning-logic count-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 1  --pruning-logic grad-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 1  --pruning-logic taylor-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 1  --pruning-logic weight-smallest; done
#python3 train.py --experiment fmnist --sparsity 0 --max-loss-gap 1 --pruning-steps 25 --pruning-start 1  --pruning-logic random; done

# -- > Run full training, saving best model
#for i in {1..2}; do python3 train.py --experiment fmnist --save --sparsity 1 --epochs 50 --pruning-steps 0 --pruning-logic random; done

# Dynamic pruning with max-loss-gap
#for i in {1..2}; do python3 train.py --experiment fmnist  --sparsity 0 --max-loss-gap 1 --pruning-steps 40 --pruning-start 1 --pruning-logic sv-acc-l0; done
#for i in {1..2}; do python3 train.py --experiment fmnist  --sparsity 0 --max-loss-gap 1 --pruning-steps 20 --pruning-start 1 --pruning-logic sv-acc-l0; done
#for i in {1..2}; do python3 train.py --experiment fmnist  --sparsity 0 --max-loss-gap 1 --pruning-steps 40 --pruning-start 1 --pruning-logic sv-loss-l0; done
#for i in {1..2}; do python3 train.py --experiment fmnist  --sparsity 0 --max-loss-gap 1 --pruning-steps 20 --pruning-start 1 --pruning-logic sv-loss-l0; done


# One shot pruning
#for i in {1..1}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 1 --pruning-start 0 --pruning-logic random; done
#for i in {1..1}; do python3 train.py --experiment fmnist --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic sv-smallest; done
#for i in {1..1}; do python3 train.py --experiment fmnist --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic sv-abs-smallest; done
#for i in {1..1}; do python3 train.py --experiment fmnist --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic grad-abs-smallest; done
#for i in {1..1}; do python3 train.py --experiment fmnist --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic count-smallest; done
#for i in {1..1}; do python3 train.py --experiment fmnist --sparsity 0.1 --pruning-steps 1 --pruning-start 0 --pruning-logic weight-smallest; done
##
# Fixed pruning ratio
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 40 --pruning-start 1 --pruning-logic grad; done
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 40 --pruning-start 1 --pruning-logic count; done
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 40 --pruning-start 1 --pruning-logic taylor; done
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 40 --pruning-start 1 --pruning-logic random; done
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 40 --pruning-start 1 --pruning-logic sv-acc; done
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.01 --pruning-steps 40 --pruning-start 1 --pruning-logic sv-loss; done
#for i in {1..3}; do python3 train.py --experiment fmnist --sparsity 0.005 --pruning-steps  --pruning-start 1 --pruning-logic random; done
#for i in {1..2}; do python3 train.py --experiment fmnist --sparsity 0.002 --pruning-steps 40 --pruning-start 1 --pruning-logic random; done
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


