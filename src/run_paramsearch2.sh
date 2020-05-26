#!/bin/bash

python3 run.py --name="low_sparsity" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --sparsity=0.05 --seed=0 --log_path="../res"
python3 run.py --name="high_alpha" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --alpha=0.9 --seed=0 --log_path="../res"
python3 run.py --name="low_alpha" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --alpha=0.1 --seed=0 --log_path="../res"
python3 run.py --name="no_triplet" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --lamb_loss=[0,100] --seed=0 --log_path="../res"