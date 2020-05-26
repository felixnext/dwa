#!/bin/bash

python3 run.py --name="base" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --seed=0 --log_path="../res"
python3 run.py --name="long_warm" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --warmup=[15,750] --seed=0 --log_path="../res"
python3 run.py --name="no_warm" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --warmup=[0,750] --seed=0 --log_path="../res"
python3 run.py --name="high_sparsity" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --sparsity=0.5 --seed=0 --log_path="../res"