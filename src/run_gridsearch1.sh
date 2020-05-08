#!/bin/bash
# Script that will execute a grid search of a part of the parameter space

# Baseline
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --seed=0
# different curriculum approaches
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --curriculum="linear:100:0.2" --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --curriculum="log:100:0.2" --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --curriculum="exp:100:0.2" --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --curriculum="linear:200:0.1" --seed=0
# different loss weightings
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_loss=0.1 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_loss=1 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_loss=10 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_loss=100 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_reg=0.5 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_reg=1 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_reg=5 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_reg=500 --seed=0
python3 run.py --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lamb_reg=5000 --seed=0
