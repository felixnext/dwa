#!/bin/bash
# Script that will execute a grid search of a part of the parameter space

# triplet loss parameters
python3 run.py --name="baselam10" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --seed=0
python3 run.py --name="alpha01" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=0.1 --seed=0
python3 run.py --name="alpha1" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=1 --seed=0
python3 run.py --name="alpha10" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=10 --seed=0
python3 run.py --name="delta05" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --delta=0.5 --seed=0
python3 run.py --name="delta5" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --delta=5 --seed=0
# combined
python3 run.py --name="alpha1delta5" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=1 --delta=5 --seed=0
python3 run.py --name="alpha10delta5" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=10 --delta=5 --seed=0
python3 run.py --name="alpha1delta05" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=1 --delta=0.5 --seed=0
python3 run.py --name="alpha10delta05" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --lambda_loss=10 --alpha=10 --delta=0.5 --seed=0