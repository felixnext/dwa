#!/bin/bash
# Script that will execute all related work approaches for comparison

# Hard Attention to the task (use default parameters used in paper - same code base) + run with curriculum
python3 run.py --name="base" --experiment=mixture --approach=hat --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="curric" --experiment=mixture --approach=hat --weight_init="kaiming:xavier" --curriculum="linear:100:0.2" --seed=0
# Elastic Weight Consolidation (try different lambda params)
python3 run.py --name="lam500" --experiment=mixture --approach=ewc --weight_init="kaiming:xavier" --lambda=500 --seed=0
python3 run.py --name="lam5000" --experiment=mixture --approach=ewc --weight_init="kaiming:xavier" --lambda=5000 --seed=0
python3 run.py --name="lam50000" --experiment=mixture --approach=ewc --weight_init="kaiming:xavier" --lambda=50000 --seed=0
python3 run.py --name="curric" --experiment=mixture --approach=ewc --weight_init="kaiming:xavier" --lambda=5000 --curriculum="linear:100:0.2" --seed=0
# SGD
python3 run.py --name="base" --experiment=mixture --approach=sgd --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="curric" --experiment=mixture --approach=sgd --weight_init="kaiming:xavier" --curriculum="linear:100:0.2" --seed=0
python3 run.py --name="base" --experiment=mixture --approach="sgd-frozen" --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="curric" --experiment=mixture --approach="sgd-frozen" --weight_init="kaiming:xavier" --curriculum="linear:100:0.2" --seed=0
python3 run.py --name="base" --experiment=mixture --approach="sgd-restart" --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="curric" --experiment=mixture --approach="sgd-restart" --weight_init="kaiming:xavier" --curriculum="linear:100:0.2" --seed=0
# Further Approaches (with default parameters)
python3 run.py --name="base" --experiment=mixture --approach=lwf --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach=lfl --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach="imm-mean" --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach="imm-mode" --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach=progressive --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach=pathnet --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach=joint --weight_init="kaiming:xavier" --seed=0
python3 run.py --name="base" --experiment=mixture --approach=random --weight_init="kaiming:xavier" --seed=0