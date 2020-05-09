#!/bin/bash
# Script that will execute a grid search of a part of the parameter space

# Different sparsity options (Note: might be influenced by lamb_reg)
python3 run.py --name="sparse01bin" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --sparsity=0.1 --bin_sparsity=true --seed=0
python3 run.py --name="sparse02bin" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --sparsity=0.2 --bin_sparsity=true --seed=0
python3 run.py --name="sparse05bin" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --sparsity=0.5 --bin_sparsity=true --seed=0
python3 run.py --name="sparse01" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --sparsity=0.1 --bin_sparsity=false --seed=0
python3 run.py --name="sparse02" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --sparsity=0.2 --bin_sparsity=false --seed=0
python3 run.py --name="sparse05" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --sparsity=0.5 --bin_sparsity=false --seed=0
# anchor losses
python3 run.py --name="firstanchor" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_anchor_first=true --seed=0
python3 run.py --name="scaleloss" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --scale_att_loss=true --seed=0
