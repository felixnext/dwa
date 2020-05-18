#!/bin/bash

python3 run.py --name="base" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=50 --seed=0 --log_path="../res"
python3 run.py --name="no_comb" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=50 --use_combination=false --seed=0 --log_path="../res"