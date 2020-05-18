#!/bin/bash

python3 run.py --name="exp_high_att" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=50 --lamb_loss=[1,100] --seed=0 --log_path="../res"
python3 run.py --name="exp_low_att" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=50 --lamb_loss=[1,0.1] --seed=0 --log_path="../res"

python3 run.py --name="exp_high_trip" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=50 --lamb_loss=[100,10] --seed=0 --log_path="../res"
python3 run.py --name="exp_low_trip" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=50 --lamb_loss=[0.1,10] --seed=0 --log_path="../res"