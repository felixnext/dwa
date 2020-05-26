#!/bin/bash

python3 run.py --name="att_scaled" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --scale_att_loss=true --seed=0 --log_path="../res"
python3 run.py --name="reg_01" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --lamb_reg=0.1 --seed=0 --log_path="../res"
python3 run.py --name="reg_09" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --lamb_reg=0.9 --seed=0 --log_path="../res"
python3 run.py --name="att_50" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --lamb_loss=[1,50] --seed=0 --log_path="../res"
python3 run.py --name="att_200" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=100 --lamb_loss=[1,200] --seed=0 --log_path="../res"