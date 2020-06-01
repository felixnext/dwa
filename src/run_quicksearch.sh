#!/bin/bash


python3 run.py --name="base" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --seed=0 --log_path="../res"
python3 run.py --name="long_warm" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --warmup=[15,750] --seed=0 --log_path="../res"
python3 run.py --name="no_warm" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --warmup=[0,750] --seed=0 --log_path="../res"
python3 run.py --name="no_triplet" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[0,100] --seed=0 --log_path="../res"
python3 run.py --name="tanh" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --gate=tanh --seed=0 --log_path="../res"
#python3 run.py --name="att_scaled" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --scale_att_loss=true --seed=0 --log_path="../res"
#python3 run.py --name="reg_01" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_reg=0.1 --seed=0 --log_path="../res"
python3 run.py --name="reg_09" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_reg=0.9 --seed=0 --log_path="../res"
python3 run.py --name="att_50" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[1,50] --seed=0 --log_path="../res"

python3 run.py --name="att_200" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[1,200] --seed=0 --log_path="../res"
python3 run.py --name="task_loss" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --use_task_loss=true --seed=0 --log_path="../res"
python3 run.py --name="proc6-24" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --processor_feats=[6,24] --seed=0 --log_path="../res"
python3 run.py --name="dropouts" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --use_dropout=true --seed=0 --log_path="../res"
python3 run.py --name="noproc" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --use_processor=false --seed=0 --log_path="../res"
python3 run.py --name="firstanchor" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --use_anchor_first=true --seed=0 --log_path="../res"
#python3 run.py --name="scaleloss" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --scale_att_loss=true --seed=0 --log_path="../res"
#python3 run.py --name="reg05" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_reg=0.5 --seed=0 --log_path="../res"
python3 run.py --name="reg1" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_reg=1 --seed=0 --log_path="../res"

python3 run.py --name="reg5" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_reg=5 --seed=0 --log_path="../res"
python3 run.py --name="reg500" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_reg=500 --seed=0 --log_path="../res"
python3 run.py --name="lam1-01" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[1,0.1] --seed=0 --log_path="../res"
python3 run.py --name="lam01-001" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[0.1,0.01] --seed=0 --log_path="../res"
python3 run.py --name="lam10-001" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[10,0.01] --seed=0 --log_path="../res"
#python3 run.py --name="lam1" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=1 --seed=0 --log_path="../res"
#python3 run.py --name="lam10-1" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[10,1] --seed=0 --log_path="../res"
python3 run.py --name="lam100-01" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=20 --lamb_loss=[100,0.1] --seed=0 --log_path="../res"

# final items
python3 run.py --name="final-base" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=200 --delta=1.0 --alpha=1.0 --gate=tanh --lamb_reg=500 --lamb_loss=[20,0.05] --seed=0 --log_path="../res"
python3 run.py --name="final-warm" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=200 --delta=1.0 --alpha=1.5 --gate=tanh --lamb_reg=500 --lamb_loss=[20,0.05] --warmup=[20,1000] --seed=0 --log_path="../res"
python3 run.py --name="final-stif" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=200 --delta=1.0 --alpha=1.0 --gate=tanh --lamb_reg=500 --lamb_loss=[20,0.05] --stiff=200 --seed=0 --log_path="../res"

# test mode
python3 run.py --name="final-base" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --nepochs=2 --test_mode=4 --delta=1.0 --alpha=1.0 --gate=tanh --lamb_reg=500 --lamb_loss=[20,0.05] --seed=0 --log_path="../res"