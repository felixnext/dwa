#!/bin/bash

# iterate over the actual network parameters

# use processor
python3 run.py --name="noproc" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_processor=false --seed=0

# stem layer size
python3 run.py --name="0stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=0 --seed=0
python3 run.py --name="2stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=2 --seed=0
python3 run.py --name="3stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=3 --seed=0
python3 run.py --name="4stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=4 --seed=0

# embedding size
python3 run.py --name="emb10" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --emb_size=10 --seed=0
python3 run.py --name="emb100" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --emb_size=100 --seed=0

# processor size
python3 run.py --name="proc32-128" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --processor_feats=[32,128] --seed=0
python3 run.py --name="proc6-24" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --processor_feats=[6,24] --seed=0

# dropouts
python3 run.py --name="dropouts" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_dropout=true --seed=0

# concat values + stem_layer size
python3 run.py --name="1stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=1 --use_concat=true --seed=0
python3 run.py --name="2stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=2 --use_concat=true --seed=0
python3 run.py --name="3stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=3 --use_concat=true --seed=0
python3 run.py --name="4stem" --experiment=mixture --approach=dwa --weight_init="kaiming:xavier" --use_stem=4 --use_concat=true --seed=0