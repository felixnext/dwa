#!/bin/bash
# Script that will execute all related work approaches for comparison

# TODO: add experiments her
python3 run.py --experiment=mixture --approach=hat --parameter=$2,$3,$4_facescrub.pkz,1 --seed=0
python3 run.py --experiment=mixture --approach=ewc --lambda=500 --seed=0
python3 run.py --experiment=mixture --approach=ewc --lambda=5000 --seed=0
python3 run.py --experiment=mixture --approach=ewc --lambda=50000 --seed=0