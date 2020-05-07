#!/bin/bash
# Script that will execute a grid search of a part of the parameter space

# TODO: add experiments her
python3 run.py --experiment=mixture --approach=dwa --parameter=$2,$3,$4_facescrub.pkz,1 --seed=0