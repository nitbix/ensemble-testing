#!/bin/bash

results_table="compboost_paper"
results_db="amosca02"
results_host="gpuvm1"

if [ "$2" != "" ]; then
    random_seeds=`tail -n +$2 random_seeds`
else
    random_seeds=`cat random_seeds`
fi
for random_seed in $random_seeds; do
    echo $random_seed
    ./ensemble.py $1 --seed=$random_seed --results-table=$results_table --results-db=$results_db --results-host=$results_host
done
