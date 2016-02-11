#!/bin/bash

results_table="uprop_paper"
results_db="amosca02"
results_host="gpuvm1"

for random_seed in `cat random_seeds`; do
    echo $random_seed
    ./mlp.py $1 --seed=$random_seed --results-table=$results_table --results-db=$results_db --results-host=$results_host
done
