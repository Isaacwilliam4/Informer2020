#!/bin/bash

#SBATCH --time=05:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=3096M   # memory per CPU core
#SBATCH -J "sim_graph"   # job name
#SBATCH --mail-user=isaacwilliam4@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -u ./main_informer.py --model informer --target 'none' --data 'custom' --data_path 'sim_graph.csv' --root_path "./data/" --features M --freq d --enc_in 90 --dec_in 90 --c_out 90 --num_workers 8 --des 'simulated_graph_informer_test' --use_multi_gpu

