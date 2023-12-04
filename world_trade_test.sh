#!/bin/bash

#SBATCH --time=05:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=3000M   # memory per CPU core
#SBATCH -J "sim_graph"   # job name
#SBATCH --mail-user=isaacwilliam4@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Execute Python script with command-line arguments



python -u ./main_informer.py --model informer --seq_len 3 --pred_len 1 --target 'none' --data 'custom' --data_path clean_tomato.csv --root_path "./data/" --features M --freq d --enc_in 6766 --dec_in 6766 --c_out 6766 --num_workers 0 --des world_trade_orig --use_multi_gpu
