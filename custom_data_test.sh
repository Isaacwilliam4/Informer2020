#!/bin/bash

#SBATCH --time=05:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=4
#SBATCH --mem=30G   # memory per CPU core
#SBATCH -J "world_trade_graph"   # job name
#SBATCH --mail-user=isaacwilliam4@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Execute Python script with command-line arguments

$num_vars = $(( 239 * 239 ))

python -u ./main_informer.py --model informer --seq_len 3 --pred_len 1 --target 'none' --data 'custom' --data_path tomato_ts_df.csv --root_path "./data/" --features M --freq d --enc_in $num_vars --dec_in $num_vars --c_out $num_vars --num_workers 0 --des world_trade_orig --use_multi_gpu
