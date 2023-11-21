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

# Execute Python script with command-line arguments

if [ "$#" -ne 2 ]; then
    echo "Not enough args, required: number of nodes and timesteps"
    exit 1
fi


if [ -f ./data/n$1_t$2.csv ]; then
    echo "Simulated graph file exists, skipping file generation..."
else
    echo "Simulated graph file doesn't exist, generating file..."
    python ./line_graph.py --num_nodes $1 --timesteps $2 --alpha .4 --random_seed 42
fi


# Calculate enc_in and c_out
enc_in=$(( $1 * $1 - $1 + 3 * $1 * $1 ))
c_out=$(( $1 * $1 - $1 ))

echo $enc_in
echo $c_out

# Load modules, insert code, and run your programs here
python -u ./main_informer.py --model informer --target 'none' --data 'custom' --data_path n$1_t$2.csv --root_path "./data/" --features M --freq d --enc_in $enc_in --dec_in $enc_in --c_out $c_out --num_workers 0 --des 'simulated_graph_informer_test'
