#!/bin/bash --login

#SBATCH --time=05:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=4
#SBATCH --mem=80G   # memory per CPU core
#SBATCH -J "sim_graph"   # job name
#SBATCH --mail-user=isaacwilliam4@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Execute Python script with command-line arguments

mamba activate informer

if [ "$1" == "simulation" ]; then

    if [ "$#" -lt 4 ]; then
        echo "Not enough args, required for simulation: number of nodes, timesteps, true or false for line graph generation"
        exit 1
    fi

    line_graph=$4

    if [ "$line_graph" == "true" ]; then
        echo 'Running training with line graph partitions'

        if [ -f ./data/lg_n$2_t$3.csv ]; then
            echo "Simulated graph file exists, skipping file generation..."
        else
            echo "Simulated graph file doesn't exist, generating file..."
            python ./line_graph.py --line_graph --num_nodes $2 --timesteps $3 --alpha .4 --random_seed 42
        fi
        # Calculate enc_in and c_out
        enc_in=$(( $2 * $2 + 3 * $2 * $2 ))
        c_out=$(( $2 * $2 ))

        # Load modules, insert code, and run your programs here
        python -u ./main_informer.py --model informer --target 'none' --data 'sim_graph' --m_true_len $c_out --data_path lg_n$2_t$3.csv --root_path "./data/" --features M --freq d --enc_in $enc_in --dec_in $enc_in --c_out $c_out --num_workers 0 --des lg_n$2_t$3_test --use_multi_gpu

    elif [ "$line_graph" == "false" ]; then
        echo 'Running training with original data'

        if [ -f ./data/g_n$2_t$3.csv ]; then
            echo "Simulated graph file exists, skipping file generation..."
        else
            echo "Simulated graph file doesn't exist, generating file..."
            python ./line_graph.py --num_nodes $2 --timesteps $3 --alpha .4 --random_seed 42
        fi
        # Calculate enc_in and c_out
        enc_in=$(( $2 * $2 ))
        c_out=$(( $2 * $2 ))

        # Load modules, insert code, and run your programs here
        python -u ./main_informer.py --model informer --target 'none' --data 'sim_graph' --m_true_len $c_out --data_path g_n$2_t$3.csv --root_path "./data/" --features M --freq d --enc_in $enc_in --dec_in $enc_in --c_out $c_out --num_workers 0 --des g_n$2_t$3_test --use_multi_gpu

    else

        echo "third argument should be 'true' or 'false'"
        exit 1

    fi

elif [ "$1" == "custom" ]; then
    echo "Preparing custom data... should provide data path of file, name of cleaned file, and then true or false for line graph partitioning"

    if [ "$#" -lt 4 ]; then
        echo "Not enough args, required for custom data: data path, cleaned file name, true or false for line graph partitioning"
        exit 1
    fi

    data_path="$2"
    cleaned_file_name="$3"
    line_graph="$4"

    if [ "$line_graph" == "true" ]; then
        echo "Running custom data with line graph partitioning"

        num_edges=$(python ./line_graph.py --line_graph --data_path "$data_path" --name "$cleaned_file_name")

        tot_in=$((4 * $num_edges))

        python -u ./main_informer.py --model informer --target 'none' --data 'sim_graph' --m_true_len $num_edges --data_path "$cleaned_file_name.csv" --root_path "./data/" --features M --freq d --enc_in $tot_in --dec_in $tot_in --c_out $num_edges --num_workers 0 --des "$cleaned_file_name" --use_multi_gpu

    elif [ "$line_graph" == "false" ]; then
        echo "Running custom data without line graph partitioning"

        num_edges=$(python ./line_graph.py --data_path "$data_path" --name "$cleaned_file_name")

        python -u ./main_informer.py --model informer --target 'none' --data 'sim_graph' --m_true_len $num_edges --data_path "$cleaned_file_name.csv" --root_path "./data/" --features M --freq d --enc_in $num_edges --dec_in $num_edges --c_out $num_edges --num_workers 0 --des "$cleaned_file_name" --use_multi_gpu

    else
        echo "Fourth argument should be 'true' or 'false'"
        exit 1
    fi

else
    echo "First argument must be either custom or simulation, run without further arguments for more information"
    exit 1
fi
