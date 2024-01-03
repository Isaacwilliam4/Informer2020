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
        echo "Args required for graph simulation: number of nodes, timesteps, true or false for line graph generation"
        exit 1
    fi

    echo "Args: Type=$1, NumNodes=$2 TimeSteps=$3, LineGraphPartitioning=$4"


    line_graph=$4

    if [ "$line_graph" == "true" ]; then
        echo 'Running training with line graph partitions'

        if [ -f ./data/lg_n$2_t$3.csv ]; then
            echo "Simulated graph file exists, skipping file generation..."
        else
            echo "Simulated graph file doesn't exist, generating file..."
            python ./line_graph.py --type "simulation" --line_graph --num_nodes $2 --timesteps $3 --alpha .4 --random_seed 42
        fi
        # Calculate enc_in and c_out
        enc_in=$(( 4 * $2 * $2 ))
        c_out=$(( $2 * $2 ))

        # Load modules, insert code, and run your programs here
        python -u ./main_informer.py --model informer --target 'none' --data 'sim_graph' --m_true_len $c_out --data_path lg_n$2_t$3.csv --root_path "./data/" --features M --freq d --enc_in $enc_in --dec_in $enc_in --c_out $c_out --num_workers 0 --des lg_n$2_t$3_test --use_multi_gpu

    elif [ "$line_graph" == "false" ]; then
        echo 'Running training with original data'

        if [ -f ./data/g_n$2_t$3.csv ]; then
            echo "Simulated graph file exists, skipping file generation..."
        else
            echo "Simulated graph file doesn't exist, generating file..."
            python ./line_graph.py --type "simulation" --num_nodes $2 --timesteps $3 --alpha .4 --random_seed 42
        fi
        # Calculate enc_in and c_out
        enc_in=$(( $2 * $2 ))
        c_out=$(( $2 * $2 ))

        # Load modules, insert code, and run your programs here
        python -u ./main_informer.py --pred_len 1 --batch_size 1 --seq_len 6 --label_len 2 --model informer --target 'none' --data 'sim_graph' --m_true_len $c_out --data_path g_n$2_t$3.csv --root_path "./data/" --features M --freq d --enc_in $enc_in --dec_in $enc_in --c_out $c_out --num_workers 0 --des g_n$2_t$3_test --use_multi_gpu

    else

        echo "third argument should be 'true' or 'false'"
        exit 1

    fi

elif [ "$1" == "custom" ]; then
    echo "Preparing custom data... "

    if [ "$#" -lt 4 ]; then
        echo "Args required for custom: data path of file, true or false for line graph partitioning, then number of nodes in graph, prepared file will have name file name + _prepared"
        exit 1
    fi

    line_graph=$3
    num_edges=$(( $4 * $4 ))
    filename="${2##*/}"
    name="${filename%.*}"
    echo "Args: Type=$1, Datapath=$2, LineGraphPartitioning=$3, NumNodes=$4"

    if [ "$line_graph" == "true" ]; then

        ext="_lg_prepared"


        echo "Running custom data with line graph partitioning"

        python ./line_graph.py --type 'custom' --line_graph --data_path "$2" --name "$name$ext"

        tot_in=$(( 4 * $num_edges ))

        if [ -f ./data/$name$ext.csv ]; then

            python -u ./main_informer.py --pred_len 1 --batch_size 1 --seq_len 6 --label_len 2 --model informer --target 'none' --data 'custom' --m_true_len $num_edges --data_path $name$ext.csv --root_path "./data/" --features M --freq d --enc_in $tot_in --dec_in $tot_in --c_out $num_edges --num_workers 0 --des $name --use_multi_gpu
        
        else 

            echo "File generation failed, exiting..."
            exit 1

        fi

    else 

        ext="_prepared"


        echo "Running custom data without line graph partitioning"


        python ./line_graph.py --type 'custom' --data_path "$2" --name "$name$ext"

        if [ -f ./data/$name$ext.csv ]; then

            python -u ./main_informer.py --pred_len 1 --batch_size 1 --seq_len 6 --label_len 2 --model informer --target 'none' --data 'custom' --m_true_len $num_edges --data_path $name$ext.csv --root_path "./data/" --features M --freq d --enc_in $num_edges --dec_in $num_edges --c_out $num_edges --num_workers 0 --des $name --use_multi_gpu
            
        else 

            echo "File generation failed, exiting..."
            exit 1

        fi

    fi

else 

    echo "first argument must be either custom or simulation, run without further arguments for more information"

fi
