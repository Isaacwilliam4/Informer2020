#!/bin/bash
#SBATCH -N1 -n1 --mem-per-cpu=100M -t00:05:00 --qos=test
source activate informer
python -u ../main_informer.py --model informer --target 'none' --data 'custom' --data_path 'sim_graph.csv' --root_path "./data/" --features M --freq d --enc_in 90 --dec_in 90 --c_out 90 --num_workers 0 --des 'simulated_graph_informer_test'

