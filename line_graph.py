import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import argparse
import sys
import os

def apply_sine_function(column, a, b):
  return a + b*np.sin(column)

# n     = num of nodes (0 to n-1 inclusive)
# t     = num of timesteps (0 to t inclusive)
# alpha = percentage of edges that have sin functions, the rest are zero edges (not deleted)
# seed  = seed number used for randomization sin functions and their distribution in the graph
def generate_graph_vals(n=10, t=1000, alpha=0.4, seed=123):
  np.random.seed(seed)
  num_edges = int(n**2)
  ### PREP RNG - requires: n, alpha, seed ###
  np.random.seed(seed)

  ### EXCLUDE VALUES: prevents values assigned to self loops
  exclude = np.array([(i*n + i) for i in range(n)])
  edge_ids_no_diag = np.setdiff1d(np.arange(num_edges), exclude)
  sin_edges = np.random.choice(edge_ids_no_diag, size=int(len(edge_ids_no_diag)*alpha), replace=False)

  ### SET VALUES FOR EACH TIME STEP - requires: n, t, sin_edges ###
  vals = np.zeros(shape=(t, num_edges))
  vals[:,sin_edges] = np.tile(np.arange(t), (len(sin_edges), 1)).T

  a = np.random.uniform(0,10,len(sin_edges))
  b = np.random.uniform(0,10,len(sin_edges))

  vals[:,sin_edges] = apply_sine_function(vals[:,sin_edges], a, b)

  return vals

### given the list of vals (num_edges x timesteps), return new data with line graph partitions concatenated
def add_partitions(v_list):

    in_in_list = []
    out_out_list = []
    in_out_list = []
    n = int(np.sqrt(len(v_list[0])))

    for v in v_list:
        #convert the vector back to matrix form
        m = np.reshape(v,(n,n))

        in_in = np.zeros(m.shape)
        out_out = np.zeros(m.shape)
        in_out = np.zeros(m.shape)

        for i in range(n):
            for j in range(n):
                in_in[i][j] = (np.sum(m[:,j]) - m[i][j])
                out_out[i][j] = (np.sum(m[i]) - m[i][j])
                in_out[i][j] = (np.sum(m[j]) + np.sum(m[:,i]))

        in_in_list.append(in_in.reshape(n**2))
        out_out_list.append(out_out.reshape(n**2))
        in_out_list.append(in_out.reshape(n**2))

    return np.concatenate((v_list, in_in_list, out_out_list, in_out_list),axis=1)

    
#hyperparameters: number of nodes, timesteps, alpha - percentage of edges with values, random seed
# Check if the correct number of arguments is provided
parser = argparse.ArgumentParser(description="Script for generating data with line graph concatenations")

# Parse arguments
parser.add_argument('--type', type=str, help='Either graph or custom', default='simulation')
parser.add_argument('--line_graph', action='store_true', help='Concatenate line graph partitions onto data', default=False)
parser.add_argument('--num_nodes', type=int, help='Number of nodes', required=False)
parser.add_argument('--timesteps', type=int, help='Number of timesteps', required=False)
parser.add_argument('--alpha', type=float, help='Alpha value (percent of edges to assign values to)', required=False)
parser.add_argument('--random_seed', type=int, help='Random seed', required=False)
parser.add_argument('--data_path', type=str, help='Path to custom data', required=False)
parser.add_argument('--name', type=str, help='name of custom data', required=False)

args = parser.parse_args()

exp_type = args.type
create_lg = args.line_graph
num_nodes = args.num_nodes
timesteps = args.timesteps
alpha = args.alpha
random_seed = args.random_seed
data_path = args.data_path
name = args.name

if exp_type == "custom":

  if (data_path or name) == None:
    print("When performing experiment on custom data must specify --data_path, and --name, --line_graph optional if you want to create line graph partitions, data must be csv in with shape (timesteps x num_edges)")
    exit(0)

  if (os.path.exists(f'./data/{name}.csv')):
    print("Prepared file exists for custom data")
    exit(0)
    
  else:
    print("Generating prepared file for custom data")

    df = pd.read_csv(data_path)
       
    
    if create_lg:
      lg_data = add_partitions(df.values)
      df = pd.DataFrame(lg_data)

    df += 1
    df = np.log(df)
    df.index.name = 'date'
    num_edges = df.shape[1] 
    df = df.reset_index()
    df.to_csv(f'./data/{name}.csv', index=False, float_format='%.10f')

else:

  if (create_lg or num_nodes or timesteps or alpha or random_seed) == None:
    print("When performing graph experiment must specify --line_graph, --num_nodes, --timesteps, --alpha, --random_seed")
    exit(0)

  #get values of the generated primal graph shape = (num timesteps, num edges)
  print("Generating graph data")
  start = time.time()
  vals = generate_graph_vals(num_nodes, timesteps, alpha, random_seed)

  if create_lg:
    #concatenate the line graph onto the original data
    lg_data = add_partitions(vals)
    line_graph_df = pd.DataFrame(lg_data)
    line_graph_df.index.name = 'date'
    line_graph_df = line_graph_df.reset_index()
    line_graph_df.to_csv(f'./data/lg_n{num_nodes}_t{timesteps}.csv', index=False)
  else:
    primal_df = pd.DataFrame(vals)
    primal_df.index.name = 'date'
    primal_df = primal_df.reset_index()
    primal_df.to_csv(f'./data/g_n{num_nodes}_t{timesteps}.csv', index=False)

  end = time.time()
  print("Graphs generated, time:", end-start)


