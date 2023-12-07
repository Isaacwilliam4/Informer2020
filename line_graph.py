import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import argparse
import sys

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
  sin_edges = np.random.choice(np.setdiff1d(np.arange(num_edges), exclude), size=int(num_edges*alpha), replace=False)

  ### SET VALUES FOR EACH TIME STEP - requires: n, t, sin_edges ###
  vals = np.zeros(shape=(t, num_edges))
  vals[:,sin_edges] = np.tile(np.arange(t), (len(sin_edges), 1)).T

  a = np.random.uniform(-10,10,len(sin_edges))
  b = np.random.uniform(-10,10,len(sin_edges))

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
                in_out[i][j] = (np.sum(m) - in_in[i][j] - out_out[i][j] - m[i][j])

        in_in_list.append(in_in.reshape(n**2))
        out_out_list.append(out_out.reshape(n**2))
        in_out_list.append(in_out.reshape(n**2))

    return np.concatenate((v_list, in_in_list, out_out_list, in_out_list),axis=1)

    
#hyperparameters: number of nodes, timesteps, alpha - percentage of edges with values, random seed
# Check if the correct number of arguments is provided
parser = argparse.ArgumentParser(description="Script for generating data with line graph concatenations")

# Parse arguments
parser.add_argument('--line_graph', action='store_true', help='Build line graph', default=False)
parser.add_argument('--num_nodes', type=int, help='Number of nodes', required=True)
parser.add_argument('--timesteps', type=int, help='Number of timesteps', required=True)
parser.add_argument('--alpha', type=float, help='Alpha value', required=True)
parser.add_argument('--random_seed', type=int, help='Random seed', required=True)



args = parser.parse_args()

create_lg = args.line_graph
num_nodes = args.num_nodes
timesteps = args.timesteps
alpha = args.alpha
random_seed = args.random_seed

#get values of the generated primal graph shape = (num timesteps, num edges)
print("Generating primal graphs")
start = time.time()
vals = generate_graph_vals(num_nodes, timesteps, alpha, random_seed)
end = time.time()
print("Graphs generated, time:", end-start)

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






