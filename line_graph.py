import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# n     = num of nodes (0 to n-1 inclusive)
# t     = num of timesteps (0 to t inclusive)
# alpha = percentage of edges that have sin functions, the rest are zero edges (not deleted)
# seed  = seed number used for randomization sin functions and their distribution in the graph
def generate_graph_vals(n=10, t=1000, alpha=0.4, seed=123):

  ### PREP RNG - requires: n, alpha, seed ###
  random.seed(seed)
  sin_edges = random.sample(range(int(n*(n-1))), int(n*(n-1)*alpha))

  ### SET VALUES FOR EACH TIME STEP - requires: n, t, sin_edges ###
  vals = np.zeros(shape=(t, int(n*(n-1))))
  for edge_index in range(int(n*(n-1))):
    if edge_index not in sin_edges:
      continue
    a = random.uniform(-10, 10)
    b = random.uniform(-10, 10)
    for time in list(range(0,t)):
      vals[time, edge_index] = a + b * np.sin(time)

  return vals

def setup_graphs(vals):

  ### Recover Node Count ###
  n = int(1/2 + np.sqrt(1 + 4*vals.shape[1])/2)
  if vals.shape[1] != n*(n-1):
    print("The values given do not form a complete graph: this logic has not been implemented.")

  ### SET UP ONE GRAPH FOR EACH TIME STEP - requires: n, t, vals ###
  graphs = []
  for time in range(vals.shape[0]):
    G = nx.complete_graph(n, nx.DiGraph())
    edge_index = 0
    for (start, end) in G.edges:
      G.add_edge(start, end, index=edge_index, weight=vals[time, edge_index])
      edge_index += 1
    graphs.append(G)

  return graphs

def get_ETV_node(primal_G, edge_index):
  edge = list(primal_G.edges)[edge_index] # Unpacking the entire edge list may be slowing it down
  node_data = {'edge': (edge[0], edge[1]), 'weight': primal_G.get_edge_data(edge[0], edge[1])['weight']}
  return (edge_index, node_data)

### a helper function for get_ETV_edges() to reduce code duplication ###

# Does not support self-loops neighboring themselves (would it be an in-out relation only?)
def make_ETV_edge(node, edge_1, edge_2, neighbor_type):

  ### logic for self-loops ###
  if edge_1 == edge_2:
    return {}

  ### Order the edge indexes in-order ###
  # elif edge_1 < edge_2:
  #   edge_name = (edge_1, edge_2)
  # else:
  #   edge_name = (edge_2, edge_1)

  edge_name = (edge_1, edge_2)

  return {edge_name: {'primal_node': node, 'type': neighbor_type}}

### THE DYNAMIC ACCESS OF ETV EDGES GIVEN A PRIMAL NODE ###

def get_ETV_edges(primal_G, node):
  ETV_edges = {}

  ### Each node in the primal becomes an edge in the ETVs according to the direction the edges face eachother ###
  for (start_1, end_1, attr_1) in primal_G.in_edges(node, data=True):
    for (start_2, end_2, attr_2) in primal_G.in_edges(node, data=True):
      ETV_edges |= make_ETV_edge(node, attr_1['index'], attr_2['index'], 'in-in')
    for (start_2, end_2, attr_2) in primal_G.out_edges(node, data=True):
      ETV_edges |= make_ETV_edge(node, attr_1['index'], attr_2['index'], 'in-out')
  for (start_1, end_1, attr_1) in primal_G.out_edges(node, data=True):
    for (start_2, end_2, attr_2) in primal_G.out_edges(node, data=True):
      ETV_edges |= make_ETV_edge(node, attr_1['index'], attr_2['index'], 'out-out')

  return ETV_edges

### THE STATIC MAKE-ENTIRE-ETV GIVEN A PRIMAL GRAPH ###
# uses the dynamic methods to reduce code duplication, but returns the entire ETV graph

def create_ETV_graphs(primal_G_list):

  ETV_list = []

  for primal_G in primal_G_list: # one for each timestep

    ### Create new undirected ETV graphs ##
    ETV_G = nx.MultiGraph()

    ### Each edge in the primal becomes a node in the ETV graphs ###
    for edge_index in range(len(primal_G.edges)):
      node, node_data = get_ETV_node(primal_G, edge_index)
      ETV_G.add_node(node, **node_data)

    ### Each node in the primal becomes an edge in the ETVs according to the direction the edges face eachother ###
    for node in primal_G.nodes:
      ETV_edges = get_ETV_edges(primal_G, node)
      for edge_name in ETV_edges.keys():
        ETV_G.add_edge(*edge_name, **ETV_edges[edge_name])

    ### Save the ETV graphs from this timestep ###
    ETV_list.append(ETV_G)

  return ETV_list


def update_mat(mat, e1, e2, w1, w2, both):
    mat[e1]+= w2
    if both:
        mat[e2] += w1
    return


def get_partitions(primal_g):

    num_nodes = primal_g.number_of_nodes()
    in_in_mat = np.zeros((num_nodes, num_nodes))
    out_out_mat = np.zeros((num_nodes, num_nodes))
    in_out_mat = np.zeros((num_nodes, num_nodes))

    for p_node in primal_g.nodes():
        ETV_edges = get_ETV_edges(primal_g, p_node)

        for key, value in ETV_edges.items():
            id1, node1_val = get_ETV_node(primal_g, key[0])
            id2, node2_val = get_ETV_node(primal_g, key[1])

            edge1 = node1_val['edge']
            edge2 = node2_val['edge']
            w1 = node1_val['weight']
            w2 = node2_val['weight']

            if value['type'] == 'in-out':
                if edge1 == (edge2[1], edge2[0]):
                    update_mat(in_out_mat, edge1, edge2, w1, w2, both = False)
                else:
                    update_mat(in_out_mat, edge1, edge2, w1, w2, both = True)
            
            elif value['type'] == 'in-in':
                update_mat(in_in_mat, edge1, edge2, w1, w2, both = True)

            elif value['type'] == 'out-out':
                update_mat(out_out_mat, edge1, edge2, w1, w2, both = True)


    return in_in_mat, out_out_mat, in_out_mat

def combine_G_LG(vals, LG_list, timesteps):
  r = LG_list.reshape(timesteps,-1)
  return np.concatenate((vals, r), axis=1)



import argparse
import sys
#hyperparameters: number of nodes, timesteps, alpha - percentage of edges with values, random seed
# Check if the correct number of arguments is provided
parser = argparse.ArgumentParser(description="Script for generating data with line graph concatenations")

# Parse arguments
parser.add_argument('--num_nodes', type=int, help='Number of nodes', required=True)
parser.add_argument('--timesteps', type=int, help='Number of timesteps', required=True)
parser.add_argument('--alpha', type=float, help='Alpha value', required=True)
parser.add_argument('--random_seed', type=int, help='Random seed', required=True)



args = parser.parse_args()

num_nodes = args.num_nodes
timesteps = args.timesteps
alpha = args.alpha
random_seed = args.random_seed

vals = generate_graph_vals(num_nodes, timesteps, alpha, random_seed)
G_list = setup_graphs(vals)
LG_list = np.array([np.array(get_partitions(G_list[i])) for i in range(len(G_list))])

master = combine_G_LG(vals, LG_list, timesteps)

np.savetxt(f'./data/{num_nodes}x{num_nodes}_t{timesteps}_train.csv',master, delimiter=',')
np.savetxt(f'./data/{num_nodes}x{num_nodes}_t{timesteps}_labels.csv',vals, delimiter=',')







