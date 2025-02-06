from visualise import generate_colormap
from rulemask import find_support
from functools import reduce
import pickle 
from bseval.utils import load_dataset_to_explain
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from graphrep.drawer import atoms
import networkx as nx
import matplotlib.pyplot as plt
from ExplanationEvaluation.datasets.ground_truth_loaders import *
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from patternmod.inside_utils import read_patterns
from patternmod.diffversify_utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import matplotlib
import pandas as pd
from bseval.utils import load_dataset_to_explain, calculate_edge_mask, h_fidelity
from tqdm import tqdm
from math import log
import torch

def generate_node_values(feature, edge_index, model, shap_values, rule_dict, ii):
    embeddings = model.embeddings(feature, edge_index)
    target_class = torch.argmax(model(feature, edge_index)[0])
    shap_values = shap_values[target_class][0]
    node_values = torch.zeros(feature.shape[0], device="cuda:0")
    i = ii-1
    layer, components, label = pattern['layer'], pattern['components'], pattern['label']
    value = shap_values[ii].detach().clone().item()
    activated_nodes = find_support(embeddings[layer] if layer < len(embeddings) else feature, components)
    if not activated_nodes.shape[0]:
        return node_values
    value/= len(activated_nodes)
    node_values[activated_nodes.cpu()] += value#/((1+layer) if layer != len(embeddings) else 1)
    return node_values

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    args = parser.parse_args()


	dataset_name = args.dataset_name
	with open(f"rule_masks_{dataset_name}_{method}_node_simple_devide.pkl", "rb") as file:
		rule_masks = pickle.load(file)

	with open(f"/home/ata/shap_inside/{dataset_name} {method} shap explanations gcn.pkl", "rb") as file:
		    shap_dict = pickle.load(file)
		#with open(f"indices_to_ignore{dataset}.pkl", "rb") as file:
		#    indices_to_ignore = pickle.load(file)
	"""rule_path = f"/home/ata/shap_inside/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation_encode_motifs.csv"
	rules = pd.read_csv(rule_path, delimiter="=")
	rule_dict = {}
	for i, row in rules.iterrows():
		pattern = row[1].strip().split()
		layer_index = int(pattern[0][2:pattern[0].index('c')])
		components = torch.tensor(list(map(lambda x: int(x[x.index('c') + 2:]), pattern)))
		info = row[0].split(":")
		positive_cover = int(info[2][:info[2].index('c')].strip())
		negative_cover = int(info[3][:info[3].index('s')].strip())
		class_index = int(info[1][0])
		rule_dict[i] = {'layer': layer_index, 'components': components, 'label': class_index,
		                'covers': (negative_cover, positive_cover)}"""
	rule_dict = read_patterns(dataset_name) if args.method == "inside" else convert_patterns_to_dict(load_diffversify_patterns(dataset_name))
	


	rule_masks = {i: {"INSIDESHAP": rule_masks[i]} for i in range(len(rule_masks))}
	train_loader, gnn_model, graphs, features, labels, train_mask = load_dataset_to_explain(dataset_name)
	for i, data in tqdm(enumerate(train_loader)):
		flag = False
		if i not in  [730, 260]:
		    continue
		node_values = {}
		feature, edge_index = data.x, data.edge_index
		shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
		for ii, pattern in rule_dict.items():
		    node_values[ii] = list(map(lambda x: x.item(), generate_node_values(feature, edge_index, gnn_model, shap_values, rule_dict, ii)))
		all_values =sorted(list(set(reduce(lambda x, y: x+ y, node_values.values())))) 
		number_of_colors = len(all_values)
		print(number_of_colors)
		color_map = ListedColormap(generate_colormap(number_of_colors))
		for ii, node_value in node_values.items():
		    colors = list(map(lambda x: all_values.index(x), node_value))
		    data.colors = colors
		    nx_graph = to_networkx(data, node_attrs=["x", "colors"])
		    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
		    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
		    if not flag:
		        pos = nx.spring_layout(nx_graph)
		        flag = True

		    plt.clf()
		    colors = {j: color for j, color in nx_graph.nodes(data="colors")}
		    colors = [colors[j] for j in sorted(colors.keys())]
		    labels = {node: atoms[dataset_name][torch.argmax(torch.tensor(node_data)).item()] for node, node_data in nx_graph.nodes(data='x')}
		    nx.draw(nx_graph, pos=pos, labels=labels, with_labels=True, node_color=colors, cmap = color_map, vmin=0, vmax=number_of_colors)
		    plt.title(f"Pattern: {ii}")
		    plt.savefig(f"bseval/figs/{dataset_name}/shap_{i}_{ii}.png")
