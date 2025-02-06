import argparse
import os
import torch 
import networkx as nx
import numpy as np
import pickle
from sklearn.neighbors import KernelDensity
from networkx import optimize_graph_edit_distance
from tqdm import tqdm
from ot_distances import Fused_Gromov_Wasserstein_distance
from graph import Graph
from sklearn.metrics import silhouette_score
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx
from patternmod.inside_utils import read_patterns
from patternmod.pattern_evaluation import calculate_pattern_support
from datasetmod.datasetloader import load_dataset_gnn, load_splited_data
from modelmod.gnn import load_gnn
from modelmod.gnn_utils import fill_embeddings
from shapmod.rulemask import find_support
from torch_geometric.data import Data
from functools import reduce
import matplotlib
from matplotlib.colors import ListedColormap
import math
from shapmod.rulemask import generate_node_values, find_support
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


atoms = {}
atoms['aids'] = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co",
                 11: "Br",
                 12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K",
                 21: "Pd",
                 22: "Au", 23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb",
                 31: "Sn",
                 32: "Ga", 33: "Hg", 34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb"}
atoms['mutag'] = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na",
                  11: "K",
                  12: "Li", 13: "Ca"}
atoms['BBBP'] = ["C", "N", "O", "S", "P", "BR", "B", "F", "CL", "I", "H", "NA", "CA"]

atoms['ba2'] = {i:0 for i in range(10)}
atoms["Benzen"] = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']

atoms["AlkaneCarbonyl"] = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']



def generate_colormap(N, base_color_hue=0.66):
    # N: Number of colors to generate
    # base_color_hue: Choose the hue (between 0 and 1), 0 is red, 0.33 is green, 0.66 is blue, etc.
    # Generate values (brightness levels) from dark to bright
    brightness_levels = np.linspace(0.1, 1, N)  # Start from a darker value for more contrast
    
    # Create an array for HSV colors
    hsv_colors = np.zeros((N, 4))  # Adding an extra dimension for alpha (transparency)
    
    # Set hue to a constant value (base_color_hue)
    hsv_colors[:, 0] = base_color_hue  # Fixed hue for a single color
    
    # Set saturation to 1 for vivid colors
    hsv_colors[:, 1] = 1
    
    # Set value (brightness) to the generated brightness levels
    hsv_colors[:, 2] = brightness_levels
    
    # Set alpha (transparency) to a constant value (e.g., 0.5 for 50% transparency)
    hsv_colors[:, 3] = 0.5
    
    # Convert HSV to RGB
    rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors[:, :3])
    
    # Add the alpha channel to the RGB colors
    rgba_colors = np.concatenate((rgb_colors, hsv_colors[:, 3:4]), axis=1)
    
    return rgba_colors


def ego_network(nx_graph, center_nodes, radius, verbose=False, pattern_id=10, cluster_index=0):
    isolated_nodes = [node for node in nx_graph.nodes() if nx_graph.degree(node) == 0]
    data = from_networkx(nx_graph)
    x, edge_index = data.x, data.edge_index
    node_mask = torch.zeros(x.shape[0]).bool()
    node_mask[center_nodes] = True
    cnt = 0

    for i in range(radius):
        edge_mask = node_mask[edge_index[0]] | node_mask[edge_index[1]]
        node_mask[edge_index[0][edge_mask]] = True
        node_mask[edge_index[1][edge_mask]] = True

        if verbose:
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(nx_graph)
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_mask.numpy(), edge_color='gray', node_size=500, font_size=10, cmap=plt.cm.Reds)
            plt.title(f"tract_{pattern_id}_{cluster_index}_{cnt}")
            plt.savefig(f"tract_{pattern_id}_{cluster_index}_{cnt}.png")
            plt.close()
            cnt += 1

    node_mask[isolated_nodes] = False
    old_to_new = torch.full((x.shape[0],), -1, dtype=torch.long)
    old_to_new[node_mask] = torch.arange(node_mask.sum())
    new_edge_index = old_to_new[edge_index[:, edge_mask.bool()].int()]
    new_feature = x[node_mask]
    node_indices = torch.where(node_mask)[0]

    return to_networkx(Data(x=new_feature, edge_index=new_edge_index), node_attrs=['x']), node_indices

def calculate_distance_matrix(nx_graphs):
    gwdist=Fused_Gromov_Wasserstein_distance(alpha=0.2,features_metric='dirac')
    distance_function  = lambda g1, g2: gwdist.graph_d(g1, g2)
    number_of_graphs = len(nx_graphs)
    distance_matrix = torch.empty((number_of_graphs, number_of_graphs))
    nx_graphs = [Graph(graph) for graph in nx_graphs]
    for i in tqdm(range(number_of_graphs)):
        for j in range(number_of_graphs):
            if i==j:
                distance_matrix[i][j] = 0
            else:
                if len(nx_graphs[i].edges()) == 0 or len(nx_graphs[j].edges()) == 0:
                    distance_matrix[i][j] = math.inf
                else:
                    distance_matrix[i][j] = distance_function(nx_graphs[i], nx_graphs[j])
    return distance_matrix    


def calculate_median(distance_matrix):
    median_index = torch.argmin(torch.mean(distance_matrix, axis=1))
    return median_index

def delete_node(feature, edge_index, node_index):
    new_feature = feature.clone()
    new_edge_index = edge_index.clone()
    new_feature[node_index] = 0
    new_edge_index = new_edge_index[:, (new_edge_index[0] != node_index) & (new_edge_index[1] != node_index)]
    return new_feature, new_edge_index

def activation_score(embedding, pattern):
    pattern_components = pattern["components"]
    mask = torch.zeros(embedding.size(0), dtype=torch.bool, device=embedding.device)
    mask[pattern_components] = True
    pattern_embedding = embedding[mask].sum(dim=0)
    rest_embedding = embedding[~mask].sum(dim=0)
    return pattern_embedding #- rest_embedding
    #cosine similarity between mask and embedding
    #print(embedding.shape, mask.shape)
    #return torch.nn.functional.cosine_similarity(embedding, mask, -1)


def genenrate_heatmap(data, model, pattern, node_indices, target_class=1, device="cuda:0"):
    if not isinstance(data, Data):
        data = from_networkx(data)
    model.eval()
    features, edge_index = data.x.to(device), data.edge_index.to(device)
    original_embedding = model.embeddings(features, edge_index)[pattern["layer"]]
    original_probs = model(features, edge_index).softmax(dim=-1)[0][target_class]
    activated_nodes = find_support(original_embedding, pattern["components"])
    number_of_active_nodes = activated_nodes.shape[0]
    new_graphs = [delete_node(features, edge_index, node_index) for node_index in node_indices]
    #heatmap = [activation_score(model.embeddings(features, edge_index)[pattern["layer"]][node], pattern) for node in node_indices]  
    #heatmap = [(original_probs.item() - model(new_graph[0], new_graph[1]).softmax(dim=-1)[0][target_class].item()) for new_graph in new_graphs]
    #heatmap = [int((original_probs.item() <0.5) != (model(new_graph[0], new_graph[1]).softmax(dim=-1)[0][target_class].item() <0.5)) for new_graph in new_graphs]
    heatmap = [number_of_active_nodes - find_support(model.embeddings(new_graph[0], new_graph[1])[pattern["layer"]], pattern["components"]).shape[0] for new_graph in new_graphs]
    return heatmap

def generate_representation(pattern, pattern_index, embeddings, shap_dict, nx_graphs, dataset_name, target=1, device="cuda:0", model_type="gcn", model=None):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--model', type=str, default='gcn', help='Model to use (default: gnn)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda)')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    
    train_loader, graphs, features, labels, train_mask, val_mask, test_mask =   load_dataset_gnn(dataset_name, device=args.device)
    train_features, train_graphs, val_features, val_graphs, test_features, test_graphs = load_splited_data(dataset_name, device=args.device)
    nx_graphs = [to_networkx(data, node_attrs=['x']) for data in train_loader]
    model = load_gnn(dataset_name, device=args.device)
    embeddings = fill_embeddings(None, train_features, train_graphs, model)
    predictions = torch.tensor([model(data=data).argmax(dim=-1) for data in train_loader]).flatten()
    pattern_dict = read_patterns(dataset_name)
    patterns_to_explain = {
        "ba2": [6, 16, 12, 14],#[10,11, 15, ], #[5, 1, 14],#[6, 12, 13, 7, 14],
        "AlkaneCarbonyl": [35, 23, 38, 26],#[35, 23, 24, 36, 37, 38],
        "Benzen": [10, 32],#[13, 15]#[41, 30, 42, 43]#, 31, 44, 20, 45, 32, 10, 21, 46, 33, 47]
    }
    with open(f"{dataset_name} inside shap explanations {args.model} 1024.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    cluster_results = {}
    number_of_patterns = number_of_patterns = {
    "ba2": 18,
    "AlkaneCarbonyl": 45,
    "Benzen": 51
    }[dataset_name]

    for ii in range(number_of_patterns): #patterns_to_explain[dataset_name]:
        i = ii if dataset_name == "ba2" else ii + 1
        support_indices = calculate_pattern_support(pattern_dict[i], embeddings).tolist()
        shapley_values = [shap_dict[(index, 1)][0][i].item() for index in support_indices]
        values = []
        for graph_index in support_indices:
            graph = nx_graphs[graph_index]
            activated_nodes = find_support(embeddings[graph_index][pattern_dict[i]['layer']], pattern_dict[i]["components"])
            if len(activated_nodes) == 0:
                continue
            predicted_class = predictions[graph_index].item()
            shapley_value = shap_dict[(graph_index, predicted_class)][0][i].item()
            other_patterns_shapley_sum = sum(
            abs(shap_dict[(graph_index, predicted_class)][0][j].item())
            for j in range(number_of_patterns)
            if graph_index in calculate_pattern_support(pattern_dict[j], embeddings)
            )
            value = abs(shapley_value) / (len(activated_nodes) * other_patterns_shapley_sum)
            values.append(value)
        
        plt.figure()
        plt.hist(values, bins=30, alpha=0.75)
        plt.title(f"Pattern {ii} Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"representationmod/{dataset_name}_pattern_{ii}_histogram.png")
        plt.close()
