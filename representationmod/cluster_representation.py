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
from sklearn.neighbors import LocalOutlierFactor


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
from scipy.stats import shapiro
from PIL import Image


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


def calculate_clusters(shapley_values, pattern_index=0, dataset_name="ba2", predicted_labels=None):
    if dataset_name != "ba2":
        pattern_index -= 1
    shapley_values = shapley_values.cpu().numpy().reshape(-1, 1)
    lof = LocalOutlierFactor(n_neighbors=3)
    lof_scores = lof.fit_predict(shapley_values)
    cluster_labels = np.zeros_like(shapley_values, dtype=int).flatten()
    label_dict = {}

    for index in np.where(lof_scores <= -1)[0]:
        label_dict[shapley_values[index][0].item()] = -1

    sorted_filtered_shapley_values = np.sort(shapley_values[lof_scores > -1].flatten())
    distances = np.diff(sorted_filtered_shapley_values.flatten())
    percentile = np.percentile(distances, 99.9)
    cut_indices = np.where(distances > percentile)[0]
    cluster_labels_filtered = np.zeros_like(sorted_filtered_shapley_values, dtype=int).flatten()
    start = 0

    for i, cut_index in enumerate(cut_indices):
        cluster_labels_filtered[start:cut_index + 1] = i
        start = cut_index + 1
    cluster_labels_filtered[start:] = len(cut_indices)

    for i in range(sorted_filtered_shapley_values.shape[0]):
        label_dict[sorted_filtered_shapley_values[i].item()] = cluster_labels_filtered[i]

    for i in range(shapley_values.shape[0]):
        cluster_labels[i] = label_dict[shapley_values[i][0].item()]
    for i, v in label_dict.items():
        for j, u in label_dict.items():
            if i  < j and v!= -1 and u != -1:
                assert v <= u           

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(sorted_filtered_shapley_values, np.zeros_like(sorted_filtered_shapley_values), c=cluster_labels_filtered, marker='o')

    # Add accolade with number of members for each cluster
    unique_labels = np.unique(cluster_labels_filtered)
    max_value, min_value = sorted_filtered_shapley_values.max(), sorted_filtered_shapley_values.min()
    num_labels = unique_labels.shape[0]
    dist = (max_value - min_value)/num_labels
    for label in unique_labels:
        if label != -1:
            ylims = plt.ylim()
            nums = sorted_filtered_shapley_values[cluster_labels_filtered==label]
            d = nums.max() - nums.min()
            m = nums.min() + d/2
            print(m, d, nums.max(), nums.min())
            ylims = plt.ylim()
            xlims = plt.xlim()
            d = d*16/(xlims[1]-xlims[0])
            plt.annotate(f'{nums.shape[0]}', xy=(m, ylims[1]*0.25), xytext=(m, ylims[1]*0.6),
                         fontsize=18, ha='center', va='bottom',
                         bbox=dict(boxstyle='square', fc='white', color='k'),
                         arrowprops=dict(arrowstyle=f'-[, widthB={d}, lengthB=1.5', lw=2.0, color='k'))
            plt.annotate('', xy=(m, -ylims[1]*0.25), xytext=(m, -ylims[1]*0.3),
                         fontsize=18, ha='center', va='bottom',
                         bbox=dict(boxstyle='square', fc='white', color='k'),
                         arrowprops=dict(arrowstyle=f'-[, widthB={d}, lengthB=1.5', lw=1.0, color='k'))
    plt.xlabel('Shapley Values', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'Shapley Value Clustering for Pattern {pattern_index}', fontsize=20)
    plt.savefig(f"representationmod/{dataset_name}_pattern_{pattern_index}_clustering.png")
    plt.close()
    return cluster_labels.flatten()


def ego_network(nx_graph, center_nodes, radius, verbose=False, pattern_id=10, cluster_index=0):
    isolated_nodes = [node for node in nx_graph.nodes() if nx_graph.degree(node) == 0]
    data = from_networkx(nx_graph)
    x, edge_index = data.x, data.edge_index
    node_mask = torch.zeros(x.shape[0]).bool()
    node_mask[center_nodes] = True
    cnt = 0

    if verbose:
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_color=node_mask.numpy(), edge_color='gray', node_size=500, font_size=10, cmap=plt.cm.Reds)
        plt.title(f"tract_{pattern_id}_{cluster_index}_{cnt}")
        plt.savefig(f"tract_{pattern_id}_{cluster_index}_{cnt}.png")
        plt.close()
        cnt += 1

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
    clustering_info_path = f"representationmod/results/{dataset_name} {pattern_index} clustering_info_l1_seg.pkl"
    support_indices = calculate_pattern_support(pattern, embeddings).tolist()
    center_nodes = {idx: find_support(embeddings[idx][pattern["layer"]], pattern["components"]).tolist() for idx in support_indices}
    masks = {}
    valid_support_indices = []
    valid_center_nodes = {}
    
    for idx in support_indices:
        ego_net, node_indices = ego_network(nx_graphs[idx], center_nodes[idx], pattern["layer"] + 1)
        if len(ego_net.edges()) > 0:
            masks[idx] = (ego_net, node_indices)
            isolated_nodes = [node for node in nx_graphs[idx].nodes() if nx_graphs[idx].degree(node) == 0]
            center_nodes[idx] = [node for node in center_nodes[idx] if node not in isolated_nodes]
            valid_support_indices.append(idx)
            valid_center_nodes[idx] = center_nodes[idx]
            
            
    support_indices = valid_support_indices
    center_nodes = valid_center_nodes

    if  os.path.exists(clustering_info_path):# and not dataset_name in ["Benzen", "AlkaneCarbonyl"]:
        with open(clustering_info_path, "rb") as file:
            clustering_info = pickle.load(file)
        valid_indices = list(reduce(lambda a, b: set(a).union(set(b)), list(clustering_info["cluster_members"].values())))
        support_indices = list(filter(lambda x: x in valid_indices, support_indices))
        shapley_values = [shap_dict[(index, target)][0][pattern_index].item() for index in support_indices]
        

        cluster_median = {}
        heat_maps = {}
        median_shapley_values = {}
        print("pattern number:", pattern_index, "number of clusters", len(clustering_info["cluster_members"]))
        cluster_labels = {}
        for cluster, members in clustering_info["cluster_members"].items():
            if len(members) > 1:
                median_index = calculate_median(clustering_info["distance_matrices"][cluster])
            else:
                median_index = 0
            median_graph_index = members[median_index]
            predicted_class = predictions[median_graph_index].item()
            print(f"Cluster {cluster}: Predicted class of the median graph: {predicted_class}")
            probability = model(from_networkx(nx_graphs[median_graph_index]).x.to(device), from_networkx(nx_graphs[median_graph_index]).edge_index.to(device)).softmax(dim=-1)[0]
            print(f"Prediction probability: {probability}")
        for cluster, members in clustering_info["cluster_members"].items():
            for memeber in members:
                cluster_labels[support_indices.index(memeber)] = cluster
            if len(members) > 1:
                print(clustering_info["distance_matrices"].keys())
                distance_matrix = clustering_info["distance_matrices"][cluster]
                median_index = calculate_median(distance_matrix)
                cluster_median[cluster] = masks[members[median_index]][0]
                node_indices = masks[members[median_index]][1].tolist()
                #node_indices = [node_indices.index(node) for node in center_nodes[members[median_index]]]                 
                node_indices = [node_indices.index(node) for node in masks[members[median_index]][1].tolist()]
                median_shapley_values[cluster] = shap_dict[(members[median_index], target)][0][pattern_index].item()
                #heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[median_index]]), model, pattern, center_nodes[members[median_index]], target_class=target, device=device)
                #heatmap = genenrate_heatmap(masks[members[median_index]][0], model, pattern, node_indices, target_class=target, device=device)
                heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[median_index]]), model, pattern, masks[members[median_index]][1].tolist(), target_class=target, device=device)

            else:
                median_index = 0 
                cluster_median[cluster] = masks[members[0]][0]
                node_indices = masks[members[0]][1].tolist()
                #node_indices = [node_indices.index(node) for node in center_nodes[members[0]]]
                node_indices = [node_indices.index(node) for node in masks[members[median_index]][1].tolist()]
                median_shapley_values[cluster] = shap_dict[(members[0], target)][0][pattern_index].item()
                heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[0]]), model, pattern, masks[members[0]][1].tolist(), target_class=target, device=device)
                #heatmap = genenrate_heatmap(masks[members[median_index]][0], model, pattern, node_indices, target_class=target, device=device)

                #heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[0]]), model, pattern, center_nodes[members[0]], target_class=target, device=device)
            heatmap = {node_indices[i]: heatmap[i] for i in range(len(node_indices))}
            heat_maps[cluster] = heatmap
        cluster_labels = [cluster_labels[i] for i in range(len(support_indices))]
    else:
        shapley_values = [shap_dict[(index, target)][0][pattern_index].item() for index in support_indices]
        #shapley_values = [shap_dict[(index, target)][0][pattern_index].item() / model(from_networkx(nx_graphs[index]).x.to(device), from_networkx(nx_graphs[index]).edge_index.to(device)).softmax(dim=-1)[0][target].item() for index in support_indices]
        cluster_labels = calculate_clusters(torch.tensor(shapley_values), pattern_index, dataset_name)
        valid_indices = [i for i, label in enumerate(cluster_labels) if label != -1]
        support_indices = [support_indices[i] for i in valid_indices]
        cluster_labels = [cluster_labels[i] for i in valid_indices]
        print("!"*100)
        print(f"pattern number: {pattern_index}, number of clusters: {len(np.unique(cluster_labels))}")
        print("??"*100)
        cluster_members = {i: [] for i in range(len(np.unique(cluster_labels)))}
        for idx, label in enumerate(cluster_labels):
            cluster_members[label].append(support_indices[idx])
        distance_matrices = {}
        cluster_median = {}
        heat_maps = {}
        median_shapley_values = {}
        for cluster, members in cluster_members.items():
            if len(members) > 1:
                subgraph_list = [masks[idx][0] for idx in members] 
                distance_matrix = calculate_distance_matrix(subgraph_list)
                distance_matrices[cluster] = distance_matrix
                median_index = calculate_median(distance_matrix)
                cluster_median[cluster] = masks[members[median_index]][0]
                node_indices = masks[members[median_index]][1].tolist()
                #node_indices = [node_indices.index(node) for node in center_nodes[members[median_index]]]
                node_indices = [node_indices.index(node) for node in masks[members[median_index]][1].tolist()]
                median_shapley_values[cluster] = shap_dict[(members[median_index], target)][0][pattern_index].item()
                #heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[median_index]]), model, pattern, center_nodes[members[median_index]], target_class=target, device=device)
                heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[median_index]]), model, pattern, masks[members[median_index]][1].tolist(), target_class=target, device=device)
                #heatmap = genenrate_heatmap(masks[members[median_index]][0], model, pattern, node_indices, target_class=target, device=device)
                heatmap = {node_indices[i]: heatmap[i] for i in range(len(node_indices))}
                heat_maps[cluster] = heatmap
            else:
                median_index = 0
                distance_matrices[cluster] = None
                cluster_median[cluster] = masks[members[0]][0]
                node_indices = masks[members[0]][1].tolist()
                #node_indices = [node_indices.index(node) for node in center_nodes[members[0]]]
                node_indices = [node_indices.index(node) for node in masks[members[median_index]][1].tolist()]
                median_shapley_values[cluster] = shap_dict[(members[0], target)][0][pattern_index].item()
                #heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[0]]), model, pattern, center_nodes[members[0]], target_class=target, device=device)
                #heatmap = genenrate_heatmap(masks[members[median_index]][0], model, pattern, node_indices, target_class=target, device=device)

                heatmap = genenrate_heatmap(from_networkx(nx_graphs[members[0]]), model, pattern, masks[members[0]][1].tolist(), target_class=target, device=device)
                heatmap = {node_indices[i]: heatmap[i] for i in range(len(node_indices))}
                heat_maps[cluster] = heatmap

        clustering_info = {
            "distance_matrices": distance_matrices,
            "cluster_median": cluster_median,
            "cluster_members": cluster_members
        }
        os.makedirs(os.path.dirname(clustering_info_path), exist_ok=True)
        with open(clustering_info_path, "wb") as file:
            pickle.dump(clustering_info, file)

    #return cluster_median, heat_maps, median_shapley_values
    activated_nodes = {}
    for cluster, members in clustering_info["cluster_members"].items():
        if len(members) > 1:
            median_index = calculate_median(clustering_info["distance_matrices"][cluster])
        else:
            median_index = 0
        median_graph_index = members[median_index]
        median_node_indices = masks[median_graph_index][1].tolist()
        activated_nodes[cluster] = [median_node_indices.index(node) for node in center_nodes[median_graph_index] if node in median_node_indices]
    for cluster, members in clustering_info["cluster_members"].items():
        if len(members) > 1:
            median_index = calculate_median(clustering_info["distance_matrices"][cluster])
        else:
            median_index = 0
        median_graph_index = members[median_index]
        median_graph = nx_graphs[median_graph_index]
        median_node_indices = masks[median_graph_index][1].tolist()

        # Remove isolated nodes
        non_isolated_nodes = [node for node in median_graph.nodes() if median_graph.degree(node) > 0]
        median_graph = median_graph.subgraph(non_isolated_nodes)

        # Draw the main graph with highlighted median graph nodes
        plt.figure(figsize=(8, 8), dpi=300)
        pos = nx.spring_layout(median_graph)
        node_colors = ['red' if node in median_node_indices else 'blue' for node in median_graph.nodes()]
        node_labels = {node: atoms[dataset_name][torch.tensor(data['x']).argmax().item()] for node, data in median_graph.nodes(data=True)}
        nx.draw(median_graph, pos, labels=node_labels, with_labels=True, node_color=node_colors, node_size=500, edge_color='gray', font_size=10, cmap=plt.cm.Reds)
        plt.title(f"Main Graph with Median Graph Nodes Highlighted (Cluster {cluster})")
        plt.savefig(f"representationmod/figs/{dataset_name}/{dataset_name}_main_graph_cluster_{pattern_index}_{cluster}.png")
        plt.close()

    for cluster, members in clustering_info["cluster_members"].items():
        if len(members) > 1:
            median_index = calculate_median(clustering_info["distance_matrices"][cluster])
        else:
            median_index = 0
        median_graph_index = members[median_index]
        print(f"Cluster {cluster}: Global index of the median graph: {median_graph_index}, {len(members)}")
    return cluster_median, heat_maps, median_shapley_values, activated_nodes


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
        "ba2": [6, 16, 14, 12],#[10,11, 15, ], #[5, 1, 14],#[6, 12, 13, 7, 14],
        "AlkaneCarbonyl": [35, 23, 26, 38],#[35, 23, 24, 36, 37, 38],
        "Benzen": [41, 42, 10, 32]#[10, 32]#, 41, 42],#[13, 15]#[41, 30, 42, 43]#, 31, 44, 20, 45, 32, 10, 21, 46, 33, 47]
    }
    with open(f"{dataset_name} inside shap explanations {args.model} 1024.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    cluster_results = {}
    number_of_patterns = number_of_patterns = {
    "ba2": 18,
    "AlkaneCarbonyl": 45,
    "Benzen": 51
    }[dataset_name]
    #a = [14, 23, 25, 31, 42, 64, 71, 78, 80, 116, 117, 137, 144, 152, 175, 178, 185, 187, 189, 195, 206, 225, 234, 244, 249, 256, 263, 268, 270, 275, 278, 280, 286, 295, 296, 299, 302, 304, 310, 317, 319, 341, 344, 346, 350, 360, 372, 375, 376, 383, 384, 395, 418, 434, 440, 445, 455, 479, 494, 504, 505, 522, 524, 546, 548, 551, 552, 561, 572, 581, 589, 590, 597, 602, 612, 622, 623, 627, 640, 647, 652, 654, 668, 671, 680, 686, 694, 700, 705, 710, 712, 722, 747, 753, 761, 773, 782, 789, 793, 795, 816, 824, 851, 867, 868, 870, 881, 896, 903, 906, 907, 908, 928, 932, 935, 936, 939, 949, 952, 972, 979, 1013, 1022, 1041, 1070, 1084, 1132, 1146, 1159, 1168, 1170, 1185, 1235, 1236, 1237, 1240, 1241, 1276, 1300, 1315, 1319, 1324, 1326, 1339, 1347, 1359, 1362, 1384, 1404, 1409, 1436, 1438, 1441, 1445, 1460, 1468, 1472, 1475, 1500, 1506, 1542, 1546, 1550, 1553, 1560, 1565, 1579, 1590, 1592, 1603, 1615, 1635, 1661, 1663, 1695, 1700, 1753, 1755, 1773, 1776, 1783, 1786, 1788, 1807, 1822, 1824, 1826, 1832, 1844, 1865, 1870, 1873, 1881, 1884, 1886, 1888, 1889, 1894, 1911, 1927, 1934, 1946, 1947, 1954, 1963, 1967, 1996, 2003, 2005, 2010, 2018, 2029, 2032, 2043, 2048, 2075, 2081, 2096, 2121, 2122, 2166, 2170, 2175, 2179, 2181, 2185, 2186, 2189, 2190, 2203, 2235, 2237, 2246, 2266, 2277, 2292, 2347, 2353, 2369, 2380, 2397, 2407, 2410, 2428, 2432, 2435, 2439, 2451, 2459, 2464, 2476, 2499, 2509, 2515, 2517, 2520, 2541, 2543, 2591, 2609, 2620, 2625, 2642, 2644, 2645, 2656, 2670, 2676, 2688, 2694, 2699, 2701, 2716, 2717, 2720, 2722, 2756, 2757, 2760, 2765, 2776, 2777, 2778, 2798, 2800, 2809, 2814, 2816, 2846, 2854, 2882, 2896, 2900, 2904, 2912, 2915, 2943, 2944, 2948, 2951, 2977, 2984, 2994, 2995, 3019, 3034, 3046, 3056, 3058, 3063, 3065, 3082, 3093, 3094, 3111, 3118, 3119, 3123, 3141, 3143, 3148, 3151, 3160, 3176, 3177, 3191, 3200, 3216, 3219, 3284, 3289, 3290, 3292, 3295, 3300, 3308, 3315, 3317, 3325, 3348, 3349, 3359, 3364, 3368, 3385, 3397, 3401, 3429, 3437, 3449, 3472, 3482, 3487, 3509, 3510, 3531, 3534, 3546, 3548, 3570, 3593, 3601, 3617, 3634, 3646, 3648, 3671, 3682, 3687, 3690, 3702, 3718, 3728, 3747, 3765, 3771, 3796, 3805, 3807, 3814, 3821, 3823, 3837, 3841, 3844, 3850, 3878, 3889, 3892, 3897, 3938, 3939, 3978]
    #a = list(filter(lambda x: predictions[x] == 1, a))
    #torch.manual_seed(1)
    #print(a[torch.randint(0, len(a), (1,)).item()])
    
    """
    pattern_supports = {i: calculate_pattern_support(pattern_dict[i], embeddings).tolist() for i in range(number_of_patterns)}
    pattern_effectiveness = {i: 0 for i in range(number_of_patterns)}
    
    for graph_idx in range(len(nx_graphs)):
        predicted_class = predictions[graph_idx].item()
        max_shap_value = -float('inf')
        most_effective_pattern = None
        for pattern_idx in range(number_of_patterns):
            if graph_idx in pattern_supports[pattern_idx]:
                shap_value = shap_dict[(graph_idx, predicted_class)][0][pattern_idx].item()
                if shap_value > max_shap_value:
                    max_shap_value = shap_value
                    most_effective_pattern = pattern_idx

        if most_effective_pattern is not None:
            pattern_effectiveness[most_effective_pattern] += 1

    pattern_support_sizes = {i: len(pattern_supports[i]) for i in range(number_of_patterns)}
    pattern_effectiveness_ratios = {i: pattern_effectiveness[i] / pattern_support_sizes[i] if pattern_support_sizes[i] > 0 else 0 for i in range(number_of_patterns)}
    print("Pattern effectiveness ratios:")
    top_10_patterns = sorted(pattern_effectiveness_ratios.items(), key=lambda x: x[1], reverse=True)[:10]
    for pattern_index, ratio in top_10_patterns:
        print(f"Pattern {pattern_index}: Effectiveness Ratio = {ratio}")
    
    remaining_graphs = set(range(len(nx_graphs)))
    remaining_graphs = {graph_idx for graph_idx in remaining_graphs if predictions[graph_idx] == 1}
    remaining_patterns = set(range(number_of_patterns))
    effective_pattern_for_graph = {i: None for i in range(len(nx_graphs))}
    remained_graphs  =remaining_graphs.copy()
    while remaining_patterns:
        pattern_effectiveness = {i: 0 for i in remaining_patterns}
        single_effective_patterns = {i: 0 for i in remaining_patterns}
        for graph_idx in remained_graphs:
            predicted_class = predictions[graph_idx].item()
            max_shap_value = -float('inf')
            most_effective_pattern = None
            for pattern_idx in remaining_patterns:
                if graph_idx in pattern_supports[pattern_idx]:
                    shap_value = shap_dict[(graph_idx, predicted_class)][0][pattern_idx].item()/find_support(embeddings[graph_idx][pattern_dict[pattern_idx]["layer"]], pattern_dict[pattern_idx]["components"]).shape[0]
                    if shap_value > max_shap_value:
                        max_shap_value = shap_value
                        most_effective_pattern = pattern_idx
                        effective_pattern_for_graph[graph_idx] = pattern_idx
            if most_effective_pattern is not None:
                pattern_effectiveness[most_effective_pattern] += 1
                if shap_dict[(graph_idx, predicted_class)][0][most_effective_pattern].item() >= 0.5:
                    single_effective_patterns[most_effective_pattern] += 1

        most_effective_pattern = max(pattern_effectiveness, key=lambda x: pattern_effectiveness[x])

        
        graphs_to_delete = set()
        for graph_idx in remained_graphs:
            if effective_pattern_for_graph[graph_idx] == most_effective_pattern:
                graphs_to_delete.add(graph_idx)

        remained_graphs -= graphs_to_delete
        remaining_patterns.remove(most_effective_pattern)
        print(f"Most effective pattern: {most_effective_pattern if dataset_name == 'ba2' else most_effective_pattern -1}, {pattern_effectiveness[most_effective_pattern]}, {len(remained_graphs)}")

    """
    
    """
    for pattern_index in range(51):
        i = pattern_index if dataset_name == "ba2" else pattern_index + 1
        target = pattern_dict[i]["Target"]
        support_indices = calculate_pattern_support(pattern_dict[i], embeddings).tolist()
        t1_values = []
        t2_values = []
        support_indices = calculate_pattern_support(pattern_dict[i], embeddings).tolist()
        for g in range(len(nx_graphs)):
            shap_value = shap_dict[(g, target)][0][i].item()
            probability = model(from_networkx(nx_graphs[g]).x.to(args.device), from_networkx(nx_graphs[g]).edge_index.to(args.device)).softmax(dim=-1)[0][target].item()
            if predictions[g] == target:
                t1_values.append(shap_value)#/ probability)
            else:
                t2_values.append(shap_value)#/ probability)

        t1 = np.mean(t1_values) if t1_values else 0
        t2 = np.mean(t2_values) if t2_values else 0

        cluster_results[i] = t1 + t2

    top_patterns = sorted(cluster_results.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 5 patterns with highest t1 - t2 values:")
    for pattern_index, value in top_patterns:
        print(f"Pattern {pattern_index}: t1 - t2 = {value}")
    
    """
    number_of_clusters = {}
    for ii in patterns_to_explain[dataset_name]:
        i = ii if dataset_name == "ba2" else ii + 1
        support_indices = calculate_pattern_support(pattern_dict[i], embeddings).tolist()
        shapley_values = [shap_dict[(index, 1)][0][i].item() for index in support_indices]
        cluster_labels = calculate_clusters(torch.tensor(shapley_values), pattern_index=i, dataset_name=dataset_name, predicted_labels=predictions[support_indices].tolist())
        """plt.figure(figsize=(10, 6))
        for label in np.unique(predictions[support_indices]):
            label_indices = [idx for idx in range(len(support_indices)) if predictions[support_indices[idx]] == label]
            label_shapley_values = [shapley_values[idx] for idx in label_indices]
            plt.hist(label_shapley_values, bins=30, alpha=0.5, label=f'Label {label}')
        
        plt.xlabel('Shapley Values')
        plt.ylabel('Frequency')
        plt.title(f'Shapley Value Distribution for Pattern {ii}')
        plt.legend()
        plt.savefig(f"representationmod/shapley_distribution_{dataset_name}_pattern_{ii}.png")
        plt.show()"""
        
        
        representations, heat_maps, average_shap_values, activated_nodes = generate_representation(pattern_dict[i], i, embeddings, shap_dict, nx_graphs, dataset_name, device=args.device, model_type=args.model, model=model)
        
        #generate_a union set of all heat_mnaps, and return the index of each value in this union when sorted 
        #heat_map_values = sorted(list(set(reduce(lambda a, b: a+b, list(map(lambda x: list(x.values()), heat_maps.values())))))) + [-100]
        #heat_map_indices = {cluster: {node: heat_map_values.index(heat_map) for node, heat_map in heat_map.items()} for cluster, heat_map in heat_maps.items()}
        #cmap = ListedColormap(np.vstack((generate_colormap(len(heat_map_values) - 3), [[1, 0, 0, 1]] * 3)))
        # Add three different red colors with increasing contrast
        #red_colors = np.array([[1, 0, 0, 0.5], [1, 0, 0, 0.75], [1, 0, 0, 1]])
        cmap = plt.cm.Blues
        heat_map_indices = heat_maps
        os.makedirs(f"representationmod/figs/{dataset_name}", exist_ok=True)
        plt.clf()

        num_clusters = len(representations)
        num_cols = min(num_clusters, 5)
        num_rows = (num_clusters + num_cols - 1) // num_cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten() if num_clusters > 1 else [axes]

        merged_images = []
        for cluster, median_graph in representations.items():
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            pos = nx.kamada_kawai_layout(median_graph)
            min_val = min(heat_map_indices[cluster].values())
            max_val = max(heat_map_indices[cluster].values())
            node_colors = [(heat_map_indices[cluster][node] - min_val + 1) / (max_val - min_val + 1) if node in heat_map_indices[cluster] else 0 for node in median_graph.nodes()]
            if len(set(node_colors)) == 1:
                node_colors = [1] * len(node_colors)
            if dataset_name == "ba2":
                nx.draw(median_graph, ax=ax, with_labels=False, node_color=node_colors, node_size=2000, edge_color='gray', pos=pos, cmap=cmap)
            else:
                node_labels = {node: atoms[dataset_name][torch.tensor(data['x']).argmax().item()] for node, data in median_graph.nodes(data=True)}
                nx.draw(median_graph, ax=ax, labels=node_labels, with_labels=True, node_color=node_colors, cmap=cmap, node_size=2000, edge_color='gray', pos=pos, font_size=20, font_color='#777777')
            median_value = np.percentile(node_colors, 75)
            max_node = list(map(lambda x: x[0], list(filter(lambda x: x[1] >= median_value, enumerate(node_colors)))))
            nx.draw_networkx_nodes(median_graph, pos, nodelist=median_graph.nodes(), node_color=node_colors, node_size=2000, edgecolors=['red' if node in activated_nodes[cluster] else 'black' for node in median_graph.nodes()], linewidths=[2 if node in activated_nodes[cluster] else 0.5 for node in median_graph.nodes()], cmap=cmap, ax=ax)
            ax.set_title(f"Cluster {cluster}\n Median Shapley Value: {average_shap_values[cluster]:.5f}", fontsize=35)
            plt.tight_layout()
            image_path = f"representationmod/figs/{dataset_name}/{dataset_name} {ii}_{args.model}_patt_cluster_{cluster}_elbow.png"
            plt.savefig(image_path)
            plt.close(fig)
            merged_images.append(image_path)

        # Merge images horizontally

        images = [Image.open(img) for img in merged_images]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        merged_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width

        merged_image_path = f"representationmod/figs/{dataset_name}/{dataset_name}_merged_{ii}_{args.model}_patt_clusters.png"
        merged_image.save(merged_image_path)

        # Read the merged image and the one built in the clustering function
        clustering_image_path = f"representationmod/{dataset_name}_pattern_{ii}_clustering.png"
        clustering_image = Image.open(clustering_image_path)
        final_width = max(merged_image.width, clustering_image.width)
        clustering_image = clustering_image.resize((final_width, int(clustering_image.height * final_width / clustering_image.width)))
        final_height = merged_image.height + clustering_image.height

        final_image = Image.new('RGB', (final_width, final_height))
        final_image.paste(clustering_image, (0, 0))
        final_image.paste(merged_image, (0, clustering_image.height))

        final_image_path = f"representationmod/figs/{dataset_name}/{dataset_name}_final_{ii}_{args.model}_patt_clusters.png"
        final_image.save(final_image_path, dpi=(300, 300))

