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
from bseval.utils import load_dataset_to_explain, calculate_edge_mask, h_fidelity
from scipy.stats import spearmanr

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
    with open(f"{dataset_name} inside shap_explanations {args.model} 10241.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    number_of_patterns = number_of_patterns = {
    "ba2": 18,
    "AlkaneCarbonyl": 45,
    "Benzen": 51
    }[dataset_name]
    pattern_supports = {i: calculate_pattern_support(pattern_dict[i], embeddings).tolist() for i in range(number_of_patterns)}
    """pattern_hfid = {i: 0 for i in range(number_of_patterns)}
    pattern_effectiveness = {i: 0 for i in range(number_of_patterns)}
    for i, data in enumerate(train_loader):
        predicted_class = predictions[i].item()
        best_pattern = None
        highest_score = -math.inf
        feature, edge_index = data.x, data.edge_index
        for j in range(number_of_patterns):
            if i in pattern_supports[j]:
                pattern = pattern_dict[j]
                activated_nodes = find_support(embeddings[i][pattern["layer"]] if pattern['layer'] < 3 else feature, pattern["components"])
                score = shap_dict[(i, predicted_class)][0][j]/len(activated_nodes)
                if score > highest_score:
                    
                    activated_nodes = find_support(embeddings[i][pattern["layer"]] if pattern['layer'] < 3 else feature, pattern["components"])
                    if not activated_nodes.shape[0] or activated_nodes.shape[0] == feature.shape[0]:
                        continue
                    node_mask = torch.zeros(feature.shape[0], device=args.device, dtype=bool)
                    node_mask[activated_nodes] = True
                    edges_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                    if edges_mask.sum() == edge_index.shape[1]:
                        continue
                    best_pattern = j
                    highest_score = score
        pattern = pattern_dict[best_pattern]
        activated_nodes = find_support(embeddings[i][pattern["layer"]] if pattern['layer'] < 3 else feature, pattern["components"])
        if not activated_nodes.shape[0] or activated_nodes.shape[0] == feature.shape[0]:
            continue
        node_mask = torch.zeros(feature.shape[0], device=args.device, dtype=bool)
        node_mask[activated_nodes] = True
        edges_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        try:
            hfid = h_fidelity(data, model, edges_mask, node_mask)
        except:
            hfid = torch.tensor((0,0,0,0))
        pattern_hfid[best_pattern] += hfid[1]
        pattern_effectiveness[best_pattern] += 1
    
    for key in pattern_hfid:
        pattern_hfid[key] /= len(pattern_supports[key])
        #pattern_hfid[key] = pattern_hfid[key] /pattern_effectiveness[key] if pattern_effectiveness[key] != 0 else 0   
    pattern_scores = [pattern_dict[i]["Score"] for i in range(number_of_patterns) if pattern_hfid[i] != 0]
    hfid_scores = [pattern_hfid[i] for i in range(number_of_patterns) if pattern_hfid[i] != 0]
    patterns = [i for i in range(number_of_patterns) if pattern_hfid[i] != 0]
    print(hfid_scores)
    print(pattern_scores)
    print(patterns)
    correlation, p_value = spearmanr(pattern_scores, hfid_scores)

    print(f"Spearman correlation: {correlation}, p-value: {p_value}")"""

    w = torch.zeros(2).to(args.device)
    class_counts = torch.zeros(2).to(args.device)
    for i, data in enumerate(train_loader):
        predicted_class = predictions[i].item()
        class_counts[predicted_class] += 1
    w[0] = max(1, class_counts[1]/class_counts[0])
    w[1] = max(class_counts[0]/class_counts[1], 1)
    graph_supports = []
    ranked_patterns = []    
    predicted_probs = torch.zeros(len(train_loader)).to(args.device)
    for i, data in enumerate(train_loader):
        feature = data.x
        graph_supports.append([j for j in range(number_of_patterns) if i in pattern_supports[j]])
        if i == 0: 
            print(graph_supports[i])
        ranked_patterns.append(sorted([j for j in range(number_of_patterns) ], key=lambda j: shap_dict[(i, predictions[i].item())][0][j].item()/len(find_support(embeddings[i][pattern_dict[j]["layer"]] if pattern_dict[j]['layer'] < 3 else feature, pattern_dict[j]["components"]))  if j in graph_supports[i] else -math.inf) )
        #/len(find_support(embeddings[i][pattern_dict[j]["layer"]] if pattern_dict[j]['layer'] < 3 else feature, pattern_dict[j]["components"]))
        predicted_class = predictions[i].item()
        predicted_prob = model(data=data).softmax(dim=-1)[0][predicted_class]
        predicted_probs[i] = predicted_prob
    pattern_values = []
    for j in range(number_of_patterns):
        pattern_value = torch.zeros(2).to(args.device)
        target = pattern_dict[j]["Target"]
        pattern = pattern_dict[j]
        for i, data in enumerate(train_loader):
            feature = data.x
            predicted_class = predictions[i].item()
            predicted_prob = predicted_probs[i]
            activated_nodes = find_support(embeddings[i][pattern["layer"]] if pattern['layer'] < 3 else feature, pattern["components"])
            nx_graph = nx_graphs[i]
            nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
            isolated_nodes = list(nx.isolates(nx_graph))
            nx_graph.remove_nodes_from(isolated_nodes)
            num_nodes = nx_graph.number_of_nodes()
            abs_sum = shap_dict[(i, predicted_class)][0].abs().sum()
            if i in pattern_supports[j]:
                try:    
                    rank = ranked_patterns[i].index(j)
                except: 
                    print(ranked_patterns[i], i)
                    raise
                if predicted_class == 0:
                    pattern_value[0] += shap_dict[(i, 0)][0][j]/(len(activated_nodes)*abs_sum)#predicted_prob#rank#/(len(activated_nodes)/num_nodes)  #shap_dict[(i, 0)][0][j]
                else:
                    pattern_value[1] += shap_dict[(i, 1)][0][j]/(len(activated_nodes)*abs_sum)#predicted_prob#rank#/(len(activated_nodes)/num_nodes)#shap_dict[(i, 1)][0][j]
        pattern_values.append((w[target]*pattern_value[target] + w[1-target]*pattern_value[1-target]).item())

    top_10_patterns = sorted(range(number_of_patterns), key=lambda i: pattern_dict[i]["Score"], reverse=True)[:10]
    pattern_scores = [pattern_dict[i]["Score"] for i in top_10_patterns]
    top_pattern_values = [pattern_values[i] for i in top_10_patterns]
    correlation, p_value = spearmanr(pattern_scores, top_pattern_values)
    print(f"Spearman correlation for top 10 patterns: {correlation}, p-value: {p_value}")

    # Calculate Spearman correlation for all patterns
    all_pattern_scores = [pattern_dict[i]["Score"] for i in range(number_of_patterns)]
    all_pattern_values = [pattern_values[i] for i in range(number_of_patterns)]
    correlation_all, p_value_all = spearmanr(all_pattern_scores, all_pattern_values)
    print(f"Spearman correlation for all patterns: {correlation_all}, p-value: {p_value_all}")
    # Print the top 4 patterns by pattern_values for each class
    # Print the top 4 patterns by pattern_value
    top_4_patterns = sorted(range(number_of_patterns), key=lambda i: pattern_values[i], reverse=True)
    print("Top 4 patterns by pattern_value:")
    for pattern_index in top_4_patterns:
        print(f"Pattern {pattern_index}: Value = {pattern_values[pattern_index]}")
    for target_class in range(2):
        class_pattern_indices = [i for i in range(number_of_patterns) if pattern_dict[i]["Target"] == target_class]
        class_pattern_values = [(i, pattern_values[i]) for i in class_pattern_indices]
        top_4_class_patterns = sorted(class_pattern_values, key=lambda x: x[1], reverse=True)
        
        print(f"Top 4 patterns for class {target_class}:")
        for pattern_index, pattern_value in top_4_class_patterns:
            print(f"Pattern {pattern_index}: Value = {pattern_value}")
    # Calculate Spearman correlation for top 10 patterns of each class
    for target_class in range(2):
        class_pattern_indices = [i for i in range(number_of_patterns) if pattern_dict[i]["Target"] == target_class]
        
        # Calculate Spearman correlation for all patterns of the target class
        all_class_pattern_scores = [pattern_dict[i]["Score"] for i in class_pattern_indices]
        all_class_pattern_values = [pattern_values[i] for i in class_pattern_indices]
        correlation_class_all, p_value_class_all = spearmanr(all_class_pattern_scores, all_class_pattern_values)
        print(f"Spearman correlation for all class {target_class} patterns: {correlation_class_all}, p-value: {p_value_class_all}")
        
        # Calculate Spearman correlation for top 10 patterns of the target class
        top_10_class_patterns = sorted(class_pattern_indices, key=lambda i: pattern_dict[i]["Score"], reverse=True)
        class_pattern_scores = [pattern_dict[i]["Score"] for i in top_10_class_patterns]
        class_pattern_values = [pattern_values[i] for i in top_10_class_patterns]
        correlation_class, p_value_class = spearmanr(class_pattern_scores, class_pattern_values)
        """if target_class == 1:
            top_patterns_with_values = sorted(zip(top_10_class_patterns, class_pattern_values), key=lambda x: x[1], reverse=True)
            for pattern_index, pattern_value in top_patterns_with_values:
                pattern_si = pattern_dict[pattern_index]["Score"]
                print(f"Pattern {pattern_index}: Value = {pattern_value}, SI = {pattern_si}")"""
        print(f"Spearman correlation for top 10 class {target_class} patterns: {correlation_class}, p-value: {p_value_class}")