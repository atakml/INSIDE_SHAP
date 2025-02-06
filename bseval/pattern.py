import pickle

import torch
import matplotlib.cm as cm

import pandas as pd
from bseval.utils import load_dataset_to_explain
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from graphrep.drawer import atoms
import networkx as nx
import matplotlib.pyplot as plt
from ExplanationEvaluation.datasets.ground_truth_loaders import *
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from torch_geometric.data import Data
from pathlib import Path
parent_directory = str(Path.cwd().parent)


def find_support(embeddings, components):
    cols = embeddings[:, components]
    activated_nodes = torch.where(cols.all(axis=1))[0]
    return activated_nodes


def generate_node_values(feature, edge_index, model, shap_values, rule_dict, index, dataset_name):
    embeddings = model.embeddings(feature, edge_index)
    target_class = torch.argmax(model(feature, edge_index)[0])
    shap_values = shap_values[target_class][0]
    flag = False
    max_value = -1000
    min_value = 1000
    values = []
    for ii, pattern in rule_dict.items():
        layer, components, label = pattern['layer'], pattern['components'], pattern['label']

        node_values = torch.zeros(feature.shape[0], device="cuda:0")
        value = shap_values[ii].detach().clone()
        activated_nodes = find_support(embeddings[layer] if layer < len(embeddings) else feature, components)
        if not activated_nodes.shape[0]:
            continue
        value/= len(activated_nodes)
        max_value = max(max_value, value)
        min_value = min(min_value, value)
        values.append(int(value.cpu().item()*1000000))
        if ii==14 and index==20:
            print("!!!"*100)
            print(values)
            print(shap_values[ii])
            print(len(activated_nodes))
            print(shap_values)
            print("!!"*100)
    values.append(0)
    values = sorted(list(set(values)))
    norm = plt.Normalize(vmin=min_value, vmax=max_value)
    for ii, pattern in rule_dict.items():
        layer, components, label = pattern['layer'], pattern['components'], pattern['label']

        node_values = torch.zeros(feature.shape[0], device="cuda:0")
        value = shap_values[ii].detach().clone()
        activated_nodes = find_support(embeddings[layer] if layer < len(embeddings) else feature, components)
        if not activated_nodes.shape[0]:
            continue
        value/= len(activated_nodes)
        if index == 20:
            print(ii, value)
        node_values[activated_nodes.cpu()] += value.cpu().item()#/((1+layer) if layer != len(embeddings) else 1)
        data = Data(x=feature, edge_index=edge_index, node_value=node_values)
        nx_graph = to_networkx(data, node_attrs=["x", "node_value"])
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
        #nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
        colors = {j: color for j, color in nx_graph.nodes(data="node_value")}
        colors = [colors[j] for j in sorted(colors.keys())]
        labels = {node: atoms[dataset_name][torch.argmax(torch.tensor(node_data)).item()] for node, node_data in nx_graph.nodes(data='x')}
        nx_graph.add_node(1000, x=0)
        nx_graph.add_node(1001, x=1)
        nx_graph.add_node(1002, x=2)
        l= ["min", "0", "max"]
        colors.append(min_value.item())
        colors.append(0)
        cmap = plt.get_cmap('viridis')

        colors.append(max_value.item())
        try:
            colors = [values.index(int(color*1000000))*2 for color in colors]
        except:
            print(sorted(list((set(map(lambda x: int(x*1000000), colors))))))
            print(values)
            print(ii)
            print(index)
            print(shap_values[ii])
            print(len(activated_nodes))
            print(shap_values)
            raise
        labels = {node: labels[node] if node in labels.keys() else l[node_data] for node, node_data in nx_graph.nodes(data="x")}
        plt.clf()
        if not flag:
            pos = nx.spring_layout(nx_graph)
            flag = True
        if index==20 and ii==3:
            print(colors)
        nx.draw(nx_graph, pos=pos, labels=labels, with_labels=True, node_color=colors, cmap=cmap)
        plt.savefig(f"bseval/figs/{dataset_name}/soft_{index}_{ii}.png")


for dataset in ["AlkaneCarbonyl", "mutag"]: #["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:# ["mutag", "BBBP", "aids"]:#["AlkaneCarbonyl", "Benzen", "ba2"]:#["ba2", "Benzen"]: #["BBBP", "AlkaneCarbonyl", "aids"]:#["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:   

    with open(f"{parent_directory}/shap_inside/{dataset} shap explanations gcn.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    with open(f"rule_mask_{dataset}_below_indices.pkl", "rb") as file:
        indices_below = pickle.load(file)
    rule_path = f"{parent_directory}/shap_inside/codebase/ExplanationEvaluation/PatternMining/{dataset}_activation_encode_motifs.csv"
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
                        'covers': (negative_cover, positive_cover)}

    train_loader, gnn_model, graphs, features, labels, train_mask = load_dataset_to_explain(dataset)
    with open(f"rule_masks_{dataset}_node_simple_devide.pkl", "rb") as file:
        node_values = pickle.load(file)
    #node_values = []
    #edge_values = []
    hfid = 0
    cnt = 0 
    for i, data in tqdm(enumerate(train_loader)):
        if i not in indices_below:
            continue
        shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
        feature, edge_index = data.x, data.edge_index
        generate_node_values(feature, edge_index, gnn_model, shap_values, rule_dict, i, dataset)
