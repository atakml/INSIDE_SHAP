import pickle

import torch

from bseval.utils import load_dataset_to_explain
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from graphrep.drawer import atoms
import networkx as nx
import matplotlib.pyplot as plt
from ExplanationEvaluation.datasets.ground_truth_loaders import *
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

dataset_name = "ba2"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import matplotlib
def generate_colormap(N, base_color_hue=0):
    # N: Number of colors to generate
    # base_color_hue: Choose the hue (between 0 and 1), 0 is red, 0.33 is green, 0.66 is blue, etc.
    
    # Generate values (brightness levels) from dark to bright
    brightness_levels = np.linspace(0.2, 1, N)
    
    # Create an array for HSV colors
    hsv_colors = np.zeros((N, 3))
    
    # Set hue to a constant value (base_color_hue)
    hsv_colors[:, 0] = base_color_hue  # Fixed hue for a single color
    
    # Set saturation to 1 for vivid colors
    hsv_colors[:, 1] = 1
    
    # Set value (brightness) to the generated brightness levels
    hsv_colors[:, 2] = brightness_levels
    
    # Convert HSV to RGB
    rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
    
    return rgb_colors
#with open(f"baseline_explanations_{dataset_name}.pkl", "rb") as file:
#    explanations = pickle.load(file)
if __name__ == "__main__":

    with open(f"rule_masks_{dataset_name}_node_simple_devide.pkl", "rb") as file:
        rule_masks = pickle.load(file)
    rule_masks = {i: {"INSIDESHAP": rule_masks[i]} for i in range(len(rule_masks))}

    if dataset_name != "Benzen":
        with open(f"baseline_explanations_{dataset_name}.pkl", "rb") as file:
            explanations = pickle.load(file)
    else:
        explanations = rule_masks
        with open(f"subgraph_Benzen.pkl", "rb") as file:
            d1 = pickle.load(file)
        with open(f"Gstar_Benzen.pkl", "rb") as file:
            d2 = pickle.load(file)
        with open("Benzen_svx_aug.pkl", "rb") as file:
            d3 = pickle.load(file)
        d3 = {key.item(): value for key, value in d3.items()}
        with open("baseline_explanations_Benzen_edgshaper.pkl", "rb") as file:
            d4 = pickle.load(file)

        dict1 = [(i, key) for i, key in enumerate(sorted(d1.keys()))]
        dict2 = [(i, key) for i, key in enumerate(sorted(d2.keys()))]
        print(dict1 == dict2)
        

        for i, key in enumerate(sorted(d1.keys())):
            explanations[i]["subgraph"] = d1[key]
            explanations[i]["gstar"] = d2[key]
            explanations[i]["svx"] = d3[key]
            explanations[i]["graphshaper"] = d4[i]['graphshaper']
             

    if dataset_name == "ba2":
        with open(f"baseline_explanations_{dataset_name}_gstar.pkl", "rb") as file:
            explanations_gstar = pickle.load(file)
    if dataset_name in ["ba2", "mutag"]:
        _, _, _, train_mask, _, _ = load_dataset(dataset_name)
        #gt = eval(f"load_{dataset_name}_ground_truth()")
        gt = load_dataset_ground_truth(dataset_name)[0]
        #print(all([len(gt[-1][i]) == gt[0][i].shape[1] for i in range(len(train_mask)) if train_mask[i]]))
        #print(gt[0][0])
        gt = [gt[-1][i] for i in range(len(train_mask)) if train_mask[i]]
        for i in range(len(gt)):
            explanations[i]['ground_truth'] = gt[i]
    train_loader, gnn_model, graphs, features, labels, train_mask = load_dataset_to_explain(dataset_name)
    if dataset_name == "Benzen":
        from graphxai.datasets.real_world.benzene.benzene import Benzene
        ds = Benzene(device="cuda:0")
        gt = ds.get_data_list(torch.where(train_mask)[0])[1]
        for i in range(len(gt)):
            explanations[i]["ground_truth"] = gt[i][0].edge_imp

    with open(f"inside_masks_{dataset_name}.pkl", "rb") as file:
        inside_mask = pickle.load(file)


    if dataset_name !="Benzen":
        for i in explanations.keys():
            explanations[i].update(rule_masks[i])
            if dataset_name == "ba2":
                explanations[i].update(explanations_gstar[i])

    #explainer = "INSIDESHAP"
    #indices = torch.where(train_mask)[0]
    for i, data in tqdm(enumerate(train_loader)):
        flag = False
        if i not in  [730, 260]:
            continue
        if dataset_name != "ba2":
            explanations[i]["inside"] = inside_mask[0]
        else:
            explanations[i]["inside"] = inside_mask[i]
        for explainer in explanations[i].keys():
            #    continue
            #data = train_loader[index]
            if explainer != "subgraph":
                data.node_value = explanations[i][explainer]
            else:
                node_values = torch.zeros(data.x.shape[0])
                node_values[explanations[i][explainer]] = 1
                data.node_value = node_values
            if explainer not in  ["graphshaper", "ground_truth"]:
                if explainer == "svx" and dataset_name in ["Benzen", "AlkaneCarbonyl"]:
                    data.node_value = data.node_value[0][1]
                    if data.node_value.shape[0] > data.x.shape[0]:
                        data.node_value = data.node_value[:data.x.shape[0]]
                    data.node_value = data.node_value.tolist()
                while data.x.shape[0] > len(data.node_value):
                    try:
                        data.node_value.append(0)
                    except AttributeError:
                        data.node_value = torch.concat((data.node_value, torch.zeros(data.x.shape[0] - data.node_value.shape[0])))
                nx_graph = to_networkx(data, node_attrs=["x", "node_value"])
            else:
                t = torch.argmax(torch.where(data.edge_index[0]!= data.edge_index[1])[0])
                data.edge_index = data.edge_index[:, :t+1]
                for j, (u, v) in enumerate(data.edge_index.T):
                    ind = torch.where(torch.logical_and(v == data.edge_index[0], u == data.edge_index[1]))[0].cpu().item()
                    data.node_value[ind] = max(data.node_value[j], data.node_value[ind])
                nx_graph = to_networkx(data, node_attrs=["x"], edge_attrs=["node_value"])
            nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
            nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
            if not flag:
                pos = nx.spring_layout(nx_graph)
                flag = True
            if explainer not in  ["graphshaper", "ground_truth"]:
                colors = {j: color for j, color in nx_graph.nodes(data="node_value")}
                colors = [colors[j] for j in sorted(colors.keys())]
            else:
                colors = {(l,j): color for l,j, color in nx_graph.edges(data="node_value")}
                colors = [colors[(l,j)] for l, j in sorted(colors.keys())]
            sorted_colors = sorted(set(colors))
            colors = [sorted_colors.index(color) for color in colors]
            labels = {node: atoms[dataset_name][torch.argmax(torch.tensor(node_data)).item()] for node, node_data in
                      nx_graph.nodes(data='x')}
            plt.clf()
            if explainer not in  ["graphshaper", "ground_truth"]:
                nx.draw(nx_graph, pos=pos, labels=labels, with_labels=True, node_color=colors, cmap = ListedColormap(generate_colormap(len(sorted_colors))))
            else:
                nx.draw(nx_graph, pos=pos, labels=labels, with_labels=True, edge_color=colors, cmap =ListedColormap(generate_colormap(len(sorted_colors))))
            plt.savefig(f"bseval/figs/{dataset_name}/{explainer}{i}.png")
