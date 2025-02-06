from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
from patternmod.inside_utils import read_patterns
from representationmod.cluster_representation import generate_colormap, atoms
from shapmod.rulemask import generate_node_values, find_support
from torch_geometric.utils import to_networkx
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import torch
from matplotlib.colors import ListedColormap
import os
from bseval.utils import h_fidelity
from tqdm import tqdm

def visualize_graph(dataset_name, data, output_dir, i, method, save_name= None, text=None, num_colors=None):
    if save_name is None:
        save_name = f"{method}_{i}.png"
    if output_dir is None:
        output_dir = f"bseval/figs/{dataset_name}/"
    node_values = data.node_values
    indexing = False #if num_colors is not None else True
    #num_colors = num_colors if num_colors is not None else len(set(node_values.tolist()))
    colormap = plt.cm.Blues#ListedColormap(generate_colormap(num_colors))
    nx_graph = to_networkx(data, node_attrs=["x", "node_values"])
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
    colors = {j: color for j, color in nx_graph.nodes(data="node_values")}
    colors = [colors[j] for j in sorted(colors.keys())]
    sorted_colors = sorted(set(colors))
    if indexing:
        colors = [sorted_colors.index(color) for color in colors]
    min_color, max_color = min(colors), max(colors)
    if min_color == max_color:
        colors = [1 for _ in colors]
    else:
        colors = list(map(lambda x: (x - min_color) / (max_color - min_color), colors))
    labels = {node: atoms[dataset_name][torch.argmax(torch.tensor(node_data)).item()] for node, node_data in
                        nx_graph.nodes(data='x')}
    plt.clf()
    if text is not None:
        plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
    #pos = nx.spring_layout(nx_graph)
    pos = nx.kamada_kawai_layout(nx_graph)
    nx.draw(nx_graph, pos, node_color=colors, cmap=colormap, with_labels=True if dataset_name != "ba2" else False, labels=labels, node_size=300, font_size=10, vmin=0, vmax=1)
    plt.savefig(f"{output_dir}/fail_{save_name}")
    plt.close()


def generate_inside_mask(data, model, pattern):
    feature, edge_index = data.x, data.edge_index
    activated_nodes = find_support(model.embeddings(feature, edge_index)[pattern['layer']] if pattern['layer'] < 3 else feature, pattern["components"])
    if not activated_nodes.shape[0]:
        return None, None
    node_mask = torch.zeros(feature.shape[0], device="cuda:0", dtype=bool)
    node_mask[activated_nodes] = True
    edges = node_mask[edge_index[0]] | node_mask[edge_index[1]]
    node_mask[edge_index[:, edges].flatten()] = True
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    return node_mask, edge_mask


def visualize_shap_inside(dataset_name, model, pattern_dict, shap_dict, output_dir, i, data):
    feature, edge_index = data.x, data.edge_index
    shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
    node_value = generate_node_values(feature, edge_index, model, shap_values, pattern_dict)
    data.node_values = node_value
    print(data.node_values)
    visualize_graph(dataset_name, data, output_dir, i, "shap_inside")


def visualize_inside(dataset_name, model, pattern_dict, i, data):
    best_pattern, best_hfid = None, 0
    sparse = None
    for cnt, pattern in enumerate(pattern_dict):
        node_mask, edge_mask = generate_inside_mask(data, model, pattern)
        if node_mask is None:
            continue
        hfid = h_fidelity(data, model, edge_mask, node_mask)
        #print(cnt,hfid, best_hfid)

        if hfid[0] > best_hfid and hfid[-1] > 0:
            best_hfid = hfid[0]
            best_pattern = pattern
            sparse = hfid[-1]
        
    
    node_values = torch.zeros(feature.shape[0], device="cuda:0")
    node_mask, edge_mask = generate_inside_mask(data, model, best_pattern)
    node_values[node_mask] = 1
    data.node_values = node_values
    if  sparse < 0.5:# or best_hfid[-1] > 0.99:
        print("!"*10000)
        visualize_graph(dataset_name, data, output_dir, i, "inside")
        return True
    else:
        #print(best_hfid)
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--model', type=str, default='gcn', help='Model to use (default: gnn)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda)')
    parser.add_argument('--i', type=int, default=None, help='Device to use (default: cuda)')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    
    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name)    
    model = load_gnn(dataset_name, args.device)
    pattern_dict = read_patterns(dataset_name)
    with open(f"{dataset_name} inside shap explanations {args.model}.pkl", "rb") as file:
        shap_dict = pickle.load(file)


    output_dir = f"bseval/figs/{dataset_name}/new"
    os.makedirs(output_dir, exist_ok=True)
    failed_indices = []
    for i, data in tqdm(enumerate(train_loader)):
        """if args.i is not None and i!= args.i:
            continue"""
        feature, edge_index = data.x, data.edge_index
        try:
            if visualize_inside(dataset_name, model, pattern_dict, i, data):
                visualize_shap_inside(dataset_name, model, pattern_dict, shap_dict, output_dir, i, data)
                failed_indices.append(i)
        except:
            continue
    print(failed_indices)        