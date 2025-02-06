import os.path

import numpy as np
import pandas as pd
import json
import numpy
from tqdm import tqdm
import networkx as nx
import torch
from torch_geometric.data import Data
import torch_geometric.utils

from ExplanationEvaluation.PatternMining.Draw_plots import read_rules_from_file
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def compute_support(rule, activation_matrix, graph_inds, graphs, features):
    rule = rule[0]
    active_components = [key for key in rule.keys() if rule[key] == 1]
    deactive_components = [key for key in rule.keys() if rule[key] == 0]
    node_support_indices = np.where(
        (~ np.any(activation_matrix[:, deactive_components], axis=1)) & np.all(
            activation_matrix[:, active_components], axis=1))
    res = []
    for node in node_support_indices[0]:
        graph_id = int(graph_inds[node])
        indices = np.where(graph_inds==graph_id)[0]
        first_index = indices[graph_inds[indices].argmin()]
        node_index = node - first_index
        graph = Data(edge_index=torch.tensor(graphs[graph_id]), x=torch.tensor(features[graph_id]))
        nx_graph = torch_geometric.utils.to_networkx(graph, node_attrs=["x"])
        ego_graph = nx.ego_graph(nx_graph, node_index)
        torch_ego = torch_geometric.utils.from_networkx(ego_graph)
        res.append((graph_id,torch_ego.edge_index.numpy(), torch_ego.x.numpy()))
    '''graph_support_inds = list(map(int, set(graph_inds[node_support_indices[0]])))
    res = [(graphs[i], features[i]) for i in graph_support_inds]'''
    return res


def get_graphs_from_rules(dataset_name, method):
    method = f"_{method}" if method != "mcts" else ""
    return_graphs = dict()
    rules_file = f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name.capitalize()}/{dataset_name.capitalize()}_activation_encode_motifs{method}.csv"
    rules_file = f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}/{dataset_name}_activation_single.csv"
    #rules_file = "Mutag_activation_encode_motifs.csv"
    '''if not os.path.exists(rules_file):
        rules_file = f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}/{dataset_name}_activation_encode_motifs{method}.csv"'''
    dict_layer = read_rules_from_file(rules_file)
    edge_index, features, labels, train_mask, val_mask, test_mask = load_dataset(f"{dataset_name}", shuffle=True)
    activation_path = f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv"
    try:
        main_activation_matrix = pd.read_csv(activation_path)
    except FileNotFoundError:
        dataset_name = dataset_name.capitalize()
        activation_path = f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv"
        main_activation_matrix = pd.read_csv(activation_path)
    graphs_inds = main_activation_matrix["id"]
    for (layer, target), rules in dict_layer.items():
        columns = main_activation_matrix.columns
        columns = list(filter(lambda col: f"l_{layer}c_" in col, columns))
        activation_matrix = main_activation_matrix[columns].to_numpy()
        for rule in rules:
            if (layer, target) not in return_graphs.keys():
                return_graphs[(layer, target)] = [
                    compute_support(rule, activation_matrix, graphs_inds, edge_index, features)]
            else:
                return_graphs[layer, target].append(
                    compute_support(rule, activation_matrix, graphs_inds, edge_index, features))
    return return_graphs


datasets = ["ba2"]
methods = ["mcts"]
for method in tqdm(methods):
    for dataset in tqdm(datasets):
        with open(f"mcts_mask/ba2_mcts.json", "w") as f:
            json.dump({str(key): value for key, value in get_graphs_from_rules(dataset, method).items()}, f,
                      cls=NumpyEncoder)
