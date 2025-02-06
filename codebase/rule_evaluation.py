import numpy as np
import pandas as pd
import torch
from numpy.random.mtrand     import RandomState
from scipy.special import softmax
from copy import copy, deepcopy
import ast
from tqdm import tqdm

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.models.GNN_paper import GraphGCN
from numpy import genfromtxt
from ExplanationEvaluation.PatternMining.utiles import read_rules_from_file, read_from_mcts_files, \
    read_from_beam_search_files


class MetricBase:
    def __init__(self, gnn, features, graphs, activation_matrix, graphs_indices, labels):
        '''
        graphs: edge_indices
        '''
        self.gnn = gnn
        self.graphs = graphs
        self.activation_matrix = activation_matrix
        self.graph_indices = graphs_indices
        self.labels = labels
        self.intervals = []
        self._build_intervals()
        self.features = features

    def _build_intervals(self):
        current_ind = self.graph_indices[0]
        start = 0
        for i in range(len(self.graph_indices)):
            if self.graph_indices[i] != current_ind:
                self.intervals.append((start, i))
                current_ind = self.graph_indices[i]
                start = i
        self.intervals.append((start, len(self.graph_indices)))

    def find_interval_index(self, node_index):
        start, end = 0, len(self.intervals)
        while start != end - 1:
            mid = (start + end) // 2
            interval = self.intervals[mid]
            if interval[0] <= node_index < interval[1]:
                return mid
            elif node_index < interval[0]:
                end = mid
            elif node_index >= interval[1]:
                start = mid + 1
        return start

    def get_neighbors(self, nodes, edge_indices):
        neighbors = []
        for node in nodes:
            for edge in edge_indices.transpose():
                if node == edge[0]:
                    neighbors.append(edge[1])
                elif node == edge[1]:
                    neighbors.append(edge[0])
        return set(neighbors)

    def compute_active_deactive_support(self, rule):
        active_components = [key for key in rule.keys() if rule[key] == 1]
        deactive_components = [key for key in rule.keys() if rule[key] == 0]
        node_support_indices = np.where(
            (~ np.any(self.activation_matrix[:, deactive_components], axis=1)) & np.all(
                self.activation_matrix[:, active_components], axis=1))

        if self.__class__.__name__ == "Infidelity":
            new_graphs, graph_index_dict = self.build_new_graphs(
                node_support_indices)
            sparsity = 0
            for i in range(len(self.graphs)):
                if i in graph_index_dict:
                    edge_trans = self.graphs[i].transpose()
                    number_of_nodes = np.max(edge_trans[np.all(edge_trans - np.fliplr(edge_trans), axis=1)]) + 1
                    sparsity += 1 - len(graph_index_dict[i]) / number_of_nodes
                    if number_of_nodes < len(graph_index_dict[i]):
                        print()
            sparsity /= len(new_graphs)
            return active_components, deactive_components, node_support_indices, new_graphs, sparsity
        else:
            new_graphs = self.build_new_graphs(
                node_support_indices)
            return active_components, deactive_components, node_support_indices, new_graphs

    def compute(self, rule):
        if self.__class__.__name__ == "Infidelity":
            active_components, deactive_components, node_support_indices, new_graphs, sparsity = self.compute_active_deactive_support(
                rule)
        else:
            active_components, deactive_components, node_support_indices, new_graphs = self.compute_active_deactive_support(
                rule)
        score = 0
        t = 0
        for i in range(len(new_graphs)):
            try:
                new_pred = softmax(self.gnn(torch.tensor(self.features[i]), torch.tensor(new_graphs[i])).detach().numpy())[
                0, 1]
            except:
                print()
            pred = softmax(self.gnn(torch.tensor(self.features[i]), torch.tensor(self.graphs[i])).detach().numpy())[
                0, 1]
            # print(new_pred, pred)
            score += int((np.array(new_pred) > 0.5) != (np.array(pred) > 0.5))
        score /= len(new_graphs)
        if self.__class__.__name__ == "Infidelity":
            return score, sparsity
        else:
            return score

    def build_new_graphs(self, node_support_indices):
        pass

    def build_graph_dict(self, nodes):
        graph_index_dict = {}
        for node in nodes[0]:
            graph_index = self.find_interval_index(node)
            new_index = node - self.intervals[graph_index][0]
            if graph_index not in graph_index_dict.keys():
                graph_index_dict[graph_index] = [new_index]
            else:
                graph_index_dict[graph_index].append(new_index)
        return graph_index_dict


class Fidelity(MetricBase):

    def build_new_graphs(self, node_support_indices):
        # just we delete all edges of those nodes and self loops
        new_graphs = copy(self.graphs)
        graph_index_dict = self.build_graph_dict(node_support_indices)
        for graph in graph_index_dict:
            new_graphs[graph] = new_graphs[graph].transpose()[
                ~np.any(np.isin(new_graphs[graph].transpose(), graph_index_dict[graph]), axis=1)
            ].transpose()
            for node in graph_index_dict[graph]:
                if [node, node] not in new_graphs[graph].transpose():
                    new_graphs[graph] = np.append(new_graphs[graph], [[node], [node]], axis=1)
        return new_graphs


class Infidelity(MetricBase):
    def build_new_graphs(self, node_support_indices):
        new_graphs = copy(self.graphs)
        number_of_nodes = max(new_graphs[0][0])
        graph_index_dict = self.build_graph_dict(node_support_indices)
        for graph in graph_index_dict:
            new_graphs[graph] = new_graphs[graph].transpose()[
                np.all(np.isin(new_graphs[graph].transpose(), graph_index_dict[graph]), axis=1)
            ].transpose()
            if not new_graphs[graph].shape[1]:
                new_graphs[graph] = np.array([np.arange(0, number_of_nodes + 1), np.arange(0, number_of_nodes + 1)])
            else:
                start = max(max(new_graphs[graph][0]), max(new_graphs[graph][1]))
                for i in range(start + 1, number_of_nodes + 1):
                    if [i, i] not in new_graphs[graph].transpose():
                        new_graphs[graph] = np.append(new_graphs[graph], [[i], [i]], axis=1)
        return new_graphs, graph_index_dict

def shuffle(arr, permutation):
    try:
        return arr[permutation]
    except:
        return [arr[i] for i in permutation]

dataset_name = "aids"
model = model_selector("GNN", f"{dataset_name}")

path = f'/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/models/pretrained/GNN/{dataset_name}/best_model'
edge_index, features, labels, train_mask, val_mask, test_mask = load_dataset(f"{dataset_name}", shuffle=True)
'''prng = RandomState(42)# Make sure that the permutation is always the same, even if we set the seed different
indices = list(range(len(labels)))
indices = prng.permutation(indices)
edge_index, features, labels = shuffle(edge_index, indices), shuffle(features,indices), shuffle(labels,indices)'''
activation_path = f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv"
try:
    main_activation_matrix = pd.read_csv(activation_path)
except FileNotFoundError:
    dataset_name = dataset_name.capitalize()
    activation_path = f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv"
    main_activation_matrix = pd.read_csv(activation_path)

graphs_inds = main_activation_matrix["id"]
'''for layer in range(3):
    columns = main_activation_matrix.columns
    columns = list(filter(lambda col: f"l_{layer}c_" in col, columns))
    activation_matrix = main_activation_matrix[columns].to_numpy()
    fid = Fidelity(model, features, edge_index, activation_matrix, graphs_inds, labels)
    infid_and_spars = Infidelity(model, features, edge_index, activation_matrix, graphs_inds, labels)
    # patterns = read_rules_from_file("/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/PatternMining/Aids_activation.out")
    #patterns = read_from_beam_search_files(
        #f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_l{layer}_beam.csv")
    save_file = f"fid_infid_spars_beam_{dataset_name}_l{layer}.csv"
    with open(save_file, "w") as f:
        f.write("pattern,score,fidelity,infidelity,sparsity\n")
        for p in tqdm(patterns.iterrows()):
            pattern = p[1]
            #rule = ast.literal_eval(pattern["pattern"])
            rule = pattern["pattern"]
            score = pattern["score"]
            #iteration = pattern["iterations"]
            fidel = fid.compute(rule)
            infid, sparsity = infid_and_spars.compute(rule)
            rule_text = " ".join([f"l0_c{key}" for key in rule.keys()])
            f.write(f"{rule_text},{score},{fidel},{infid},{sparsity}\n")
            '''


input_file = f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/PatternMining/Aids_activation_encode_motifs.csv"
dict_layer = read_rules_from_file(input_file)
save_file = f"fid_infid_spars_{dataset_name}_exhaustive.csv"
with open(save_file, "w") as f:
    f.write("pattern,score,fidelity,infidelity,sparsity\n")
    for (layer, target), rules in dict_layer.items():
        columns = main_activation_matrix.columns
        columns = list(filter(lambda col: f"l_{layer}c_" in col, columns))
        activation_matrix = main_activation_matrix[columns].to_numpy()
        fid = Fidelity(model, features, edge_index, activation_matrix, graphs_inds, labels)
        infid_and_spars = Infidelity(model, features, edge_index, activation_matrix, graphs_inds, labels)
        for pattern, score in tqdm(rules):
            #fidel = fid.compute(pattern)
            fidel = 0
            infid, sparsity = infid_and_spars.compute(pattern)
            rule_text = " ".join([f"l0_c{key}" for key in pattern.keys()])
            f.write(f"{rule_text},{score},{fidel},{infid},{sparsity}\n")
