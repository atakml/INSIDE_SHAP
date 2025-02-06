import pickle

import numpy as np
import numpy
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
from torch_geometric.utils.map import map_index
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from codebase.GINUtil import GinDataset
from ExplanationEvaluation.tasks.replication import get_classification_task, select_explainer, to_torch_graph
from ExplanationEvaluation.models.model_selector import model_selector
from codebase.ExplanationEvaluation.PatternMining.utiles import build_model

target_count = [0, 0]

maxent_probs, indices = [], None
dataset_name = "aids"

for layer in range(4 if dataset_name != 'ba2' else 3):
    maxent_probs.append([])
    for label in range(2):
        model, _, _, _, indices, _, _, _, _, _, _ = build_model(
            f"/home/ata/shap_inside/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv", layer,
            label)
        maxent_probs[-1].append(model.proba)


def calculate_information_content(pattern, graph_index, gnn_model, feature_matrix, edge_index):
    global maxent_probs, indices
    layer, components = pattern['layer'], pattern['components']
    graph_index = np.where(indices == graph_index)
    embeddings = gnn_model.embeddings(feature_matrix, edge_index)
    layer_embedding = embeddings[layer] if layer < len(embeddings) else feature_matrix.clone().detach()
    mask = torch.sgn(layer_embedding)[:, components].all(dim=1)
    support_ic = np.average(
        -np.log(maxent_probs[layer][pattern['class']][graph_index][mask][:, components.tolist()]).sum(
            axis=1)) if mask.any() else 0
    out_support_ic = np.average(
        -np.log(maxent_probs[layer][pattern['class']][graph_index][~mask][:, components.tolist()]).sum(
            axis=1)) if not mask.all() else 0
    information_content = support_ic #+ out_support_ic

    return information_content


def is_active(feature_matrix, masked_feature_matrix):
    return not feature_matrix.shape == masked_feature_matrix.shape


def ego_network_mask(center_mask, radius, edge_index):
    final_mask = center_mask.clone().detach()
    for _ in range(radius):
        edge_mask = final_mask[edge_index[0]] | final_mask[edge_index[1]]
        final_mask[edge_index[0][edge_mask]] = 1
        final_mask[edge_index[1][edge_mask]] = 1
    return final_mask


def generate_masked_features_for_pattern(feature_matrix, edge_index, pattern, gnn_model):
    layer, components = pattern['layer'], pattern['components']
    embeddings = gnn_model.embeddings(feature_matrix, edge_index)
    layer_embedding = embeddings[layer] if layer < len(embeddings) else feature_matrix.clone().detach()
    # masked_feature_matrix = feature_matrix.clone().detach()
    mask = torch.sgn(layer_embedding)[:, components].all(dim=1)
    if not mask.any() or layer == len(embeddings
                                      ):
        return None, None
    final_mask = ego_network_mask(mask, (layer + 1) % (len(embeddings) + 1), edge_index)
    # masked_feature_matrix[final_mask] = 0
    return remove_mask(~final_mask, feature_matrix, edge_index)


def calculate_non_activated_score(target_class, pattern):
    global target_count
    covers = pattern['covers']
    total_cover = sum(covers)
    return (total_cover / (target_count[0] + target_count[1])) * (
            covers[target_class] / total_cover - target_count[target_class] / sum(target_count))


def remove_mask(mask, feature_matrix, edge_index):
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask].clone()
    remaining_nodes = torch.arange(mask.shape[0])[mask]

    feature_matrix = feature_matrix[mask].clone()
    edge_index = torch.stack(
        (map_index(edge_index[0], remaining_nodes)[0], map_index(edge_index[1], remaining_nodes)[0]))
    return feature_matrix, edge_index


def compute_rule_fidelity(data_loader, shap_dict, gnn_model, pattern_set):
    """
    :param data_loader: Dataloader containing original feature matrix, edge_index and labels
    :param shap_dict: The dictionary containing rule contributions to the decision of the model for each instance
    :param gnn_model: The GNN to explain
    :param pattern_set: The list of activation patterns. Each pattern is a dict with two keys of
    the layer and components. layer is an int and components is the set
    :return:
    """
    fid_list = []
    for i, data in enumerate(data_loader):
        edge_index, features, labels = data
        # edge_index, features = edge_index[0], features[0].float()
        original_model_probs = torch.softmax(gnn_model(features, edge_index), -1)[0]
        target_class = torch.argmax(original_model_probs)
        if target_class == 3:
            fid_list.append([-10000] * 17)
            continue
        target_class_prob = original_model_probs[target_class]
        pattern_contributions = shap_dict[i]
        fid_list.append([])
        for pattern_index in range(pattern_contributions.shape[-1]):
            # masked_features, masked_edge_index = generate_masked_features_for_pattern(features, edge_index,
            #                                                                          pattern_set[pattern_index],
            #                                                                          gnn_model)
            # if masked_features is not None and is_active(masked_features, features):
            #    masked_prob = torch.softmax(gnn_model(masked_features, masked_edge_index), -1)[0][target_class]
            #    fid_list[-1].append(torch.abs(masked_prob - target_class_prob))
            # else:
            # score = calculate_non_activated_score(pattern_set[pattern_index]['class'], pattern_set[pattern_index])
            #   score = -10
            # if target_class == pattern_set[pattern_index]['class']:
            #    score = -score
            #  fid_list[-1].append(score)
            fid_list[-1].append(
                calculate_information_content(pattern_set[pattern_index], i, gnn_model, features, edge_index))
    return torch.tensor(fid_list)


def calculate_coefficient_for_activated(fid_matrix, shap_dict):
    # indices = torch.where(fid_matrix != -10)
    # if len(indices[0]) < 2:
    #    return np.array((1., 0.))
    # res = spearmanr(fid_matrix[indices], shap_dict[indices])
    res = spearmanr(fid_matrix, shap_dict)
    # if res[0] is numpy.nan:
    #    return np.array((0., 1.))
    return res


def calculate_find_contrib_pc(data_loader, shap_dict, gnn_model, pattern_set):
    """
    :param data_loader: Dataloader containing original feature matrix, edge_index and labels
    :param shap_dict: The dictionary containing rule contributions to the decision of the model for each instance
    :param gnn_model: The GNN to explain
    :param pattern_set: The list of activation patterns. Each pattern is a dict with two keys of
    the layer and components. layer is an int and components is the set.
    :return:
    """
    fid_matrix = compute_rule_fidelity(data_loader, shap_dict, gnn_model, pattern_set)
    coefficients = torch.tensor(
        [calculate_coefficient_for_activated(fid_matrix[i], shap_dict[i][0]) for i in range(len(data_loader)) if
         fid_matrix[i][0] > -10000])
    return coefficients


def prepare_dataset_to_evaluate(dataset_name):
    global target_count
    graphs, features, labels, train_mask, _, test_mask = load_dataset(dataset_name)
    task = get_classification_task(graphs)
    features = torch.tensor(features)
    graphs = to_torch_graph(graphs, task)
    data_loader = GinDataset(graphs, features, labels, train_mask)
    target_count[1] = labels[train_mask].sum()
    target_count[0] = len(train_mask) - target_count[1]
    gnn_model, _ = model_selector("GNN",
                                  dataset_name,
                                  pretrained=True,
                                  return_checkpoint=True)
    gnn_model.eval()
    pattern_set = []
    rule_path = f"/home/ata/shap_inside/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation_encode_motifs.csv"
    rules = pd.read_csv(rule_path, delimiter="=")
    for i, row in rules.iterrows():
        pattern = row[1].strip().split()
        info = row[0].split(":")
        positive_cover = int(info[2][:info[2].index('c')].strip())
        negative_cover = int(info[3][:info[3].index('s')].strip())
        class_index = int(info[1][0])
        layer_index = int(pattern[0][2:pattern[0].index('c')])
        components = torch.tensor(list(map(lambda x: int(x[x.index('c') + 2:]), pattern)))
        pattern_set.append({'layer': layer_index, 'components': components, "class": class_index,
                            "covers": (negative_cover, positive_cover)})
    with open(f"{dataset_name} shap explanations baseline on train.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    t = calculate_find_contrib_pc(data_loader, shap_dict, gnn_model, pattern_set)
    print(torch.abs(t[:, 0]).mean())
    print(torch.abs(t[:, 1]).mean())
    print("P-value")
    print(torch.abs(t[labels[train_mask][:, 0].astype(bool)][:, 1]).mean())
    print(torch.abs(t[labels[train_mask][:, 1].astype(bool)][:, 1]).mean())
    print("Coefficient")
    print(torch.abs(t[labels[train_mask][:, 0].astype(bool)][:, 0].mean()))
    print(torch.abs(t[labels[train_mask][:, 1].astype(bool)][:, 0]).mean())
    print(torch.sum(torch.abs(t[labels[train_mask][:, 1].astype(bool)][:, 0]) >= 0.26) /
          t[labels[train_mask][:, 1].astype(bool)].shape[0])
    # print(torch.sum(t[:, 1] < 0.102) / t.shape[0])
    print("!!!!!!!!!!!!!")
    print(t)
    return t


prepare_dataset_to_evaluate(dataset_name)
