import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0\
).total_memory/1e9)
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)
import pickle
import pandas as pd
from bseval.utils import load_dataset_to_explain, calculate_edge_mask, h_fidelity
from tqdm import tqdm
from math import log
from surrogatemod.surrogate_utils import select_best_model, load_gin_dataset, input_feature_size
from patternmod.inside_utils import read_patterns
from patternmod.diffversify_utils import *
from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
import argparse
from concurrent.futures import ThreadPoolExecutor

def find_support(embeddings, components):
    cols = embeddings[:, components]
    activated_nodes = torch.where(cols.all(axis=1))[0]
    return activated_nodes


def generate_node_values(feature, edge_index, model, shap_values, rule_dict, atom_rules=False, device="cuda:0"):
    embeddings = model.embeddings(feature, edge_index)
    target_class = torch.argmax(model(feature, edge_index)[0])
    shap_values = shap_values[target_class][0]
    node_values = torch.zeros(feature.shape[0], device=device)
    for ii, pattern in enumerate(rule_dict):

        layer, components, label = pattern['layer'], pattern['components'], pattern['Target']
        value = shap_values[ii]
        activated_nodes = find_support(embeddings[layer] if layer < len(embeddings) else feature, components)
        if layer == 3:
            layer = -1
            if not atom_rules:
                continue
        if not activated_nodes.shape[0]:
            continue
        value /= len(activated_nodes)
        node_values[activated_nodes.cpu()] += value.to(device)
        node_mask = torch.zeros(feature.shape[0]).bool().to(device)
        node_mask[activated_nodes.to(device)] = True
        continue
        for ell in range(layer - layer + 1):
            edge_mask = torch.logical_xor(node_mask[edge_index[0]], node_mask[edge_index[1]]).to(edge_index.device)
            neighbor_list = edge_index[:, edge_mask].reshape(1, -1).unique(sorted=True)
            nodes_to_add = neighbor_list[node_mask[neighbor_list].to(neighbor_list.device) == False]
            if not nodes_to_add.shape[0]:
                break
            node_values[nodes_to_add] += value * (layer - layer + 1) / (ell + 1)
            node_mask[nodes_to_add] = True
    return node_values


def generate_edge_values_from_node_values(node_value, edge_index):
    edge_values = torch.zeros(edge_index.shape[-1])
    deg = {}
    for i, (v, u) in enumerate(edge_index.T): 
        if v not in deg.keys():
            deg[v] = (edge_index[0] == v).sum() + (edge_index[1] == v).sum()
        if u not in deg.keys():
            deg[u] = (edge_index[0] == u).sum() + (edge_index[1] == u).sum()
        edge_values[i] = (node_value[v]/deg[v] + node_value[u]/deg[u])
    return edge_values

def process_data(i, data, shap_dict, gnn_model, rule_dict, args, node_values, hfid, below_indices, atom_rules):
    best_hfid = torch.tensor((0,0,0,0)).float()
    #if i in indices_to_ignore:
    #    continue
    flag = True
    feature, edge_index = data.x, data.edge_index
    shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
    node_value = generate_node_values(feature, edge_index, gnn_model, shap_values, rule_dict, atom_rules=atom_rules)
    negative_indices = torch.where(node_value <=0)[0]
    #shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
    #
    #edge_values.append(generate_edge_values_from_node_values(node_values[-1], edge_index))
    for j in list(range(1,5)) + list(range(5, 50, 5)):
        edge_mask, node_mask = calculate_edge_mask(data, node_value, j/100)
        #edge_mask, node_mask = calculate_edge_mask(data, edge_values[i], j/100, True)
        if args.remove_negative:
            if not (node_value[negative_indices] >= 0).all():
                flag = False
                try:
                    node_mask[negative_indices] = False
                except:
                    print("!"*100)
                    print(node_mask.shape)
                    print(node_value.shape)
                    print(node_value[node_mask] <= 0)
                    print(node_mask.cpu()[node_value[node_mask].cpu() <= 0].shape)
                    print("!"*100)
                    raise
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        best_hfid = max(h_fidelity(data, gnn_model, edge_mask, node_mask), best_hfid, key=lambda x: x[0])
        flag = False
    return node_value, best_hfid, flag, i
        #if best_hfid[0] <= 0.5:
        #    cnt += 1 
        #    below_indices.append(i)
    #print(f"{len(train_loader)=}: {cnt=}")
    #with open(f"rule_mask_{dataset}_below_indices.pkl", "wb") as file:
    #    pickle.dump(below_indices, file)
    #with open(f"rule_masks_{dataset}_node_simple_devide.pkl", "wb") as file:
    #    pickle.dump(node_values, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--model', type=str, default='gcn', help='Model to use (default: gnn)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda)')
    parser.add_argument('--remove_negative', action='store_true', help='Remove negative node values')
    parser.add_argument('--mean', action='store_true', help='Use mean shap explanations')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model
    device = args.device
    dataset = dataset_name
    mean_str = " mean" if args.mean else ""
    with open(f"{dataset} inside shap explanations {model_name}{mean_str} 1024.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    #with open(f"indices_to_ignore{dataset}.pkl", "rb") as file:
    #    indices_to_ignore = pickle.load(file)
    rule_dict = read_patterns(dataset)# if args.method == "inside" else convert_patterns_to_dict(load_diffversify_patterns(dataset_name))
    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset, device=device)
    gnn_model = load_gnn(dataset)
    #with open(f"rule_masks_{dataset}_node_simple_devide.pkl", "rb") as file:
    #    node_values = pickle.load(file)
    node_values = []
    edge_values = []
    best_hfid = torch.tensor((0,0,0,0)).float()

    cnt = 0 
    below_indices = []
    for atom_rules in [False, True]:
        with ThreadPoolExecutor() as executor:
            futures = []
            hfid = torch.tensor((0,0,0,0)).float()
            for i, data in enumerate(train_loader):
                futures.append(executor.submit(process_data, i, data, shap_dict, gnn_model, rule_dict, args, node_values, hfid, below_indices, atom_rules))
            for future in tqdm(futures):
                result = future.result()
                node_values.append(result[0])
                hfid += result[1]
                if result[2]:
                    below_indices.append(result[3])
        best_hfid = max(hfid,best_hfid, key=lambda x: x[0])
    print(f"{dataset=} metric: {best_hfid/len(train_loader)}")

