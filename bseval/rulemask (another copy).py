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

def find_support(embeddings, components):
    cols = embeddings[:, components]
    activated_nodes = torch.where(cols.all(axis=1))[0]
    return activated_nodes


def generate_node_values(feature, edge_index, model, shap_values, rule_dict):
    embeddings = model.embeddings(feature, edge_index)
    target_class = torch.argmax(model(feature, edge_index)[0])
    shap_values = shap_values[target_class][0]
    node_values = torch.zeros(feature.shape[0], device="cuda:0")
    for ii, pattern in rule_dict.items():
        i = ii-1
        layer, components, label = pattern['layer'], pattern['components'], pattern['label']

        value = shap_values[ii]
        activated_nodes = find_support(embeddings[layer] if layer < len(embeddings) else feature, components)
        if not activated_nodes.shape[0]:
            continue
        value/= len(activated_nodes)
        node_values[activated_nodes.cpu()] += value.cuda()#/((1+layer) if layer != len(embeddings) else 1)
        node_mask = torch.zeros(feature.shape[0]).bool().cuda()
        node_mask[activated_nodes.cuda()] = True
        if layer == 3:
            layer = -1
            continue
        continue
        for ell in range(layer + 1):
            edge_mask = torch.logical_xor(node_mask[edge_index[0]], node_mask[edge_index[1]]).to(edge_index.device)
            neighbor_list = edge_index[:, edge_mask].reshape(1, -1).unique(sorted=True)
            nodes_to_add = neighbor_list[node_mask[neighbor_list].to(neighbor_list.device) == False]
            if not nodes_to_add.shape[0]:
                break
            node_values[nodes_to_add] += value*(layer - layer+1) / (ell+1)
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
        edge_values[i] = (node_value[v]/deg[v] + node_value[u]/deg[u])/2
    return edge_values

if __name__ == "__main__":

    for dataset in ["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:#["AlkaneCarbonyl", "mutag"]: #["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:# ["mutag", "BBBP", "aids"]:#["AlkaneCarbonyl", "Benzen", "ba2"]:#["ba2", "Benzen"]: #["BBBP", "AlkaneCarbonyl", "aids"]:#["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:   

        with open(f"/home/ata/shap_inside/{dataset} shap explanations gcn.pkl", "rb") as file:
            shap_dict = pickle.load(file)
        #with open(f"indices_to_ignore{dataset}.pkl", "rb") as file:
        #    indices_to_ignore = pickle.load(file)
        rule_path = f"/home/ata/shap_inside/codebase/ExplanationEvaluation/PatternMining/{dataset}_activation_encode_motifs.csv"
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
        below_indices = []
        for i, data in tqdm(enumerate(train_loader)):
            best_hfid = (0,0,0,0) 
            #if i in indices_to_ignore:
            #    continue
            flag = True
            feature, edge_index = data.x, data.edge_index
            negative_indices = torch.where(node_values[i] <=0)[0]
            #shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
            #node_values.append(generate_node_values(feature, edge_index, gnn_model, shap_values, rule_dict))
            #edge_values.append(generate_edge_values_from_node_values(node_values[-1], edge_index))
            for j in list(range(1,5)) + list(range(5, 50, 5)):
                edge_mask, node_mask = calculate_edge_mask(data, node_values[i], j/100)
                #edge_mask, node_mask = calculate_edge_mask(data, edge_values[i], j/100, True)
                if not (node_values[i][negative_indices] >= 0).all():
                    flag = False
                    try:
                        node_mask[negative_indices] = False
                    except:
                        print("!"*100)
                        print(node_mask.shape)
                        print(node_values[i].shape)
                        print(node_values[i][node_mask] <=0)
                        print(node_mask.cpu()[node_values[i][node_mask].cpu() <=0].shape)
                        print("!"*100)
                        raise
                    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                best_hfid = max(h_fidelity(data, gnn_model, edge_mask, node_mask), best_hfid, key=lambda x: x[0])
                flag = False
            hfid += best_hfid
            #if best_hfid[0] <= 0.5:
            #    cnt += 1 
            #    below_indices.append(i)
        print(f"{dataset=} metric: {hfid/len(train_loader)}")
        #print(f"{len(train_loader)=}: {cnt=}")
        #with open(f"rule_mask_{dataset}_below_indices.pkl", "wb") as file:
        #    pickle.dump(below_indices, file)
        #with open(f"rule_masks_{dataset}_node_simple_devide.pkl", "wb") as file:
        #    pickle.dump(node_values, file)

