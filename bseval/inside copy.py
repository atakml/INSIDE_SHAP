import torch
from torch_geometric.utils import to_networkx, to_dgl
import networkx as nx
import pickle
import pandas as pd
from bseval.utils import load_dataset_to_explain, calculate_edge_mask, h_fidelity
from tqdm import tqdm
from math import log
from modelmod.gnn import load_gnn
from datasetmode.datasetloader import load_dataset_gnn



def find_support(embeddings, components, debug=False):
    cols = embeddings[:, components]
    activated_nodes = torch.where(cols.all(axis=1))[0]
    if debug:
        print(f"{embeddings=}")
        print(f"{components=}")
        print(f"{activated_nodes=}")
    return activated_nodes


def generate_node_values(feature, edge_index, model, shap_values, rule_dict):
    embeddings = model.embeddings(feature.to("cuda:0"), edge_index.to("cuda:0"))
   # target_class = torch.argmax(model(feature.to("cuda:0"), edge_index.to("cuda:0"))[0])
    #shap_values = shap_values[target_class][0]
    node_values = torch.zeros(feature.shape[0], device="cuda:0")
    for ii, pattern in rule_dict.items():
        i = ii-1
        layer, components, label = pattern['layer'], pattern['components'], pattern['label']

        #value = shap_values[ii]
        activated_nodes = find_support(embeddings[layer] if layer < len(embeddings) else feature, components)
        if not activated_nodes.shape[0]:
            continue
        node_values[activated_nodes.cpu()] += 1 
    return node_values


def generate_edge_values_from_node_values(node_value, edge_index):
    edge_values = torch.zeros(edge_index.shape[-1])
    for i, (v, u) in enumerate(edge_index.T): 
        edge_values[i] = (node_value[v] + node_value[u])/2
    return edge_values


for dataset in["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:#["mutag", "BBBP", "aids"]:#["AlkaneCarbonyl", "Benzen", "ba2"]:#["ba2", "Benzen"]: #["BBBP", "AlkaneCarbonyl", "aids"]:#["ba2", "aids", "BBBP", "AlkaneCarbonyl", "Benzen", "mutag"]:   
    cnt = 0 
    indices_to_ignore = []
    with open(f"/home/ata/shap_inside/{dataset} shap explanations gcn.pkl", "rb") as file:
        shap_dict = pickle.load(file)
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
    gnn_model = gnn_model.to("cuda:0")
    #with open(f"rule_masks_{dataset}.pkl", "rb") as file:
    #    node_values = pickle.load(file)
    node_values = []
    edge_values = []
    hfid = 0
    #with open(f"rule_mask_{dataset}_below_indices.pkl", "rb") as file:
    #    indices_below = pickle.load(file)
    for i, data in tqdm(enumerate(train_loader)):
        best_hfid = (-1,-1,-1,-1) 
        feature, edge_index = data.x, data.edge_index.cuda()
        shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
        selfloop_indices = [i for i in range(edge_index.shape[1]) if edge_index[0][i] == edge_index[1][i]]
        original_nx_graph = to_networkx(data)
        original_nx_graph.remove_edges_from(nx.selfloop_edges(original_nx_graph))
        original_nx_graph.remove_nodes_from(list(nx.isolates(original_nx_graph)))
        number_of_nodes = original_nx_graph.number_of_nodes()
        isolated_nodes = [i for i in range(number_of_nodes, data.x.shape[0])]
        for key, value in rule_dict.items():
            node_mask = generate_node_values(feature, edge_index, gnn_model, shap_values, {key:value}).bool()
            edges = node_mask[edge_index[0]] | node_mask[edge_index[1]]
            node_mask[edge_index[:, edges].flatten()] = True
            if not sum(node_mask):
                continue
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            edge_mask[selfloop_indices] = False
            node_mask[isolated_nodes] = False
            try:
                hf = h_fidelity(data.to("cuda:0"), gnn_model, edge_mask.to("cuda:0"), node_mask.to("cuda:0"))
                if best_hfid[0] < hf[0]:
                    best_hfid = hf
                    best_mask = node_mask.clone()
                elif False and best_hfid[1] == 0:
                    print(hf)
                    print(edges.sum())
                    print(generate_node_values(feature, edge_index, gnn_model, shap_values, {key:value}).bool().sum())
                    print(node_mask.sum(), key)

                #best_hfid = max(h_fidelity(data.to("cuda:0"), gnn_model, edge_mask.to("cuda:0"), node_mask.to("cuda:0")), best_hfid, key=lambda x: x[0])
            except:
                embeddings = gnn_model.embeddings(feature.to("cuda:0"), edge_index.to("cuda:0"))
                for ii, pattern in {key: value}.items():
                    layer, components, label = pattern['layer'], pattern['components'], pattern['label']
                    print(f"{layer=}")
                    find_support(embeddings[layer] if layer < len(embeddings) else feature, components, True)

                raise
            if best_hfid[0] > 1:
                print(best_hfid, i)
                assert False
        #if (best_hfid[-1] < 0.001 or best_hfid[-1] > 0.999): 
        #    cnt += 1
        #    indices_to_ignore.append(i)
        #    continue
        
        hfid += best_hfid
        node_values.append(best_mask)

    #print(f"{dataset=} {hfid/len(indices_below)=}")
    print(f"{dataset=} metric: {hfid/(len(train_loader))}")
    with open(f"indices_to_ignore{dataset}.pkl", "wb") as file:
        pickle.dump(indices_to_ignore, file)
   # with open(f"inside_masks_{dataset}.pkl", "wb") as file:
        #pickle.dump(node_values, file)

