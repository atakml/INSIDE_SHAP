import argparse
import torch 
import pickle
from torch_geometric.utils import to_networkx, from_networkx
from patternmod.inside_utils import read_patterns
from patternmod.pattern_evaluation import calculate_pattern_support
from datasetmod.datasetloader import load_dataset_gnn, load_splited_data
from modelmod.gnn import load_gnn
from modelmod.gnn_utils import fill_embeddings
from shapmod.rulemask import find_support
from representationmod.cluster_representation import delete_node
import numpy as np
from sklearn.metrics import roc_auc_score

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
    patterns_to_explain = {
        "ba2": [6, 16, 14, 12],
        "AlkaneCarbonyl": [35, 23, 26, 38],
        "Benzen": [41, 42, 10, 32]
    }
    with open(f"{dataset_name} inside shap explanations {args.model} 1024.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    predictions = predictions.cpu().numpy()   
    
    for i in range(1, len(patterns_to_explain[dataset_name]) + 1):
        remaining_graphs = {idx for idx in range(len(nx_graphs))}
        total_count = len(remaining_graphs)
        cumulative_covered_graphs = 0
        changed_class_count = 0
        predicted_class = {}
        for pattern_index in list(patterns_to_explain[dataset_name])[:i]:
            if dataset_name != "ba2":
                pattern_index += 1
            support_indices = calculate_pattern_support(pattern_dict[pattern_index], embeddings).tolist()
            #activated_nodes = {idx: find_support(embeddings[idx][pattern_dict[pattern_index]["layer"]], pattern_dict[pattern_index]["components"]).tolist() for idx in support_indices if idx in remaining_graphs}
            for idx in  support_indices:
                if idx not in predicted_class:
                    predicted_class[idx] = 0
                t = shap_dict[(idx, pattern_dict[pattern_index]["Target"])][0][pattern_index].item()
                if pattern_dict[pattern_index]["Target"] == 0:
                    t = -t
                predicted_class[idx] += t#shap_dict[(idx, pattern_dict[pattern_index]["Target"])][0][pattern_index].item() if
            

            """for idx in activated_nodes.keys():
                data = from_networkx(nx_graphs[idx])
                features, edge_index = data.x.to(args.device), data.edge_index.to(args.device)
                #new_features, new_edge_index = delete_node(features, edge_index, activated_nodes[idx])
                mask = torch.zeros(features.size(0), dtype=torch.bool, device=args.device)
                mask[activated_nodes[idx]] = True
                edge_mask = mask[edge_index[0]] | mask[edge_index[1]]
                new_edge_index = edge_index[:, edge_mask]
                mask[edge_index[0]] = True
                mask[edge_index[1]] = True
                new_features = features.clone()
                new_features[~mask] = 0
                new_prediction = model(new_features, new_edge_index.reshape(2, -1))[0].argmax(dim=-1).item()
                predicted_class[idx] = new_prediction
                #if new_prediction != predictions[idx]:
                #    predicted_class[idx] = pattern_dict[pattern_index]["Target"]
                #else:
                #    predicted_class[idx] = 1 - pattern_dict[pattern_index]["Target"]
                    #changed_class_count += 1
                    #remaining_graphs.remove(idx)
            #absent_indices = [idx for idx in remaining_graphs if idx in support_indices and predictions[idx] != pattern_dict[pattern_index]["Target"]]
            remaining_graphs -= set(predicted_class.keys())"""
            #changed_class_count += len(absent_indices)
            #percentage = ((changed_class_count) / total_count) * 100
        #for index in remaining_graphs:
        #    predicted_class[index] = 0#predictions[index]
        #majority_class = predictions.mode().values.item()
        #print(f"Majority class: {majority_class}")
        for index in remaining_graphs:
            if index not in predicted_class.keys():
                predicted_class[index] = 0
        #    predicted_class[index] = majority_class
        #percentage = ((changed_class_count) / total_count) * 100
        #print(predicted_class.values())
        #for idx in predicted_class.keys():
        #    if predicted_class[idx]>=0.5:
        #        predicted_class[idx] = 1
        #    else:
        #        predicted_class[idx] = 0 
        scores = np.array([predicted_class[idx] for idx in sorted(predicted_class.keys())])  
         
        percentage =roc_auc_score(predictions, scores)
        #percentage = sum([1 for idx in predicted_class.keys() if predicted_class[idx] == predictions[idx].item()]) / len(predictions)*100
        print(f"Percentage of graphs that changed class after deleting activated nodes: {percentage:.2f}%")