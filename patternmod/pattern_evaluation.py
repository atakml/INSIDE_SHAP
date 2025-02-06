import numpy as np 
import torch
from datasetmod.datasetloader import load_splited_data
from modelmod.gnn_utils import fill_embeddings
from modelmod.gnn import load_gnn
from patternmod.diffversify_utils import *
from patternmod.inside_utils import *
from tqdm import tqdm
def calculate_graph_support(dataset_name, feature, embedding, pattern):
    #current_layer = 0
    #if 'type' not in pattern[0].keys() or 'diff' not in pattern[0]['type']:
    #    current_embedding = embedding[current_layer]
    #else:
    #    current_embedding = torch.hstack(embedding)

    feature_pattern_start = 0 
    if 'type' not in pattern[0].keys() or 'diff' not in pattern[0]['type']:
        while pattern[feature_pattern_start]['layer'] != len(embedding):
            feature_pattern_start += 1
            if feature_pattern_start == len(pattern):
                break
        support_indices = list(filter(lambda index: embedding[pattern[index]['layer']][:, pattern[index]['components']].all(dim=1).any(), range(feature_pattern_start)))
        support_indices += list(filter(lambda index: feature[:, pattern[index]['components']].all(dim=1).any(), range(feature_pattern_start, len(pattern))))
    else:
        embeddings = torch.concat(embedding, dim=1)
        if dataset_name != "ba2":
            embeddings = torch.concat((feature, embeddings), dim=1)
        try:
            support_indices = list(filter(lambda index: embeddings[:, pattern[index]['components']].all(dim=1).any(), range(len(pattern))))
        except:
            print(embeddings.shape)
            print(len(pattern))
            for i in range(len(pattern)):
                if max(pattern[i]['components']) >= embeddings.shape[1]:
                    print(i, pattern)
            raise
    return support_indices


def calculate_pattern_support(pattern, embeddings):
    layer = pattern['layer']
    em = torch.stack(list(map(lambda embedding: embedding[layer], embeddings)))
    print(em.shape)
    return torch.where(em[:, :, pattern['components']].all(dim=2).any(dim=1))[0]

def target_indices(labels, target):
    return set(filter(lambda index: labels[index] == target, range(len(labels))))

def weighted_f1(dataset_name, graphs, features, pattern_dict, model, embeddings= None, purity_list=None):
    if purity_list is None:
        purity_list, _ = purity(pattern_dict)
    
    #ground labels
    embeddings = fill_embeddings(embeddings, features, graphs, model)
    predicted_labels = list(map(lambda edge_index, x: torch.argmax(model(x, edge_index)[0]).item(), graphs, features))
    predicted_indices = list(map(lambda target: target_indices(predicted_labels, target), [0,1]))

    
    #predicted labels
    default = max([0, 1], key= lambda target: len(predicted_indices[target]))
    support_list = list(map(lambda feature, embedding: calculate_graph_support(dataset_name, feature, embedding, pattern_dict), features, embeddings))
    best_pattern = list(map(lambda support: max(support, key=lambda index: purity_list[index]) if len(support) else -1, support_list))
    pattern_predicted_labels = list(map(lambda pattern_index: int(pattern_dict[pattern_index]["Target"]) if pattern_index > -1 else default, best_pattern))
    pattern_predicted_indices = list(map(lambda target: target_indices(pattern_predicted_labels, target), [0,1]))
    TP = len(predicted_indices[1].intersection(pattern_predicted_indices[1]))
    TN = len(predicted_indices[0].intersection(pattern_predicted_indices[0]))
    FP = len(predicted_indices[0].intersection(pattern_predicted_indices[1]))
    FN = len(predicted_indices[1].intersection(pattern_predicted_indices[0]))
    w1 = (TP + FN)/(TN + TP + FP + FN)
    w2 = (TN + FP)/(TN + TP + FP + FN)
    return w1*TP/(TP + (FP + FN)/2) + w2*TN/(TN + (FP + FN)/2)



def purity(pattern_dict):
    try:
        purity_list = list(map(lambda pattern: max(pattern['c+'], pattern['c-'])/(pattern['c+']+ pattern['c-']), pattern_dict))
    except:
        print(list(map(lambda pattern: (pattern['c+'], pattern['c-'], pattern['components'], pattern['layer']), pattern_dict)))
        raise
    return purity_list, sum(purity_list)/len(pattern_dict)

def cover(dataset_name,graphs, features, pattern_dict, model, embeddings=None):
    embeddings = fill_embeddings(embeddings, features, graphs, model)
    cnt = len(list(filter(lambda feature_embedding: len(calculate_graph_support(dataset_name ,feature_embedding[0], feature_embedding[1], pattern_dict)), zip(features, embeddings))))
    return cnt/len(embeddings)

def recalculate_covers(dataset_name,graphs, features, embeddings, model, pattern_dict):
    for i in range(len(pattern_dict)):
        pattern_dict[i]['c-'], pattern_dict[i]['c+'] = 0, 0
    for i, (graph, feature) in enumerate(zip(graphs, features)):
        label = torch.argmax(model(feature, graph)[0].softmax(-1))
        graph_support = calculate_graph_support(dataset_name, feature, embeddings[i], pattern_dict)
        to_add = "c+" if label == 1 else 'c-'
        for index in graph_support:
            pattern_dict[index][to_add] += 1
    pattern_dict = list(filter(lambda pattern: pattern['c-'] + pattern['c+'], pattern_dict))
    for i in range(len(pattern_dict)):
        if "Target" in pattern_dict[i].keys():
            break
        pattern_dict[i]["Target"] = 1 if pattern_dict[i]["c+"] > pattern_dict[i]["c-"] else 0
    return pattern_dict


def calculate_mertics(dataset_name, train_graphs, train_features, pattern_dict, model, embeddings):
    cover_metric = cover(dataset_name, train_graphs, train_features, pattern_dict, model, embeddings)   
    purity_list, purity_metric =  purity(pattern_dict)
    weighted_f1_metric = weighted_f1(dataset_name, train_graphs, train_features, pattern_dict, model, embeddings, purity_list)
    return cover_metric, purity_metric, weighted_f1_metric

def process_binary_patterns(file_path):
    """
    Reads the file and generates a list of dictionaries for each row in the file.
    
    Args:
        file_path (str): Path to the text file containing patterns.
    
    Returns:
        list: A list of dictionaries where each dictionary represents the processed row.
    """
    def read_file(file_path):
        """Reads the file and returns its content as a list of strings."""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Skip the header line ("Patterns") and strip whitespace
        return [line.strip() for line in lines if line.strip() != "Patterns"]

    def get_non_zero_components(pattern):
        """Returns the indices of non-zero components in the binary pattern."""
        return [i for i, char in enumerate(pattern) if char == '1']
    
    def create_entry(non_zero_components):
        """Creates a dictionary for a given list of non-zero components."""
        return {"components": non_zero_components, "layer": -1, "type": "diff_bin"}
    
    # Read and process the file
    lines = read_file(file_path)
    # Generate the list of dictionaries
    return [create_entry(get_non_zero_components(line)) for line in lines]
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    parser.add_argument("--t1", type=float, default=0.15)
    parser.add_argument("--t2", type=float, default=0.1)
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--evolution", action="store_true", help="Evaluate metrics over number of patterns (default: False)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    method = args.method
    device = "cpu" 
    evaluate_number_of_patterns = args.evolution
    if method == "inside":
        pattern_dict = read_patterns(dataset_name)
    elif method == "diffversify":
        pattern_dict = convert_patterns_to_dict(load_diffversify_patterns(dataset_name, method))
    else:
        pattern_dict = process_binary_patterns(args.file)
    train_features, train_graphs, val_features, val_graphs, test_features, test_graphs = load_splited_data(dataset_name, device=device)
    
    model = load_gnn(dataset_name, device=device)
    model.eval()
    
    embeddings = fill_embeddings(None, train_features, train_graphs, model)
    pattern_dict = recalculate_covers(dataset_name, train_graphs, train_features, embeddings, model, pattern_dict)
    if evaluate_number_of_patterns:
        #sorted_cover_indices = list(map(lambda x: x[0], sorted(enumerate(pattern_dict), key=lambda pattern: pattern[1]["c+"] + pattern[1]['c-'], reverse=True)))
        purity_list,_ =  purity(pattern_dict)
        sorted_purity_list = list(map(lambda x: x[0], sorted(enumerate(purity_list), key=lambda x: x[1], reverse=True)))
        for number_of_patterns in tqdm(range(1, len(pattern_dict) + 1)):
            #indices_to_keep = sorted_cover_indices[:number_of_patterns]
            indices_to_keep = sorted_purity_list[:number_of_patterns]
            pattern_dict_to_calculate = list(map(lambda x: x[1], list(filter(lambda x: x[0] in indices_to_keep, enumerate(pattern_dict)))))
            cover_metric, purity_metric, weighted_f1_metric = calculate_mertics(dataset_name, train_graphs, train_features, pattern_dict_to_calculate, model, embeddings)
            with open(f"{dataset_name}_{method}_pattern_evaluation_res.txt", "a") as file:
                file.write(f"{dataset_name=} {method=} {number_of_patterns=} {purity_metric=} {cover_metric=} {weighted_f1_metric=}")
            #if cover_metric == 1:
            #    break
    else:
            cover_metric, purity_metric, weighted_f1_metric = calculate_mertics(dataset_name, train_graphs, train_features, pattern_dict, model, embeddings)
            print(f"{dataset_name=} {method=} {purity_metric=} {cover_metric=} {weighted_f1_metric=} {len(pattern_dict)=}")

