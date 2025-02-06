from os import walk
import pickle
from itertools import product
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.nn.functional import log_softmax, softmax, relu
from GModel import GModel
from datasetmod.DatasetObj import GinDataset, GinDatasetBatch
from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
from modelmod.gnn_utils import concat_embeddings
from patternmod.inside_utils import read_patterns
from patternmod.diffversify_utils import *
from pathlib import Path
parent_directory = str(Path.cwd().parent)

input_feature_size = {"aids": 79, "ba2": 17, "mutag": 71, "BBBP": 71, "AlkaneCarbonyl": 52, "Benzen": 60, "FluorideCarbonyl": 49}



def generate_features(embeddings, patterns):
    """
    :param embeddings: embedding matrix of the graph in the dimension of (n, l, d) where l is the number of layers,
    n number of nodes, and d is the embedding dimension
    :param patterns: patterns in the binary form in shape of (l,
    m, d) where m is number of patterns for each layer :return: feature matrix of the GIN in the dimension of (n, l*m)
    """
    new_features = []
    features_to_delete = []
    for graph_index in range(len(embeddings)):
        new_features.append([])
        for layer in range(len(patterns)):
            act_mat = torch.sign(embeddings[graph_index][layer]).bool()
            new_features[graph_index].append(((act_mat.unsqueeze(1).cuda() & patterns[layer].unsqueeze(0).cuda()) ==
                                              patterns[layer].unsqueeze(0).cuda()).all(dim=2).bool().cuda())
        new_features[graph_index] = torch.hstack(new_features[graph_index]).cuda().float()
    return new_features, []#,features_to_delete


def prepare_rules(dataset_name, method="inside", layer_size=20):
    if method == "inside":
        rules = read_patterns(dataset_name)
    else: 
        rules = convert_patterns_to_dict(load_diffversify_patterns(dataset_name, method))
    binary_representation_rules = []
    for i, pattern in enumerate(rules):
        layer_index = pattern["layer"]
        if layer_index == -1: 
            layer_index = 0 
        while len(binary_representation_rules) <= layer_index:
            binary_representation_rules.append([])
        components = torch.tensor(pattern["components"])
        binary_rule = torch.zeros(layer_size[layer_index])
        binary_rule = binary_rule.scatter_(-1, components, 1).to(dtype=torch.long)
        binary_representation_rules[layer_index].append(binary_rule)

    for i in range(len(binary_representation_rules)):
        binary_representation_rules[i] = torch.vstack(binary_representation_rules[i])
    return binary_representation_rules


def prepare_data(dataset_name, model, layer_size=20, random= False, device="cuda:0", method="inside", batch_size=(1,1,1)):
    print(method)
    file_address = f"preloaded_ds_{dataset_name}.pkl" if not random else f"preloaded_random_ds_{dataset_name}.pkl"
    if method != "inside":
        file_address  = file_address[:-4] + method + ".pkl"
    try: 
        raise
        with open(file_address, "rb") as file:
            train_data_loader, validation_data_loader, test_data_loader, rules_to_delete = pickle.load(file)
        return train_data_loader, validation_data_loader, test_data_loader, rules_to_delete
    except: 
        print("Building dataset")
        train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device=device)
        embeddings = []
        model_labels = []
        for i in range(len(graphs)):
            if method == "inside":
                embeddings.append(model.embeddings(features[i], graphs[i]) + [features[i]])
            else:
                embeddings.append(model.embeddings(features[i], graphs[i]))
            model_labels.append(model(features[i], graphs[i]).detach().clone().cuda())
        if method != "inside":
            embeddings = concat_embeddings(dataset_name, embeddings, features, stack=True).unsqueeze(1)

        patterns = prepare_rules(dataset_name, layer_size=list(map(lambda x: x.shape[1], embeddings[0])) + [features[0].shape[1]], method=method)
        new_features, rules_to_delete = generate_features(embeddings, patterns)
        if random: 
            new_features = generate_random_features(len(graphs), list(map(lambda x: x.shape, new_features)), device=device)
        dataset_object = GinDatasetBatch if batch_size != (1,1,1) else GinDataset
        train_dataset = dataset_object(graphs, new_features, model_labels, train_mask)
        validation_dataset = dataset_object(graphs, new_features, model_labels, val_mask)
        test_dataset = dataset_object(graphs, new_features, model_labels, test_mask)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size[0])
        validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size[1])
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size[2])
        with open(file_address, "wb") as file:
            pickle.dump((train_data_loader, validation_data_loader, test_data_loader, rules_to_delete), file)
        return train_data_loader, validation_data_loader, test_data_loader, rules_to_delete


def get_the_latest_date(model_names):
    date_and_times = list(map(lambda x: list(map(int, x.split("_")[2])), model_names))
    return max(date_and_times)


def select_best_model(dataset, method="inside"):
    if method == "inside":
        rules = read_patterns(dataset)
    else: 
        rules = convert_patterns_to_dict(load_diffversify_patterns(dataset, method))
    feature_size = len(rules)
    path = f"{parent_directory}/shap_inside/models/{dataset}"
    model_names = [filenames for filenames in next(walk(path))[2] if dataset in filenames and "model" in filenames ]#and "feature" in filenames]
    latest_date = get_the_latest_date(model_names)
    best_model_name = min(list(filter(lambda x: list(map(int, x.split("_")[2])) == latest_date, model_names)),
                          key=lambda x: float(x.split("_")[-2]))
    print(best_model_name)
    model = GModel(feature_size, 20, 2)
    model.load_state_dict(torch.load(f"{path}/{best_model_name}"))
    return model


def generate_random_features(number_of_graphs, feature_shape_list, device="cuda:0"):
    feature_list = []
    for i in range(number_of_graphs):
        feature_list.append(torch.randint(0, 2, feature_shape_list[i], dtype=torch.float, device=device))
    return feature_list

def load_gin_dataset(dataset, method="inside", random=False, device="cuda:0", batch_size=(1,1,1)):
    print(method)
    gnn_model = load_gnn(dataset, device=device)
    gnn_model.eval()
    training_loader, validation_loader, test_data_loader, rule_to_delete = prepare_data(dataset, gnn_model, method=method, random=random, device=device, batch_size=batch_size)
    return training_loader, validation_loader, test_data_loader, rule_to_delete

def prepare_data_for_simpler_model(data_loader):
    X = []
    labels = []
    for data in data_loader:
        edge_index, features, label = data
        features = features[0]
        features = features.sum(axis=0).cpu().numpy()
        label = softmax(label[0], -1)[0]
        labels.append(label.cpu())
        X.append(features)
    return X, labels
    
    
def evaluate_simpler_model_on_data(model, X, labels):
    predicted_labels = model.predict(X)
    acc = sum([(labels[i][0] < 0.5) == (predicted_labels[i][0] < 0.5) for i in range(len(labels))])/len(labels)
    loss = kl_loss(torch.from_numpy(predicted_labels).log(),torch.vstack(labels)).mean()
    return loss, acc
    
    
    
def optimize_model_with_grid_search(model, train_data, train_labels, val_data, val_labels, param_grid):
    """
    Function to perform grid search to optimize the given model on the validation set.
    
    Args:
    - model: The model to be optimized (e.g., DecisionTreeRegressor).
    - train_data: The training data (features).
    - train_labels: The training labels.
    - val_data: The validation data (features).
    - val_labels: The validation labels.
    - param_grid: A dictionary containing the hyperparameters to search over. 
      Example: {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}.
    
    Returns:
    - best_model: The model with the best validation score.
    - best_params: The best hyperparameters found during the grid search.
    """
    import copy
    best_val_score = np.inf  # Keep track of the best validation score
    best_params = None  # Store the best parameters
    best_model = None  # Store the best model

    # Grid search (manual loop through the hyperparameters dictionary)
    # For each combination of hyperparameters in the grid, fit the model and evaluate

    param_combinations = product(*param_grid.values())
    param_keys = list(param_grid.keys())

    for combination in param_combinations:
        # Create a dictionary of the current parameter combination
        current_params = dict(zip(param_keys, combination))
        
        # Set the model parameters using the current parameter combination
        model.set_params(**current_params)

        # Train the model on the training set
        model.fit(train_data, train_labels)

        # Evaluate the model using the validation set (X_val, val_labels)
        val_score = evaluate_simpler_model_on_data(model, val_data, val_labels)[0]

        # Update best model if this model performs better on validation set
        if val_score < best_val_score:
            best_val_score = val_score
            best_params = copy.deepcopy(current_params)
            best_model = model
    return best_model, best_params
    
    
def kl_loss(model_log_probs, gnn_probs):
    #print(model_log_probs.shape, gnn_probs.shape)
    loss = -gnn_probs[:, 0] * (model_log_probs[:, 0] - torch.log(gnn_probs[:, 0]+1e-8)) - gnn_probs[:, 1] * (
        	model_log_probs[:, 1] - torch.log(gnn_probs[:, 1]+1e-8))
    return loss
