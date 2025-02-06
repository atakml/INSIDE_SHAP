import torch
import pandas as pd

from torch.utils.data import Dataset
from ExplanationEvaluation.tasks.replication import get_classification_task, select_explainer, to_torch_graph
from torch_geometric.loader import DataLoader
from os import walk
from GIN1 import GIN
from ExplanationEvaluation.models.model_selector import model_selector
from codebase import config
from codebase.ExplanationEvaluation.datasets.dataset_loaders import load_dataset
import pickle
from GModel import GModel
from datasetmod.DatasetObj import GinDataset
from modelmod.gnn import load_gnn


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
    flag = False
    for graph_index in range(len(embeddings)):
        new_features.append([])
        for layer in range(len(patterns)):
            act_mat = torch.sign(embeddings[graph_index][layer]).bool()
            new_features[graph_index].append(((act_mat.unsqueeze(1).cuda() & patterns[layer].unsqueeze(0).cuda()) ==
                                              patterns[layer].unsqueeze(0).cuda()).all(dim=2).bool().cuda())
        new_features[graph_index] = torch.hstack(new_features[graph_index]).cuda().float()
        if not flag:
            features_to_delete = set(torch.where(new_features[graph_index].all(axis=0))[0].tolist())
            flag = True
        else:
            features_to_delete = features_to_delete.intersection(
                set(torch.where(new_features[graph_index].all(axis=0))[0].tolist()))
    indices_to_keep = torch.ones(new_features[0].shape[1], device="cuda:0")
    #indices_to_keep[list(features_to_delete)] = False
    for graph_index in range(len(embeddings)):
        new_features[graph_index] = new_features[graph_index][:, indices_to_keep.bool()]
    return new_features, []#,features_to_delete


def prepare_rules(dataset_name, layer_size=20):
    rule_path = f"/home/ata/shap_inside/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation_encode_motifs.csv"
    rules = pd.read_csv(rule_path, delimiter="=")
    binary_representation_rules = []
    for i, row in rules.iterrows():
        pattern = row[1].strip().split()
        layer_index = int(pattern[0][2:pattern[0].index('c')])
        while len(binary_representation_rules) <= layer_index:
            binary_representation_rules.append([])
        components = torch.tensor(list(map(lambda x: int(x[x.index('c') + 2:]), pattern)))
        binary_rule = torch.zeros(layer_size[layer_index])
        binary_rule = binary_rule.scatter_(-1, components, 1).to(dtype=torch.long)
        binary_representation_rules[layer_index].append(binary_rule)

    for i in range(len(binary_representation_rules)):
        binary_representation_rules[i] = torch.vstack(binary_representation_rules[i])
    return binary_representation_rules


def prepare_data(dataset_name, model, layer_size=20):

    try: 
        with open(f"preloaded_ds_{dataset_name}.pkl", "rb") as file:
            train_data_loader, validation_data_loader, test_data_loader, rules_to_delete = pickle.load(file)
        return train_data_loader, validation_data_loader, test_data_loader, rules_to_delete
    except: 

        if dataset_name in["AlkaneCarbonyl", "FluorideCarbonyl", "Benzen"]:
            from graphxai.datasets.real_world.alkane_carbonyl.alkane_carbonyl import AlkaneCarbonyl
            from graphxai.datasets.real_world.fluoride_carbonyl.fluoride_carbonyl import FluorideCarbonyl
            from graphxai.datasets.real_world.benzene.benzene import Benzene
            model = model.to("cuda:0")
            try:
                dataset = eval(dataset_name)(device="cuda:0", downsample_seed=1)
            except NameError:
                dataset = Benzene(device="cuda:0")
            dataset_list = dataset.graphs
            with open(f"/home/ata/shap_extend/inidices_split_{dataset_name}.pkl", "rb") as file:
                (train_indices, val_indices, test_indices) = pickle.load(file)
            labels = [dataset_list[i].y.int() for i in range(len(dataset_list))]
            features, graphs = [dataset_list[i].x for i in range(len(dataset_list))], [dataset_list[i].edge_index for i in range(len(dataset_list))]
            #with open("/home/ata/shap_extend/Benzen_train_data_for_rule.pkl", "rb") as file:
            #    (graphs, features, labels, indices) = pickle.load(file)    

            for i in range(len(features)):
                f = features[i]
                if f.shape[0] < 25: 
                    f = torch.vstack((f, torch.zeros((25-f.shape[0], f.shape[1]), device=f.device)))
                    features[i] = f
            features = torch.concatenate(features).reshape(len(features), 25, 14).cuda()
            print(features.shape)
            #graphs = [x.to("cpu") for x in graphs]
            graphs = [x.cuda() for x in graphs]
            if dataset_name == "Benzen":
                #print("condition holds")
                #print(traom_indices[4])
                with open("/home/ata/shap_extend/Benzen_train_indices_sampled_new.pkl", "rb") as file:
                    train_indices = pickle.load(file)
                #print(len(train_indices))
            train_indices, val_indices, test_indices = torch.tensor(train_indices), torch.tensor(val_indices), torch.tensor(test_indices)
            train_mask, val_mask, test_mask = torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, train_indices, True), torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, val_indices, True), torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, test_indices, True)
        else:
            graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(dataset_name)
        task = get_classification_task(graphs)
        features = torch.tensor(features)
        graphs = to_torch_graph(graphs, task)
        embeddings = []
        model_labels = []
        for i in range(len(graphs)):
            embeddings.append(model.embeddings(features[i], graphs[i]) + [features[i]])
            model_labels.append(model(features[i], graphs[i]).detach().clone().cuda())
        patterns = prepare_rules(dataset_name, list(map(lambda x: x.shape[1], embeddings[i])) + [features[0].shape[1]])
        new_features, rules_to_delete = generate_features(embeddings, patterns)
        train_dataset = GinDataset(graphs, new_features, model_labels, train_mask)
        validation_dataset = GinDataset(graphs, new_features, model_labels, val_mask)
        test_dataset = GinDataset(graphs, new_features, model_labels, test_mask)
        train_data_loader = DataLoader(train_dataset)
        validation_data_loader = DataLoader(validation_dataset)
        test_data_loader = DataLoader(test_dataset)
        with open(f"preloaded_ds_{dataset_name}.pkl", "wb") as file:
            pickle.dump((train_data_loader, validation_data_loader, test_data_loader, rules_to_delete), file)
        return train_data_loader, validation_data_loader, test_data_loader, rules_to_delete


def get_the_latest_date(model_names):
    date_and_times = list(map(lambda x: list(map(int, x.split("_")[2:4])), model_names))
    return max(date_and_times)


def select_best_model(dataset):
    path = f"/home/ata/shap_inside/models/{dataset}"
    model_names = [filenames for filenames in next(walk(path))[2] if dataset in filenames and "model" in filenames ]#and "feature" in filenames]
    latest_date = get_the_latest_date(model_names)
    best_model_name = max(list(filter(lambda x: list(map(int, x.split("_")[2:4])) == latest_date, model_names)),
                          key=lambda x: float(x.split("_")[-1]))
    print(best_model_name)
    model = init_gin(dataset)
    model.load_state_dict(torch.load(f"{path}/{best_model_name}"))
    return model

def load_gin_dataset(dataset, random=False):
	gnn_model = load_gnn(dataset)
    gnn_model.eval()
    training_loader, validation_loader, test_data_loader, rule_to_delete = prepare_data(dataset, gnn_model)
    return training_loader, validation_loader, test_data_loader, rule_to_delete



