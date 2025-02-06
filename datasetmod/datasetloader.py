import pickle

import torch
from torch_geometric.utils import to_networkx, to_dgl
import networkx as nx
from codebase.ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from codebase.ExplanationEvaluation.tasks.replication import get_classification_task, to_torch_graph
from codebase.ExplanationEvaluation.models.model_selector import model_selector
from codebase.ExplanationEvaluation.explainers.SVXExplainer import GraphSVX
from GStarX.gstarx import GStarX
from dgl.nn import SubgraphX
from EdgeSHAPer.src.edgeshaper import Edgeshaper
from tqdm import tqdm
from codebase.GINUtil import GinDataset
from torch_geometric.data import Data
#from dig.xgraph.method import FlowX
from dgl import DGLGraph
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from datasetmod.DatasetObj import SPMDataSet
from datasetmod.data_utils import select_mask
from pathlib import Path
parent_directory = str(Path.cwd().parent)
def load_dataset_gnn(dataset_name, device="cuda:0"):
    """
    load dataset in the format of the SPMDataSet. The dataset the input data for the gnn to explain
    :return: train_loader, all the graphs, features and labels, and the train_mask, test_mask, valiadtion_mask 

    """
    if dataset_name in["AlkaneCarbonyl", "FluorideCarbonyl", "Benzen"]:
        from graphxai.datasets.real_world.alkane_carbonyl.alkane_carbonyl import AlkaneCarbonyl
        from graphxai.datasets.real_world.fluoride_carbonyl.fluoride_carbonyl import FluorideCarbonyl
        from graphxai.datasets.real_world.benzene.benzene import Benzene
        try:
            dataset = eval(dataset_name)(device=device, downsample_seed=1)
        except NameError:
            dataset = Benzene(device=device)
        dataset_list = dataset.graphs
        with open(f"inidices_split_{dataset_name}.pkl", "rb") as file:
            (train_indices, val_indices, test_indices) = pickle.load(file)
        if dataset_name == "Benzen":
            with open(f"Benzen_train_indices_sampled_new.pkl", "rb") as file:
                train_indices = pickle.load(file)
        labels = [dataset_list[i].y for i in range(len(dataset_list))]
        features, graphs = [dataset_list[i].x for i in range(len(dataset_list))], [dataset_list[i].edge_index for i in range(len(dataset_list))]
        for i in range(len(features)):
            f = features[i]
            if f.shape[0] < 25: 
                f = torch.vstack((f, torch.zeros((25-f.shape[0], f.shape[1]), device=f.device)))
                features[i] = f
        features = torch.concatenate(features).reshape(len(features), 25, 14).to(device)
        train_indices, val_indices, test_indices = torch.tensor(train_indices), torch.tensor(val_indices), torch.tensor(test_indices)
        train_mask, val_mask, test_mask = train_mask, val_mask, test_mask = torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, train_indices, True), torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, val_indices, True), torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, test_indices, True)
    else:
        graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(dataset_name)
    task = get_classification_task(graphs)
    features = torch.tensor(features, device=device)
    graphs = to_torch_graph(graphs, task)
    graphs = [x.to(device) for x in graphs]
    train_loader = SPMDataSet(graphs, features, labels, train_mask)
    return train_loader, graphs, features, labels, train_mask, val_mask, test_mask



def load_splited_data(dataset_name, device="cuda:0"):
    _, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device)
    train_features, train_graphs = select_mask(features, graphs, train_mask)
    test_features, test_graphs = select_mask(features, graphs, test_mask)
    val_features, val_graphs = select_mask(features, graphs, val_mask)
    return train_features, train_graphs, val_features, val_graphs, test_features, test_graphs



