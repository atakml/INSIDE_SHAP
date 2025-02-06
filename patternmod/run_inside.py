from codebase.ExplanationEvaluation.explainers.InsideGNN import InsideGNN
import torch
import numpy as np 
from datasetmod.datasetloader import load_splited_data, load_dataset_gnn
from modelmod.gnn import load_gnn
from datasetmod.data_utils import select_mask

class Config:
    def __init__(self, dataset_name):
        self.dataset = dataset_name
        self.model = "GNN"
        self.explainer = "INSIDE"

def run_inside(dataset_name):
    config = Config(dataset_name)

    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name)
    train_features, train_graphs = select_mask(features, graphs, train_mask)
    gnn_model = load_gnn(dataset_name)
    explainer = InsideGNN(gnn_model, train_graphs, train_features, "graph", config, labels)
    s = 4
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)
    indices = np.arange(len(train_graphs))
    explainer.prepare(indices)

