from GModel import GModel
from graphxai.gnn_models.graph_classification.utils import  test, train
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss, init
from torch.nn.functional import log_softmax, softmax, relu
import torch
from graphxai.datasets import GraphDataset
from torch_geometric.data import Data
from ExplanationEvaluation.tasks.replication import get_classification_task, select_explainer, to_torch_graph
from os import listdir
from os.path import isfile, join
from modelmod.gnn_utils import evaluate_metric_on_loader
from datasetmod.DatasetObj import *
from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn


# Define a function to apply He initialization to model layers
def initialize_weights_he(model):
    # Iterate through all layers in the model
    for layer in model.modules():
        # Check if the layer has a `weight` attribute (e.g., Conv layers, Linear layers)
        if hasattr(layer, 'weight') and layer.weight is not None:
            if layer.weight.dim() >= 2:
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
        # Check if the layer has a `bias` attribute and initialize it to zeros
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)

from pathlib import Path
parent_directory = str(Path.cwd().parent)
def train_model(train_loader, validation_loader, model, epochs=2000):
    initialize_weights_he(model)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    early_stop = 0
    best_val = -1
    for epoch in range(epochs):
        train(model, optimizer, criterion, train_loader)
        if epoch%5==0:
            f1, precision, recall, auprc, auroc = test(model, validation_loader)
            print(f"{epoch=} {auroc=}, {best_val}")
            if best_val < auroc:
                best_val = auroc
                early_stop = 0
                torch.save(model, f"{parent_directory}/shap_inside/modelmod/models/gcn_unique_f1_{dataset_name}_{epoch}_{auroc}.pth")
            else:
                early_stop += 1
            if early_stop == 10:
                break
    return model


def acc_func(dataloader, model):
    cnt = 0
    pos_probs = []
    neg_probs = []
    for data in dataloader:
        y_pred = torch.argmax(model(data.x.to("cuda:0"), data.edge_index.to("cuda:0"), batch=None).softmax(dim=-1))
        cnt += int(y_pred == data.y.to("cuda:0"))
    acc = cnt/len(dataloader)
    #return ((pos_mean.item(), pos_std.item()), (neg_mean.item(), neg_std.item()))
    return acc



if __name__ == "__main__":
    import argparse
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Training the GNN.")
    # Add arguments
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model (default: False)")
    # Parse arguments
    args = parser.parse_args()
    dataset_name, device = args.dataset_name, args.device
    
    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device)
    dataset = ClassicDataset(dataset_name, graphs, features, torch.tensor(labels), train_mask, val_mask, test_mask, False)
    train_loader = dataset.get_train_loader()[0]
    validation_loader = dataset.get_val_loader()[0]
    test_loader = dataset.get_loader(dataset.test_index, batch_size=1)[0]
    model = load_gnn(dataset_name, pretrained=args.pretrained)
    model = model.to(device)

    if dataset_name == "ba2":
        model.conv1.normalize = False
        model.conv2.normalize = False
        model.conv3.normalize = False
        model.conv1.add_self_loops = False
        model.conv2.add_self_loops = False
        model.conv3.add_self_loops = False
    if not args.pretrained:
        model = train_model(train_loader, validation_loader, model)

    model.eval()
    train_loader = dataset.get_loader(dataset.train_index, batch_size=1)[0]
    train_acc = acc_func(train_loader, model)
    val_acc = acc_func(validation_loader, model)
    test_acc = acc_func(test_loader, model)
    print(f"{train_acc=}\n {val_acc=}\n {test_acc=}")
    with open(f"{dataset_name} training results.txt", "a") as file:
        file.write(f"{train_acc=} {val_acc=} {test_acc=}\n")
