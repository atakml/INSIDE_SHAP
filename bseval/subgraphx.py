import pickle
from bseval.utils import h_fidelity, load_dataset_to_explain
from tqdm import tqdm
import torch
from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
from dgl.nn import SubgraphX
from dgl import DGLGraph
from torch_geometric.utils import to_networkx, to_dgl

device="cpu"

for dataset_name in ["ba2", "aids", "BBBP", "mutag", "AlkaneCarbonyl", "Benzen"]:

    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device=device)
    model = load_gnn(dataset_name, device=device)
    hfid = 0
    subgraph_explainer = SubgraphX(model, 3, shapley_steps=10)
    """if dataset in ["Benzen", "AlkaneCarbonyl"]:
        with open(f"subgraph_{dataset}.pkl", "rb") as file:
            exp = pickle.load(file)
        index = torch.where(train_mask)[0]
    else:
        with open(f"baseline_explanations_{dataset}.pkl", "rb") as file:
            exp = pickle.load(file)
    """

    for i, data in tqdm(enumerate(train_loader)):
        target_class = torch.argmax(model(data.x, data.edge_index))
        subgraph_explanation = subgraph_explainer.explain_graph(to_dgl(data), data.x, target_class)
        node_mask = torch.zeros(data.x.shape[0]).to(device)
        """if dataset in ["Benzen", "AlkaneCarbonyl"]:
            node_mask[exp[index[i].item()]] = 1
        else:
            node_mask[exp[i]["subgraph"]] = 1"""
        node_mask[subgraph_explanation] = 1
        node_mask = node_mask.bool()
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        hfid += h_fidelity(data.to(device), model.to(device), edge_mask.to(device), node_mask.to(device))
    print(f"{dataset_name=}: {hfid/len(train_loader)}")
