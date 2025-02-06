import pickle

import torch

from bseval.utils import load_dataset_to_explain
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from graphrep.drawer import atoms
import networkx as nx
import matplotlib.pyplot as plt



dataset_name = "Benzen"

#with open(f"baseline_explanations_{dataset_name}.pkl", "rb") as file:
#    explanations = pickle.load(file)

with open(f"rule_masks_{dataset_name}.pkl", "rb") as file:
    explanations = pickle.load(file)
print(len(explanations))
explainer = "INSIDESHAP"
train_loader, gnn_model, graphs, features, labels, train_mask = load_dataset_to_explain(dataset_name)
indices = torch.where(train_mask)[0]
#for explainer in explanations[0].keys():
for i, data in tqdm(enumerate(train_loader)):
    #if i not in [1249, 1482, 1471, 1442, 1381, 1254]:
    #    continue
    #data = train_loader[index]
    if explainer != "subgraph":
        data.node_value = explanations[i]#[explainer]
    else:
        node_values = torch.zeros(data.x.shape[0])
        node_values[explanations[i][explainer]] = 1
        data.node_value = node_values
    if explainer != "graphshaper":
        nx_graph = to_networkx(data, node_attrs=["x", "node_value"])
    else:
        nx_graph = to_networkx(data, node_attrs=["x"], edge_attrs=["node_value"])
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
    if explainer != "graphshaper":
        colors = {j: color for j, color in nx_graph.nodes(data="node_value")}
        colors = [colors[j] for j in sorted(colors.keys())]
    else:
        colors = {(l,j): color for l,j, color in nx_graph.edges(data="node_value")}
        colors = [colors[(l,j)] for l, j in sorted(colors.keys())]
    labels = {node: atoms[dataset_name][torch.argmax(torch.tensor(node_data)).item()] for node, node_data in
              nx_graph.nodes(data='x')}
    plt.clf()
    if explainer != "graphshaper":
        nx.draw(nx_graph, labels=labels, with_labels=True, node_color=colors)
    else:
        nx.draw(nx_graph, labels=labels, with_labels=True, edge_color=colors)
    plt.savefig(f"bseval/figs/{dataset_name}/{explainer}{i}_{data.y}.png")
