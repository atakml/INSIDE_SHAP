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
from pathlib import Path
parent_directory = str(Path.cwd().parent)

class SPMDataSet(GinDataset):
    def __getitem__(self, idx):
        index = self.indices[idx]
        data = Data(edge_index=self.graphs[index],
                    x=self.features[index][0][0],
                    label=self.labels[index])
        return data

class GStarModel(torch.nn.Module):
    def __init__(self, model):
        super(GStarModel, self).__init__()
        self.model = model
    def embeddingss(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), device=x.device)
        stack = []
        x = x.to("cuda:0")
        edge_index = edge_index.to("cuda:0")
        edge_weights = edge_weights.to("cuda:0")
        self.model = self.model.to("cuda:0")
        out1 = self.model.conv1(x, edge_index, edge_weights)
        out1 = out1.relu()
        stack.append(out1)

        out2 = self.model.conv2(out1, edge_index, edge_weights)
        out2 = out2.relu()
        stack.append(out2)

        out3 = self.model.conv3(out2, edge_index, edge_weights)

        return [out1]+[out2]+[out3]
    def embeddings(self, x, edge_index, edge_weights=None):
        return self.embeddingss(x, edge_index, edge_weights)[:-1]
    def embedding(self, x, edge_index, edge_weights=None):
        return self.embeddingss(x, edge_index, edge_weights)[-1]
    def forward(self, x=None, edge_index=None, batch=None, edge_weights=None, data=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
        if isinstance(x, DGLGraph):
            data = from_dgl(x)
            edge_index = data.edge_index
            x = data.x
        embed = self.embedding(x, edge_index, edge_weights)
        embed = global_mean_pool(embed, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        embed = self.model.lin(embed)

        return embed
def fidelity(data, model, edge_mask, node_mask, original_prob):
    data = data.detach().clone()
    edge_mask = edge_mask.to(data.x.device)
    if node_mask is not None:
        edge_mask = node_mask[data.edge_index[0]] | node_mask[data.edge_index[1]]
        data.x[node_mask] = torch.zeros((node_mask.sum().int(), data.x.shape[1]), device=data.x.device)
    masked_prob = model(data.x, data.edge_index, edge_weights=(~edge_mask).float())
    target_class = torch.argmax(original_prob[0])
    #return int((torch.softmax(original_prob[0], dim=-1)[target_class] >0.5) != (torch.softmax(masked_prob[0], dim=-1)[target_class]>0.5))
    return torch.softmax(original_prob[0], dim=-1)[target_class] - torch.softmax(masked_prob[0], dim=-1)[target_class]


def infidelity(data, model, edge_mask, node_mask, original_prob):
    data = data.detach().clone()
    edge_mask = edge_mask.to(data.x.device)
    if node_mask is not None:
        data.x[~node_mask] = torch.zeros(((~node_mask).sum().int(), data.x.shape[1]), device=data.x.device)
    masked_prob = model(data.x, data.edge_index, edge_weights=edge_mask.float())
    target_class = torch.argmax(original_prob[0])
    #return int((torch.softmax(original_prob[0], dim=-1)[target_class] >0.5) != (torch.softmax(masked_prob[0], dim=-1)[target_class]>0.5))
    return torch.softmax(original_prob[0], dim=-1)[target_class] - torch.softmax(masked_prob[0], dim=-1)[target_class]


def sparsity(data, edge_mask, node_mask):
    edge_mask = edge_mask.to(data.x.device)
    #edges = torch.reshape(data.edge_index[:, edge_mask], (-1,)).unique()
    original_nx_graph = to_networkx(data)
    original_nx_graph.remove_edges_from(nx.selfloop_edges(original_nx_graph))
    original_nx_graph.remove_nodes_from(list(nx.isolates(original_nx_graph)))
    original_num_node = original_nx_graph.number_of_nodes()
    number_of_nodes = node_mask[:original_num_node].sum() if node_mask is not None else torch.unique(data.edge_index[:, edge_mask].flatten()).numel()
    number_of_edges = torch.sum(edge_mask)
    if number_of_edges > original_nx_graph.number_of_edges() or number_of_nodes > original_nx_graph.number_of_nodes():
        print(f"{number_of_edges=} {original_nx_graph.number_of_edges()=}  {number_of_nodes=}, {original_nx_graph.number_of_nodes()=}")
        print(f"{data.x=}, {data.edge_index=}")
        print(f"{node_mask=}")
        print(f"{edge_mask=}")
        assert False
    return 1 - (number_of_edges + number_of_nodes) / (
            original_num_node + original_nx_graph.number_of_edges())


def h_fidelity(data, model, edge_mask, node_mask):
    original_prob = model(data.x, data.edge_index)
    fid = fidelity(data, model, edge_mask, node_mask, original_prob)
    #return fid
    in_fid = infidelity(data, model, edge_mask, node_mask, original_prob)
    #return in_fid
    spars = sparsity(data, edge_mask, node_mask)
    #return torch.tensor((fid, in_fid, spars))
    n_fid = fid * spars
    n_inv_fid = in_fid * (1 - spars)
    return torch.tensor(((1 + n_fid) * (1 - n_inv_fid) / (2 + n_fid - n_inv_fid), fid, in_fid, spars))


def calculate_edge_mask(data, attribution, edge_percent, is_edge=False):
    attribution = torch.tensor(attribution).detach()
    sorted_value, sorted_indices = torch.sort(attribution, descending=True)
    data =  data.to(data.x.device)
    edge_mask = torch.zeros(data.edge_index.shape[1], device=data.x.device).bool()
    
    original_nx_graph = to_networkx(data)
    original_nx_graph.remove_edges_from(nx.selfloop_edges(original_nx_graph))
    original_nx_graph.remove_nodes_from(list(nx.isolates(original_nx_graph)))
    number_of_nodes = original_nx_graph.number_of_nodes()

    if not is_edge:
        number_of_edges = 0
        selected_nodes = []
        cnt = 0
        node_mask = torch.zeros(data.x.shape[0], device=data.x.device).bool()
        while 1 - sparsity(data, edge_mask, node_mask) < edge_percent: #* data.edge_index.shape[1]:
            if cnt == sorted_value.shape[0]:
                break
            if sorted_indices[cnt] >= number_of_nodes:
                cnt += 1 
                continue
            selected_nodes.append(sorted_indices[cnt])
            node_mask[sorted_indices[cnt]] = True
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            number_of_edges = torch.sum(edge_mask)
            # print(number_of_edges, edge_percent * data.edge_index.shape[1])
            cnt += 1
        return edge_mask.bool(), node_mask.bool()
    else:
        try:
            cnt = 0 
            while 1 - sparsity(data, edge_mask, None) < edge_percent:
                if cnt == sorted_value.shape[0]: 
                    break
                if data.edge_index[0][sorted_indices[cnt]] == data.edge_index[1][sorted_indices[cnt]]:
                    cnt += 1
                    continue 
                edge_mask[sorted_indices[cnt]] = True
                cnt += 1
        except:
            print(edge_percent, edge_mask.shape[0], sorted_indices.shape, data.edge_index.shape, attribution.shape)
            raise
    return edge_mask.bool(), None

from GModel import GModel
def load_dataset_to_explain(dataset_name):
    if dataset_name in["AlkaneCarbonyl", "FluorideCarbonyl", "Benzen"]:
        from graphxai.datasets.real_world.alkane_carbonyl.alkane_carbonyl import AlkaneCarbonyl
        from graphxai.datasets.real_world.fluoride_carbonyl.fluoride_carbonyl import FluorideCarbonyl
        from graphxai.datasets.real_world.benzene.benzene import Benzene
        try:
            dataset = eval(dataset_name)(device="cuda:0", downsample_seed=1)
        except NameError:
            dataset = Benzene(device="cuda:0")
        dataset_list = dataset.graphs
        with open(f"{parent_directory}/shap_extend/inidices_split_{dataset_name}.pkl", "rb") as file:
            (train_indices, val_indices, test_indices) = pickle.load(file)
        if dataset_name == "Benzen":
            with open(f"{parent_directory}/shap_extend/Benzen_train_indices_sampled_new.pkl", "rb") as file:
                train_indices = pickle.load(file)
        labels = [dataset_list[i].y.int() for i in range(len(dataset_list))]
        features, graphs = [dataset_list[i].x for i in range(len(dataset_list))], [dataset_list[i].edge_index for i in range(len(dataset_list))]
        #with open("{parent_directory}/shap_extend/Benzen_train_data_for_rule.pkl", "rb") as file:
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
        train_indices, val_indices, test_indices = torch.tensor(train_indices), torch.tensor(val_indices), torch.tensor(test_indices)
        train_mask, val_mask, test_mask = train_mask, val_mask, test_mask = torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, train_indices, True), torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, val_indices, True), torch.zeros(len(dataset_list), dtype=torch.bool).scatter_(0, test_indices, True)
        from graphxai.gnn_models.graph_classification.gcn import GCN_3layer
        from os import listdir
        from os.path import isfile, join
        mypath = f"{parent_directory}/shap_extend/models/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        models = list(filter(lambda x: dataset_name in x, onlyfiles))
        if dataset == "AlkaneCarbonyl":
            models = list(filter(lambda x: "f1" in x and "unique" in x, models))
        best_model_name = max(models, key=lambda x: float(x.split("_")[-1].split(".")[0]))
        path = f"{parent_directory}/shap_extend/models/{best_model_name}"
        if dataset_name in ["AlkaneCarbonyl", "Benzen"]:
            gnn_model = torch.load(path)
            #gnn_model = GModel(gnn_model)
        else:
            model.load_state_dict(torch.load(path))
            gnn_model = GStarModel(model)
        gnn_model = gnn_model.to("cuda:0")
    else:
        graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(dataset_name)
        gnn_model, checkpoint = model_selector("GNN",
                                           dataset_name,
                                           pretrained=True,
                                           return_checkpoint=True)
    task = get_classification_task(graphs)
    features = torch.tensor(features)
    graphs = to_torch_graph(graphs, task)
    train_loader = SPMDataSet(graphs, features, labels, train_mask)
    return train_loader, gnn_model, graphs, features, labels, train_mask


def calculate_h_fid_for_dataset(dataset_name):
    dataset, model, graphs, features, labels, train_mask = load_dataset_to_explain(dataset_name)
    h_fid_dict = {}
    if dataset_name in ["AlkaneCarbonyl", "Benzen"]:

        index = torch.where(train_mask)[0].tolist()
    #gstar_explainer = GStarX(model, device="cuda", payoff_type="prob")
    #subgraph_explainer = SubgraphX(model, 3, shapley_steps=10)
    #explanations = {i: {} for i in range(sum(train_mask))}
    #with open(f"baseline_explanations_{dataset_name}.pkl", "rb") as file:
    #with open(f"Gstar_{dataset_name}.pkl" if dataset_name in ["AlkaneCarbonyl", "Benzen"] else f"baseline_explanations_{dataset_name}.pkl" if dataset_name == "aids" else f"baseline_explanations_{dataset_name}_gstar.pkl", "rb") as file:
    with open(f"baseline_explanations_{dataset_name}_edgshaper.pkl", "rb") as file:
        explanations = pickle.load(file)
    h_fid_dict['svx'] = 0
    h_fid_dict['gstar'] = 0
    h_fid_dict['subgraph'] = 0
    h_fid_dict['edgeshaper'] = 0
    hf = 0
    model = model.to("cuda:0")
    #floxp = FlowX(model, explain_graph=True)
    for i, data in tqdm(enumerate(dataset)):
        #_, edge_mask, _ = floxp(data.x, data.edge_index, sparsity=0.9, num_classes=2)
        #hf += h_fidelity(data, model, edge_mask)
        #svx_explainer = GraphSVX(model, [data.edge_index], torch.unsqueeze(data.x, 0), "graph", gpu=True)
        #svx_values = svx_explainer.explain_graphs()
        #explanations[i]['svx'] = svx_values[0][1]
        #svx_values = explanations[i]['svx']
        data = data.to("cuda")

        #if dataset_name not in ["AlkaneCarbonyl", "Benzen"]:
        #    try:
        #        gstar_explanation =explanations[i]['gstar']
        #    except:
        #        print(explanations[i].keys())
        #        raise
        #else:
        #    gstar_explanation =explanations[index[i]]
        max_hfid = (0,0,0,0)
        for j in list(range(1,5)) + list(range(5, 50, 5)):
        #edge_mask = calculate_edge_mask(data, svx_values[0][1], 0.1)
        #h_fid_dict['svx'] += h_fidelity(data, model, edge_mask)
        #gstar_explanation = gstar_explainer.explain(data)
 
            #edge_mask, node_mask = calculate_edge_mask(data, gstar_explanation, j/100)
            #max_hfid = max(max_hfid,  h_fidelity(data, model, edge_mask, node_mask), key=lambda x: x[0])
        #explanations[i]['gstar'] = gstar_explanation
        #h_fid_dict['gstar'] += h_fidelity(data, model, edge_mask)
        #target_class = torch.argmax(model(data.x, data.edge_index))
        #data = data.to("cpu")
        #model = model.to("cpu")
        #subgraph_explanation = subgraph_explainer.explain_graph(to_dgl(data), data.x, target_class)
        #node_values = torch.zeros(data.x.shape[0])
        #node_values[subgraph_explanation] = 1
        #edge_mask = calculate_edge_mask(data, node_values, 0.1)
        #explanations[i]['subgraph'] = subgraph_explanation
        #node_mask = torch.zeros(data.x.shape[0])
        #node_mask[subgraph_explanation] = 1
        #node_mask = node_mask.bool()
        #edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        #h_fid_dict['subgraph'] += h_fidelity(data, model, edge_mask)
        #data = data.to("cpu")
        #edge_shaper_explainer = Edgeshaper(model, data.x, data.edge_index, device="cuda")
        #target_class = torch.argmax(model(data.x, data.edge_index)[0]).item()
        #edge_values = edge_shaper_explainer.explain(M=50, target_class=target_class)
            edge_values = explanations[i]['graphshaper']
            
            edge_mask, node_mask = calculate_edge_mask(data, edge_values, j/100, True)
            max_hfid = max(max_hfid,  h_fidelity(data, model, edge_mask, node_mask), key=lambda x: x[0])
        #explanations[i]['graphshaper'] = edge_values
        h_fid_dict['edgeshaper'] += max_hfid
        #h_fid_dict['gstar'] += max_hfid
        #with open(f"baseline_explanations_{dataset_name}_edgshaper.pkl", "wb") as pkl:
    #    pickle.dump(explanations, pkl)
    #return h_fid_dict"""
    for key in h_fid_dict.keys():
        h_fid_dict[key] /= len(dataset)
    return h_fid_dict
if __name__ == "__main__":
    with torch.no_grad():
        for dataset in ["ba2", "aids", "AlkaneCarbonyl", "BBBP", "Benzen"]:
            print(f"{dataset=}: {calculate_h_fid_for_dataset(dataset)}")
