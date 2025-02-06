import torch
from bseval.utils import h_fidelity, calculate_edge_mask, load_dataset_to_explain
import pickle 
from tqdm import tqdm 
from codebase.ExplanationEvaluation.explainers.SVXExplainer import GraphSVX
from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
device ="cpu"
for dataset_name in ["aids", "BBBP", "mutag", "ba2", "Benzen", "AlkaneCarbonyl"]:
    dataset, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device=device)
    gnn_model = load_gnn(dataset_name, device=device)
    """if dataset in ["AlkaneCarbonyl", "Benzen"]:
        index_list = torch.where(train_mask)[0]
        with open(f"/home/ata/shap_extend/{dataset}_svx_aug.pkl", "rb") as file:
            explanations = pickle.load(file)
    else: 
        with open(f"baseline_explanations_{dataset}.pkl", "rb") as file:
            explanations = pickle.load(file)
    hfid = 0
    if dataset == "Benzen":
        explanations = {key.item(): value for key, value in explanations.items()}"""
    hfid = None
    #print(explanations.keys())
    for i, data in tqdm(enumerate(dataset)):
        x, edge_index = data.x, data.edge_index
        best_hfid = (0, 0,0,0)
        svx_explainer = GraphSVX(gnn_model, [data.edge_index], torch.unsqueeze(data.x, 0), "graph", gpu=True)
        svx_values = svx_explainer.explain_graphs()
        best_hfid = (0,0,0,0)
        for j in list(range(1,5)) + list(range(5, 50, 5)):
                #print(explanations[index.item()][0][1][:x.shape[0]])
                #node_mask = explanations[index.item()][0][1][:x.shape[0]]
            edge_mask, node_mask =  calculate_edge_mask(data, svx_values[0][1][:x.shape[0]], j/100)
            best_hfid = max(best_hfid, h_fidelity(data, gnn_model, edge_mask, node_mask), key=lambda x: x[0])
        if hfid is None:
            hfid = best_hfid
        else:
            hfid += best_hfid

    print(f"{dataset=}: {hfid/len(dataset)}")


