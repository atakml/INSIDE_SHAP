from bseval.utils import h_fidelity, calculate_edge_mask
import torch
import pickle 
from EdgeSHAPer.src.edgeshaper import Edgeshaper

from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
from tqdm import tqdm
def gastar_eval(dataset_name, device="cuda:0"):
    dataset, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device=device)
    model = load_gnn(dataset_name)
    h_fid_dict = {}
    explanations = {i: {} for i in range(sum(train_mask))}

    h_fid_dict['graphshaper'] = 0
    model = model.to("cuda:0")
    for i, data in tqdm(enumerate(dataset)):
        data = data.to("cuda")
        edge_shaper_explainer = Edgeshaper(model, data.x, data.edge_index, device="cuda")
        target_class = torch.argmax(model(data.x, data.edge_index)[0]).item()
        edge_values = edge_shaper_explainer.explain(M=50, target_class=target_class)
        max_hfid = (0,0,0,0)
        
        for j in list(range(1,5)) + list(range(5, 50, 5)):
            edge_mask, node_mask = calculate_edge_mask(data, edge_values, j/100, True)
            max_hfid = max(max_hfid,  h_fidelity(data, model, edge_mask, node_mask), key=lambda x: x[0])
                
        h_fid_dict['graphshaper'] += max_hfid
    for key in h_fid_dict.keys():
        h_fid_dict[key] /= len(dataset)
    return h_fid_dict
for dataset in ["aids", "BBBP", "mutag", "ba2", "Benzen", "AlkaneCarbonyl"]:
    print(gastar_eval(dataset))