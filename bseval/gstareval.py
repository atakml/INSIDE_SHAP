from bseval.utils import h_fidelity, calculate_edge_mask
import torch
import pickle 

from GStarX.gstarx import GStarX
from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
from tqdm import tqdm
def gastar_eval(dataset_name, device="cuda:0"):
    dataset, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name, device=device)
    model = load_gnn(dataset_name)
    h_fid_dict = {}
    if dataset_name in ["AlkaneCarbonyl", "Benzen"]:
        index = torch.where(train_mask)[0].tolist()
    gstar_explainer = GStarX(model, device="cuda", payoff_type="prob")
    explanations = {i: {} for i in range(sum(train_mask))}
    pretrained = True
    """try:
        with open(f"Gstar_{dataset_name}.pkl" if dataset_name in ["AlkaneCarbonyl", "Benzen"] else f"baseline_explanations_{dataset_name}.pkl" if dataset_name == "aids" else f"baseline_explanations_{dataset_name}_gstar.pkl", "rb") as file:
            explanations = pickle.load(file)
    except: 
        pretrained = False"""
    h_fid_dict['gstar'] = 0
    hf = 0
    model = model.to("cuda:0")
    for i, data in tqdm(enumerate(dataset)):
        data = data.to("cuda")
        """try:
            if dataset_name not in ["AlkaneCarbonyl", "Benzen"]:
                try:
                    gstar_explanation =explanations[i]['gstar']
                except:
                    print(explanations[i].keys())
                    raise
            else:
                gstar_explanation =explanations[index[i]]
        except:"""
        gstar_explanation = gstar_explainer.explain(data)
        explanations[i]['gstar'] = gstar_explanation
        max_hfid = (0,0,0,0)
        
        for j in list(range(1,5)) + list(range(5, 50, 5)):
            edge_mask, node_mask = calculate_edge_mask(data, gstar_explanation, j/100)
            max_hfid = max(max_hfid,  h_fidelity(data, model, edge_mask, node_mask), key=lambda x: x[0])
                
        h_fid_dict['gstar'] += max_hfid
    for key in h_fid_dict.keys():
        h_fid_dict[key] /= len(dataset)
    return h_fid_dict
for dataset in ["aids", "BBBP", "mutag", "ba2", "Benzen", "AlkaneCarbonyl"]:
    print(gastar_eval(dataset))