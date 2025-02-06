from GModel import GModel
from graphxai.gnn_models.graph_classification.gcn import GCN_3layer
from os import listdir
from os.path import isfile, join
from ExplanationEvaluation.models.model_selector import model_selector
import torch
from pathlib import Path
parent_directory = str(Path.cwd().parent)

def load_gnn(dataset_name, pretrained=True, device="cuda:0"):
    if dataset_name in ["Benzen", "AlkaneCarbonyl"]:
        gnn_model = GModel(14, 20, 2)
        if pretrained:
            mypath = f"{parent_directory}/shap_inside/modelmod/models/"
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            models = list(filter(lambda x: dataset_name in x, onlyfiles))
            models = list(filter(lambda x: "f1" in x and "unique" in x, models))
            best_model_name = max(models, key=lambda x: float(x.split("_")[-1].split(".")[0]))
            path = f"{parent_directory}/shap_inside/modelmod/models/{best_model_name}"
            gnn_model = torch.load(path)
    else:
        gnn_model = model_selector("GNN",
                                           dataset_name,
                                           pretrained=pretrained,
                                           return_checkpoint=False)
    gnn_model = gnn_model.to(device)

    return gnn_model
