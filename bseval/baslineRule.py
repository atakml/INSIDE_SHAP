from graphxai.datasets.real_world.benzene.benzene import Benzene
from graphxai.datasets.real_world.fluoride_carbonyl.fluoride_carbonyl import FluorideCarbonyl
from graphxai.datasets.real_world.alkane_carbonyl.alkane_carbonyl import AlkaneCarbonyl
datasets = {
"Benzen": Benzene(split_sizes=(0.8,0.1,0.1), device="cuda:0"),
#"FluorideCarbonyl": FluorideCarbonyl(split_sizes=(0.8,0.1,0.1), device="cuda:0"),
#"AlkaneCarbonyl": AlkaneCarbonyl(split_sizes=(0.8,0.1,0.1))
}

from os import listdir
from os.path import isfile, join
def load_best_model(dataset_name, model_type):
    model = model_type(14, 20, 2)
    mypath = "models"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    models = list(filter(lambda x: dataset_name in x, onlyfiles))
    best_model_name = max(models, key=lambda x: float(x.split("_")[-1].split(".")[0]))
    path = f"models/{best_model_name}"
    model.load_state_dict(torch.load(path))
    return model

import pickle
with open(f"baselines_graphxai_gcn_benzen.pkl", "rb") as file:
    explain_dict = pickle.load(file)



   