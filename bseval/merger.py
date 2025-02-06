import pickle 
import os
from tqdm import tqdm
directory = "/home/ata/shap_extend/subgraphA/"
save_directory = "/home/ata/shap_inside/"
save_name = "subgraph_AlkaneCarbonyl.pkl"
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


merged_dict = {}

for file_name in tqdm(files):
    with open(directory + file_name, "rb") as file:
        data = pickle.load(file)
    merged_dict[int(file_name.split(".")[0])] = data


with open(save_directory + save_name, "wb") as file:
    pickle.dump(merged_dict, file)

