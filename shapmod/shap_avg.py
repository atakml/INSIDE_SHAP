from bseval.utils import load_dataset_to_explain
import pickle
import torch

dataset_name = "Benzen"
pattern_list = [30, 33, 35, 41, 42, 43, 44, 45, 46,47,48, 51]
train_loader, gnn_model, graphs, features, labels, train_mask = load_dataset_to_explain(dataset_name)
with open(f"{dataset_name} shap explanations gcn.pkl", "rb") as file:
        shap_dict = pickle.load(file)

number_of_patterns = list(shap_dict.values())[0].shape[-1]
print(number_of_patterns)
avg = torch.zeros((number_of_patterns, 2), device="cuda:0")

train_indices = torch.where(torch.tensor(train_mask))[0]
from tqdm import tqdm 
#print(labels)
cnt = [0,0]
for i, data in tqdm(enumerate(train_loader)):
    label =labels[train_indices[i]]
    #index_to_add = 0 if label[0] > label[1] else 1
    index_to_add = label.item()
    for j in range(len(shap_dict[(i, 1)][0])):
        avg[j][index_to_add] += shap_dict[(i,1)][0][j]
    cnt[index_to_add] += 1

avg[:, 0] /= cnt[0]
avg[:, 1] /= cnt[1]

print(avg[pattern_list])
