import argparse
import time
import subprocess
import torch 
import os
import pickle
import torch
from tqdm import tqdm
from captum.attr import KernelShap

def shap_model_function(model_to_explain, features, edge_index, baselines, target_class=None, device="cuda:0"):
    if target_class is None:
        target_class = torch.argmax(torch.softmax(model_to_explain(features, edge_index), -1))

    def model_call(mask):
        extended_mask = torch.ones_like(features, device=device) * mask
        masked_features = extended_mask * features + (1 - extended_mask) * baselines
        output = torch.softmax(model_to_explain(masked_features, edge_index.to(device))[0], -1)[target_class]
        return output

    return model_call

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("index", type=int, help="Index value")
    parser.add_argument("target_class", type=int, help="Target class value")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    i = args.index
    target_class = args.target_class
    device = args.device
    method = args.method
    with open(f"{dataset_name}_{i}.pkl", "rb") as f:
        data, model = pickle.load(f)
    edge_index, features, labels = data
    edge_index, features = edge_index[0], features[0].float()
    baseline_vector = torch.zeros(features.shape[1], device=device)
    shap_model = shap_model_function(model, features, edge_index, baseline_vector, target_class, device=device)
    shap_explainer = KernelShap(shap_model)
    explaination = shap_explainer.attribute(torch.ones((1, features.shape[1]), device=device), n_samples=1024)
    #save it in this file f"out_{dataset}_{i}_{target}.pkl"
    with open(f"out_{dataset_name}_{i}_{target_class}.pkl", "wb") as f:
        pickle.dump(explaination, f)
