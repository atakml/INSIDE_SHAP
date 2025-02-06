#for dataset in ["mutag", "aids", "ba2", "BBBP"]:
"""if True:
    dataset = "AlkaneCarbonyl"
    with open(f"/home/ata/shap_inside/{dataset} shap explanations baseline zero feature both.pkl", "rb") as file:
        shap_dict = pickle.load(file)

    plt.clf()
    shap_matrix = [shap_dict[(i,1)] for i,_ in sorted(shap_dict.keys())]
    shap_matrix = torch.vstack(shap_matrix).cpu().numpy()
    plot = violin(shap_matrix, show=False, max_display=shap_matrix.shape[1])
    plt.savefig(f"plotss/{dataset}_violin_1.png")

:q

"""
import numpy as np
from shap.plots import violin
from shap import summary_plot
import pickle
import torch
import matplotlib.pyplot as plt
import os
from surrogatemod.surrogate_utils import load_gin_dataset
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--model', type=str, default='gcn', help='Model to use (default: gnn)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda)')
    args = parser.parse_args()
    dataset = args.dataset_name
    
    with open(f"{dataset} inside shap explanations {args.model}.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    training_loader, validation_loader, test_data_loader, rule_to_delete = load_gin_dataset(dataset, device=args.device)    
    features_class = []
    features_occ = []
    features_supp = []
    for data in training_loader:
        edge_index, feature, label = data
        vector = torch.zeros(feature[0].shape[-1]) if label[0][0][0] > label[0][0][1] else torch.ones(feature.shape[-1])
        features_occ.append(feature[0].int().sum(axis=0).int().cpu().numpy())
        features_supp.append(feature[0].int().sum(axis=0).bool().int().cpu().numpy())
        features_class.append(vector)
    features_class = np.array(features_class)
    features_supp = np.array(features_supp)
    features_occ = np.array(features_occ)
    plt.clf()
    shap_matrix = [shap_dict[(i,1)] for i,j in sorted(shap_dict.keys()) if j==1]
    shap_matrix = torch.vstack(shap_matrix).cpu().numpy()
    feature_names = [f"Pattern{i}" for i in range(shap_matrix.shape[1])]
    # Plot for class
    plot = summary_plot(shap_matrix, features_class, plot_type="violin", show=False, feature_names=feature_names, color_bar_label="Class")
    color_bar = plt.gcf().axes[-1]  # Access the color bar (last axis in the figure)
    color_bar.set_yticklabels(['Negative', 'Positive'])  # Modify tick labels
    plt.savefig(f"plotss/{dataset}_violin_1_class_{args.model}.png")

    # Plot for occurrence
    plt.clf()
    plot = summary_plot(shap_matrix, features_occ, plot_type="violin", show=False, feature_names=feature_names, color_bar_label="Occurrence")
    color_bar = plt.gcf().axes[-1]  # Access the color bar (last axis in the figure)
    color_bar.set_yticklabels(['Low', 'High'])  # Modify tick labels
    plt.savefig(f"plotss/{dataset}_violin_1_occurrence_{args.model}.png")

    # Plot for support
    plt.clf()
    plot = summary_plot(shap_matrix, features_supp, plot_type="violin", show=False, feature_names=feature_names, color_bar_label="Support")
    color_bar = plt.gcf().axes[-1]  # Access the color bar (last axis in the figure)
    color_bar.set_yticklabels(['Absent', 'Present'])  # Modify tick labels
    plt.savefig(f"plotss/{dataset}_violin_1_support_{args.model}.png")
    
