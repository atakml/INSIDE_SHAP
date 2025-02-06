import os
import pickle
import torch
from tqdm import tqdm
from captum.attr import KernelShap
from shapbaseline import find_base_line
from codebase.GINUtil import select_best_model, load_gin_dataset, input_feature_size


def shap_model_function(model_to_explain, features, edge_index, baselines, target_class=None):
    if target_class is None:
        target_class = torch.argmax(torch.softmax(model_to_explain(features, edge_index), -1))

    def model_call(mask):
        extended_mask = torch.ones_like(features, device="cuda:0") * mask
        masked_features = extended_mask * features + (1 - extended_mask) * baselines
        return torch.softmax(model_to_explain(masked_features, edge_index.to("cuda:0"))[0], -1)[target_class]

    return model_call


def explain_dataset(dataset):
    model = select_best_model(dataset).to("cuda:0")
    training_loader, validation_loader, test_data_loader,_ = load_gin_dataset(dataset)
    #if not os.path.exists(f"baseline_{dataset} train epoch zero.pkl"):
    #    mask = torch.rand(len(training_loader)) < 0.1
    #print(mask.sum())
    #    baseline_vector = find_base_line(training_loader, model, max_set_size=10, num_of_noisy=10,
    #                                     feature_size=input_feature_size[dataset], mask=mask)
    #with open(f"baseline_{dataset} train epoch zero.pkl", "wb") as file:
    #    pickle.dump(baseline_vector, file)
    #else:
    if False:
        with open(f"baseline_{dataset}.pkl", "rb") as file:
            baseline_vector = pickle.load(file)
    explanation_dict = {}

    for i, data in tqdm(enumerate(training_loader)):
        edge_index, features, labels = data
        edge_index, features = edge_index[0], features[0].float()
        baseline_vector = torch.zeros(features.shape[1], device="cuda:0")
        for target_class in range(2):
            shap_model = shap_model_function(model, features, edge_index, baseline_vector, target_class)
            shap_explainer = KernelShap(shap_model)
            explanation_dict[(i, target_class)] = shap_explainer.attribute(torch.ones((1, features.shape[1]), device="cuda:0"), n_samples=512)
    print(explanation_dict)
    with open(f"{dataset} shap explanations gcn.pkl", "wb") as file:
        pickle.dump(explanation_dict, file)

explain_dataset("AlkaneCarbonyl")
