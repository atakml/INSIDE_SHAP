import pickle
from tqdm import tqdm
from captum.attr import KernelShap
from surrogatemod.surrogate_utils import load_gin_dataset
from surrogatemod.simplemodel.mlp import MLP, prepare_data, load_mlp
from datetime import datetime



    
def explain_dataset(dataset, method="inside", device="cuda:0"):
    training_loader, validation_loader, test_loader, rule_to_delete = load_gin_dataset(dataset, device="cpu")
    X_train, y_train = prepare_data(training_loader)
    input_size = X_train.shape[-1]
    model = load_mlp(dataset, input_size, device="cuda:0")
    explanation_dict = {}
    baseline_vector = X_train.mean(axis=0)
    print(baseline_vector.shape)
    for i in tqdm(range(len(X_train))):
        features = X_train[i].float()
        if i == 0:
            print(f"Features shape: {features.shape}, Labels shape: {y_train.shape}")

        for target_class in range(2):
            shap_explainer = KernelShap(model)
            explanation_dict[(i, target_class)] = shap_explainer.attribute(features.unsqueeze(0), target=target_class, n_samples=1024)
    with open(f"{dataset} {method} shap explanations mlp.pkl", "wb") as file:
        pickle.dump(explanation_dict, file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    args = parser.parse_args()
    explain_dataset(args.dataset_name, device=args.device, method=args.method)
