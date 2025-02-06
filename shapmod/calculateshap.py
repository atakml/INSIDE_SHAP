import os
import pickle
import torch
from tqdm import tqdm
from captum.attr import KernelShap
#from shapbaseline import find_base_line
from surrogatemod.surrogate_utils import select_best_model, load_gin_dataset, input_feature_size


def shap_model_function(model_to_explain, features, edge_index, baselines, target_class=None, device="cuda:0"):
    if target_class is None:
        target_class = torch.argmax(torch.softmax(model_to_explain(features, edge_index), -1))

    def model_call(mask):
        extended_mask = torch.ones_like(features, device=device) * mask
        masked_features = extended_mask * features + (1 - extended_mask) * baselines
        output = torch.softmax(model_to_explain(masked_features, edge_index.to(device))[0], -1)[target_class]
        return output

    return model_call


def explain_dataset(dataset, method="inside", device="cuda:0"):
    model = select_best_model(dataset).to(device)
    training_loader, validation_loader, test_data_loader,_ = load_gin_dataset(dataset, method=method, device=device)
    explanation_dict = {}


    def process_data(dataset, i, data):
        file_name = f"{dataset}_{i}.pkl"
        with open(file_name, "wb") as file:
            pickle.dump((data,model), file)

    def check_memory():
        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        return free_memory / (1024 ** 3)  # Convert to GB

    def count_active_processes():
        result = subprocess.run(["pgrep", "-f", "shapmod/shapcalculatesubprocess.py"], stdout=subprocess.PIPE)
        return len(result.stdout.splitlines())
    flag = True
    while flag:
        flag = False
        for i, data in tqdm(enumerate(training_loader)):
            for target in range(2):
                file_name = f"out_{dataset}_{i}_{target}.pkl"
                if os.path.exists(file_name):
                    print(f"File {file_name} already exists, skipping...")
                    continue
                flag = True
                process_data(dataset, i, data)
                while check_memory() < 10 or count_active_processes() >= 25:
                    print("Waiting for resources...")
                    for j in range(i):
                        for t in range(2):
                            result = subprocess.run(["pgrep", "-f", f"shapmod/shapcalculatesubprocess.py {dataset} {j} {t}"], stdout=subprocess.PIPE)
                            if not result.stdout:
                                try:
                                    os.remove(f"{dataset}_{j}_{t}.pkl")
                                except FileNotFoundError:
                                    pass
                    time.sleep(5)
                subprocess.Popen(["python", "shapmod/shapcalculatesubprocess.py", dataset, str(i), str(target)])

    for i in range(len(training_loader)):        
        file_name = f"out_{dataset}_{i}_0.pkl"
        with open(file_name, "rb") as file:
            explanation_dict[(i, 0)] = pickle.load(file)
        
        file_name = f"out_{dataset}_{i}_1.pkl"
        with open(file_name, "rb") as file:
            explanation_dict[(i, 1)] = pickle.load(file)
        #delete the files
        os.remove(f"out_{dataset}_{i}_0.pkl")
        os.remove(f"out_{dataset}_{i}_1.pkl")
    #save the explanation dictionary in the file with the name f"{dataset} {method} shap explanations gcn 1024.pkl"
    with open(f"{dataset} {method} shap explanations gcn 1024.pkl", "wb") as file:
        pickle.dump(explanation_dict, file)

if __name__ == "__main__":
    import argparse
    import time
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    args = parser.parse_args()
    explain_dataset(args.dataset_name, device=args.device, method=args.method)
