import os
from tqdm import tqdm 
#run_dict = {"aids": 1, "ba2": 4, "BBBP": 3, "Benzen": 4, "AlkaneCarbonyl": 4, "mutag": 3}
run_dict = {"ba2": 5}

for dataset, number_of_runs in run_dict.items():
    print(f"{dataset=}")
    for _ in tqdm(range(number_of_runs)):
        command = f"python modelmod/train_gnn.py {dataset}"
        os.system(command)
