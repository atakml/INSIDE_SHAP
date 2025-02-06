import os
model_dict = {"Decision Tree": "dtree", "Linear Model": "linear", "MLP": "mlp"}
datasets = ["aids", "ba2", "BBBP", "mutag", "Benzen", "AlkaneCarbonyl"]
for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    with open("surrogatemod/simplemodel/results.txt", "a") as file:
        file.write(dataset + "\n")    
    for i in range(5):
        for model_name, model_file in model_dict.items():
            if model_name != "Linear Model":
                continue
            print(f"  Running model: {model_name}")
            with open("surrogatemod/simplemodel/results.txt", "a") as file:
                file.write(f"  {model_name}\n")
            command = f"python surrogatemod/simplemodel/{model_file}.py {dataset}"
            os.system(command)
            print(f"  Finished running model: {model_name}")
        print(f"Finished processing dataset: {dataset}")
