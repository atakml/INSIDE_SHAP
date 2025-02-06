"""from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from surrogatemod.surrogate_utils import load_gin_dataset, prepare_data_for_simpler_model, evaluate_simpler_model_on_data
from torch.nn.functional import softmax



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()
    
    dataset = args.dataset_name
    training_loader, validation_loader, test_loader, rule_to_delete = load_gin_dataset(dataset, device="cpu")
    X_train, train_labels = prepare_data_for_simpler_model(training_loader)
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(X_train, train_labels)

    split_list = ["training", "validation", "test"]
    for data_split in split_list:
        data_loader = eval(f"{data_split}_loader")
        X, labels = prepare_data_for_simpler_model(data_loader)
        print(f"{data_split}_acc: {evaluate_simpler_model_on_data(dtr_model, X, labels)}")"""
from sklearn.tree import DecisionTreeRegressor
from surrogatemod.surrogate_utils import load_gin_dataset, prepare_data_for_simpler_model, evaluate_simpler_model_on_data, optimize_model_with_grid_search
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()
    
    dataset = args.dataset_name
    training_loader, validation_loader, test_loader, rule_to_delete = load_gin_dataset(dataset, device="cpu")
    X_train, train_labels = prepare_data_for_simpler_model(training_loader)
    X_val, val_labels = prepare_data_for_simpler_model(validation_loader)

    # Define the Decision Tree Regressor
    dtr_model = DecisionTreeRegressor()

    # Define the hyperparameters for optimization
    param_grid = {
        'max_depth': [3, 5, 10, None],  # Range of depths for the tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 5],    # Minimum samples required to be at a leaf node
        'max_features': [None, 'sqrt', 'log2']  # Number of features to consider for splitting
    }

    # Perform Grid Search Optimization
    best_model, best_params = optimize_model_with_grid_search(
        dtr_model, X_train, train_labels, X_val, val_labels, param_grid
    )
    best_model = DecisionTreeRegressor()
    best_model.set_params(**best_params)
    best_model.fit(X_train, train_labels)

    print(f"Best Hyperparameters: {best_params}")
    
    # Evaluate the best model on the training, validation, and test sets
    split_list = ["training", "validation", "test"]
    for data_split in split_list:
        data_loader = eval(f"{data_split}_loader")
        X, labels = prepare_data_for_simpler_model(data_loader)
        acc = evaluate_simpler_model_on_data(best_model, X, labels)
        with open("surrogatemod/simplemodel/results.txt", "a") as f:
            f.write(f"{data_split}_acc: {acc}\n")
    
