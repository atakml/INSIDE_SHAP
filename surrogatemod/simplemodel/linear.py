"""from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from codebase.GINUtil import load_gin_dataset
from torch.nn.functional import softmax
from surrogatemod.surrogate_utils import load_gin_dataset, prepare_data_for_simpler_model, evaluate_simpler_model_on_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()
    
    dataset = args.dataset_name
    training_loader, validation_loader, test_loader, rule_to_delete = load_gin_dataset(dataset, device="cpu")
    X_train, train_labels = prepare_data_for_simpler_model(training_loader)
    linear_model = LinearRegression()
    linear_model.fit(X_train, train_labels)
    train_labels = linear_model.predict(X_train)

    split_list = ["training", "validation", "test"]
    for data_split in split_list:
        data_loader = eval(f"{data_split}_loader")
        X, labels = prepare_data_for_simpler_model(data_loader)
        print(f"{data_split}_acc: {evaluate_simpler_model_on_data(linear_model, X, labels)}")"""

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from surrogatemod.surrogate_utils import load_gin_dataset, prepare_data_for_simpler_model, evaluate_simpler_model_on_data, optimize_model_with_grid_search
from torch.nn.functional import softmax

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()

    dataset = args.dataset_name
    training_loader, validation_loader, test_loader, rule_to_delete = load_gin_dataset(dataset, device="cpu")
    X_train, train_labels = prepare_data_for_simpler_model(training_loader)
    X_val, val_labels = prepare_data_for_simpler_model(validation_loader)

    # Choose Ridge or Lasso
    model = Lasso()

    # Define the hyperparameters for optimization
    param_grid = {
    'alpha': [0.1, 1, 10, 100],
    'fit_intercept': [True, False]    
    }

    # Perform Grid Search Optimization
    best_model, best_params = optimize_model_with_grid_search(
        model, X_train, train_labels, X_val, val_labels, param_grid
    )

    # Setting the best parameters and fitting the model
    best_model = Lasso()
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

