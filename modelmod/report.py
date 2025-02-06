import os
import re
import numpy as np

def extract_accuracies_from_file(file_path):
    """
    Extracts all train, validation, and test accuracies from multiple lines in a file.
    """
    train_accs = []
    val_accs = []
    test_accs = []

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'train_acc=([\d.]+)\s+val_acc=([\d.]+)\s+test_acc=([\d.]+)', line)
            if match:
                train_accs.append(float(match.group(1)))
                val_accs.append(float(match.group(2)))
                test_accs.append(float(match.group(3)))

    return train_accs, val_accs, test_accs

def process_datasets(datasets, directory):
    """
    Processes all datasets and generates statistics for each dataset.
    """
    results = {}

    for dataset in datasets:
        all_train_accs = []
        all_val_accs = []
        all_test_accs = []

        # Iterate over all relevant files for the current dataset
        for filename in os.listdir(directory):
            if filename.startswith(f"{dataset} training results") and filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                train_accs, val_accs, test_accs = extract_accuracies_from_file(file_path)

                # Accumulate all extracted accuracies
                all_train_accs.extend(train_accs)
                all_val_accs.extend(val_accs)
                all_test_accs.extend(test_accs)

        # Calculate mean, std, and best values
        if all_train_accs:
            mean_train = np.mean(all_train_accs)
            std_train = np.std(all_train_accs)
            best_train = np.max(all_train_accs)

            mean_val = np.mean(all_val_accs)
            std_val = np.std(all_val_accs)
            best_val = np.max(all_val_accs)

            mean_test = np.mean(all_test_accs)
            std_test = np.std(all_test_accs)
            best_test = np.max(all_test_accs)

            # Store the results
            results[dataset] = {
                'mean_train': mean_train, 'std_train': std_train, 'best_train': best_train,
                'mean_val': mean_val, 'std_val': std_val, 'best_val': best_val,
                'mean_test': mean_test, 'std_test': std_test, 'best_test': best_test
            }

    return results

def generate_latex_table(results):
    """
    Generates a LaTeX table from the results.
    """
    latex_str = "\\begin{table}[ht]\n\\centering\n"
    latex_str += "\\begin{tabular}{|l|c|c|c|c|c|c|}\n"
    latex_str += "\\hline\n"
    latex_str += ("Dataset & Train Acc (mean $\\pm$ std) & "
                  "Validation Acc (mean $\\pm$ std) & Test Acc (mean $\\pm$ std) & "
                  "Best Train Acc & Best Val Acc & Best Test Acc \\\\ \n")
    latex_str += "\\hline\n"

    for dataset, stats in results.items():
        latex_str += (f"{dataset} & "
                      f"{stats['mean_train']:.5f} $\\pm$ {stats['std_train']:.5f} & "
                      f"{stats['mean_val']:.5f} $\\pm$ {stats['std_val']:.5f} & "
                      f"{stats['mean_test']:.5f} $\\pm$ {stats['std_test']:.5f} & "
                      f"{stats['best_train']:.5f} & {stats['best_val']:.5f} & {stats['best_test']:.5f} \\\\ \n")

    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\caption{Training, Validation, and Testing Accuracies}\n"
    latex_str += "\\label{table:results}\n"
    latex_str += "\\end{table}\n"

    return latex_str

# Example usage
if __name__ == "__main__":
    # List of dataset names
    datasets = ['aids', 'ba2', 'BBBP', "mutag", "Benzen", "AlkaneCarbonyl"]
    
    # Directory where your result files are stored
    directory = '.'

    results = process_datasets(datasets, directory)
    latex_table = generate_latex_table(results)
    
    # Output the LaTeX table
    print(latex_table)


