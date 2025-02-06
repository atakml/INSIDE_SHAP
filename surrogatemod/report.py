import os
import re
import numpy as np

def parse_results(file_path):
    """
    Parses the results file to extract the train, validation, and test accuracies and losses as a list of tuples.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regular expression to extract accuracies and losses
    pattern = (r"best_train_acc=tensor\(([\d\.eE+-]+).*?best_vacc=tensor\(([\d\.eE+-]+).*?best_test_acc=tensor\(([\d\.eE+-]+).*?"
               r"best_train_loss=tensor\(([\d\.eE+-]+).*?best_vloss=tensor\(([\d\.eE+-]+).*?best_test_loss=tensor\(([\d\.eE+-]+).*?\)")
    matches = re.findall(pattern, content)
    results = [(float(train_acc), float(val_acc), float(test_acc), float(train_loss), float(val_loss), float(test_loss)) for train_acc, val_acc, test_acc, train_loss, val_loss, test_loss in matches]
    return results

def calculate_statistics(results):
    """
    Calculates mean ± std for train, validation, and test accuracies and losses.
    """
    train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = zip(*results)
    train_acc_stats = (np.mean(train_accs), np.std(train_accs))
    val_acc_stats = (np.mean(val_accs), np.std(val_accs))
    test_acc_stats = (np.mean(test_accs)*100, np.std(test_accs)*100)
    train_loss_stats = (np.mean(train_losses), np.std(train_losses))
    val_loss_stats = (np.mean(val_losses), np.std(val_losses))
    test_loss_stats = (np.mean(test_losses), np.std(test_losses))
    return train_acc_stats, val_acc_stats, test_acc_stats, train_loss_stats, val_loss_stats, test_loss_stats

def find_best_validation_triple(results):
    """
    Finds the tuple with the lowest validation loss.
    """
    return min(results, key=lambda x: x[4])

def check_stability(results, threshold=0.8):
    """
    Checks if all validation accuracies in the results are above the threshold.
    """
    return all(val > threshold for _, val, _, _, _, _ in results)

def generate_latex_table(datasets_results, metric):
    """
    Generates a single LaTeX table for all datasets based on the specified metric (accuracy or loss).
    """
    rows = []
    for dataset_name, train_stats, val_stats, test_stats, best_triple, stability in datasets_results:
        train_mean_std = f"{train_stats[0]:.4f} $\\pm$ {train_stats[1]:.4f}"
        val_mean_std = f"{val_stats[0]:.4f} $\\pm$ {val_stats[1]:.4f}"
        test_mean_std = f"{test_stats[0]:.4f} $\\pm$ {test_stats[1]:4f}"
        if metric == "accuracy":
            best_train, best_val, best_test = best_triple[:3]
        else:
            best_train, best_val, best_test = best_triple[3:]
        stability_mark = "\\checkmark" if stability else ""
        #rows.append(f"{dataset_name} & {train_mean_std} & {val_mean_std} & {test_mean_std} & {best_train:.4f} & {best_val:.4f} & {best_test:.4f} & {stability_mark} \\\\ \\hline\n")
        rows.append(f" {test_mean_std}  \\\\ \\hline\n")
    table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|c|}}
\\hline
Dataset & Train {metric.capitalize()} (mean $\\pm$ std) & Valid {metric.capitalize()} (mean $\\pm$ std) & Test {metric.capitalize()} (mean $\\pm$ std) & Best Train {metric.capitalize()} & Best Valid {metric.capitalize()} & Best Test {metric.capitalize()} & Stability \\\\ \\hline
{''.join(rows)}
\\end{{tabular}}
\\caption{{Results across all datasets. Stability is marked with a checkmark if all validation accuracies exceed 0.8.}}
\\label{{tab:all_{metric}_results}}
\\end{{table}}
"""
    return table

def main():
    """
    Main function to process predefined datasets and generate separate LaTeX tables for accuracy and loss.
    """
    datasets = ["aids", "ba2", "BBBP", "mutag", "Benzen", "AlkaneCarbonyl"]
    datasets_results_acc = []
    datasets_results_loss = []
    
    for dataset_name in datasets:
        file_path = f"models/{dataset_name}_inside_results.txt"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Step 1: Parse results
        results = parse_results(file_path)
        
        # Step 2: Calculate mean ± std
        train_acc_stats, val_acc_stats, test_acc_stats, train_loss_stats, val_loss_stats, test_loss_stats = calculate_statistics(results)
        
        # Step 3: Find the best validation triple
        best_triple = find_best_validation_triple(results)
        
        # Step 4: Check stability
        stability = check_stability(results)
        
        # Collect results for the dataset
        datasets_results_acc.append((dataset_name, train_acc_stats, val_acc_stats, test_acc_stats, best_triple, stability))
        datasets_results_loss.append((dataset_name, train_loss_stats, val_loss_stats, test_loss_stats, best_triple, stability))
    
    # Step 5: Generate LaTeX tables
    latex_table_acc = generate_latex_table(datasets_results_acc, "accuracy")
    latex_table_loss = generate_latex_table(datasets_results_loss, "loss")
    print(latex_table_acc)
    print(latex_table_loss)

# Execute the script
if __name__ == "__main__":
    main()
