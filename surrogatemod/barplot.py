import os
import re

import matplotlib.pyplot as plt

def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    test_acc_pattern = re.compile(r'best_test_acc=tensor\((\d+\.\d+), device=\'cuda:0\'\)')
    test_loss_pattern = re.compile(r'best_test_loss=tensor\((\d+\.\d+), device=\'cuda:0\'\)')
    
    test_accs = [float(match) for match in test_acc_pattern.findall(data)]
    test_losses = [float(match) for match in test_loss_pattern.findall(data)]
    
    return test_accs, test_losses

def plot_metrics(datasets):
    accs = []
    losses = []
    labels = []

    for dataset in datasets:
        file1 = f"models/{dataset}_inside_results.txt"
        file2 = f"models/random features/{dataset}_inside_results.txt"
        
        acc1, loss1 = extract_metrics(file1)
        acc2, loss2 = extract_metrics(file2)
        
        accs.append((acc1, acc2))
        losses.append((loss1, loss2))
        labels.append(dataset)
    
    # Plot accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 10))
    positions = range(len(datasets))
    width = 0.35
    ax.bar([p - width/2 for p in positions], [sum(acc[0])/len(acc[0]) for acc in accs], width=width, label="GCN Surrogate with Pattern Activations", color='C0')
    ax.bar([p + width/2 for p in positions], [sum(acc[1])/len(acc[1]) for acc in accs], width=width, label= "GCN Surrogate with Random Activations", color='C1')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Aids', 'BA2-Motifs', 'BBBP', 'Mutagenicity', 'Benzene', 'Alkane-Carbonyl'], rotation=45, ha='right', fontsize=35)
    ax.set_ylabel('Test Accuracy', fontsize=30)
    ax.set_title('Test Accuracy Comparison', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20, loc='lower right')
    plt.tight_layout()
    plt.savefig('surrogatemod/acc_barplot.png')
    
    # Plot loss comparison
    fig, ax = plt.subplots(figsize=(10,10))
    ax.bar([p - width/2 for p in positions], [sum(loss[0])/len(loss[0]) for loss in losses], width=width, label='GCN Surrogate with Pattern Activations', color='C0')
    ax.bar([p + width/2 for p in positions], [sum(loss[1])/len(loss[1]) for loss in losses], width=width, label= "GCN Surrogate with Random Activations", color='C1')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Aids', 'BA2-Motifs', 'BBBP', 'Mutagenicity', 'Benzene', 'Alkane-Carbonyl'], rotation=45, ha='right', fontsize=35)
    ax.set_ylabel('Test Loss', fontsize=30)
    ax.set_title('Test Loss Comparison', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20, loc='upper left')
    plt.tight_layout()
    plt.savefig('surrogatemod/loss_barplot.png')

datasets = ["aids", "ba2", "BBBP", "mutag", "Benzen", "AlkaneCarbonyl"]  # Replace with your dataset names
plot_metrics(datasets)