"""

def parse_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    datasets = re.split(r'\ndataset:', content)
    results = {}
    for dataset in datasets:
        lines = dataset.strip().split('\n')
        dataset_name = lines[0].strip()
        results[dataset_name] = {}
        model_name = None
        for line in lines[1:]:
            if line.strip():
                if not line.startswith(' '):
                    model_name = line.strip()
                    results[dataset_name][model_name] = {}
                else:
                    key, value = line.split(':')
                    if model_name in ['Decision Tree', 'Linear Model']:
                        loss, acc = value.strip().split(',')
                        results[dataset_name][model_name][key.strip()] = (eval(loss.strip()), eval(acc.strip()))
                    else:
                        values = value.split(',')
                        for v in values:
                            k, val = v.split('(')
                            results[dataset_name][model_name][key.strip() + ' ' + k.strip()] = float(val.strip().rstrip(')'))
    return results

def parse_inside_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    pattern = re.compile(r'best_train_acc=tensor\(([\d.e-]+),.*?best_vacc=tensor\(([\d.e-]+),.*?best_test_acc=tensor\(([\d.e-]+),.*?best_train_loss=tensor\(([\d.e-]+),.*?best_vloss=tensor\(([\d.e-]+),.*?best_test_loss=tensor\(([\d.e-]+)', re.DOTALL)
    matches = pattern.findall(content)
    results = []
    for match in matches:
        results.append({
            'train_acc': float(match[0]),
            'val_acc': float(match[1]),
            'test_acc': float(match[2]),
            'train_loss': float(match[3]),
            'val_loss': float(match[4]),
            'test_loss': float(match[5])
        })
    return results

def calculate_decline(acc1, acc2, loss1, loss2):
    acc_decline = (1 - acc1 / acc2) * 100
    loss_decline = (loss1 / loss2) * 100
    return acc_decline, loss_decline

def generate_report(results, inside_results):
    acc_table = []
    loss_table = []
    for dataset, models in results.items():
        best_inside_result = min(inside_results[dataset], key=lambda x: x['val_loss'])
        for model, metrics in models.items():
            if model in ['Decision Tree', 'Linear Model']:
                train_acc_decline, train_loss_decline = calculate_decline(metrics['training_acc'][1].item(), best_inside_result['train_acc'], metrics['training_acc'][0].item(), best_inside_result['train_loss'])
                val_acc_decline, val_loss_decline = calculate_decline(metrics['validation_acc'][1].item(), best_inside_result['train_acc'], metrics['validation_acc'][0].item(), best_inside_result['val_loss'])
                test_acc_decline, test_loss_decline = calculate_decline(metrics['test_acc'][1].item(), best_inside_result['test_acc'], metrics['test_acc'][0].item(), best_inside_result['test_loss'])
            else:
                train_acc_decline, train_loss_decline = calculate_decline(metrics['Best Train Accuracy'], best_inside_result['train_acc'], metrics['Best Train Loss'], best_inside_result['train_loss'])
                val_acc_decline, val_loss_decline = calculate_decline(metrics['Best Val Accuracy'], best_inside_result['train_acc'], metrics['Best Val Loss'], best_inside_result['val_loss'])
                test_acc_decline, test_loss_decline = calculate_decline(metrics['Best Test Accuracy'], best_inside_result['test_acc'], metrics['Best Test Loss'], best_inside_result['test_loss'])
            acc_table.append([model, dataset, train_acc_decline, val_acc_decline, test_acc_decline])
            loss_table.append([model, dataset, train_loss_decline, val_loss_decline, test_loss_decline])
    return acc_table, loss_table

def save_table(table, file_path):
    for row in table:
        print(','.join(map(str, row)))

results = parse_results('surrogatemod/simplemodel/results.txt')
inside_results = {dataset.split()[0]: parse_inside_results(f'models/{dataset.split()[0]}_inside_results.txt') for dataset in results.keys()}
acc_table, loss_table = generate_report(results, inside_results)
save_table(acc_table, 'accuracy_decline_report.csv')
save_table(loss_table, 'loss_decline_report.csv')
def save_table_as_latex(table, file_path, caption, label):
    df = pd.DataFrame(table, columns=['Model', 'Dataset', 'Train Decline (%)', 'Valid Decline (%)', 'Test Decline (%)'])
    with open(file_path, 'w') as file:
        file.write(df.to_latex(index=False, caption=caption, label=label, escape=False))

save_table_as_latex(acc_table, 'accuracy_decline_report.tex', 'Accuracy Decline Report', 'tab:accuracy_decline')
save_table_as_latex(loss_table, 'loss_decline_report.tex', 'Loss Decline Report', 'tab:loss_decline')"""
import re
import torch
import pandas as pd
from tabulate import tabulate 

ref_acc = {
    'aids': {'train': 0.9969, 'val': 0.9900, 'test': 0.9900},
    'ba2': {'train': 0.9962, 'val': 0.9900, 'test': 1.0000},
    'BBBP': {'train': 0.9909, 'val': 0.9573, 'test': 0.9695},
    'mutag': {'train': 0.9686, 'val': 0.9562, 'test': 0.9332},
    'Benzen': {'train': 0.9877, 'val': 0.9780, 'test': 0.9723},
    'AlkaneCarbonyl': {'train': 0.9989, 'val': 1.0000, 'test': 1.0000}
}

ref_loss = {
    'aids': {'train': 0.0031, 'val': 0.0182, 'test': 0.0152},
    'ba2': {'train': 0.0002, 'val': 0.0004, 'test': 0.0003},
    'BBBP': {'train': 0.0015, 'val': 0.0103, 'test': 0.0094},
    'mutag': {'train': 0.0080, 'val': 0.0188, 'test': 0.0199},
    'Benzen': {'train': 0.0044, 'val': 0.0140, 'test': 0.0141},
    'AlkaneCarbonyl': {'train': 0.0001, 'val': 0.0003, 'test': 0.0004}
}

# Results extracted from the provided data
results = {
    'aids': {
        'Decision Tree': {'train_acc': 0.9812, 'val_acc': 0.9500, 'test_acc': 0.9550, 
                          'train_loss': 0.0215, 'val_loss': 0.1170, 'test_loss': 0.1163},
        'Linear Model': {'train_acc': 0.8725, 'val_acc': 0.8850, 'test_acc': 0.8250, 
                         'train_loss': None, 'val_loss': 0.2046, 'test_loss': 0.2486},
        'MLP': {'train_acc': 0.997500, 'val_acc': 0.995000, 'test_acc': 1.000000, 
                'train_loss': 0.004528, 'val_loss': 0.020154, 'test_loss': 0.010268}
    },
    'ba2': {
        'Decision Tree': {'train_acc': 0.9887, 'val_acc': 0.9900, 'test_acc': 0.9800, 
                          'train_loss': 0.0015, 'val_loss': 0.0017, 'test_loss': 0.0040},
        'Linear Model': {'train_acc': 0.9837, 'val_acc': 0.9600, 'test_acc': 0.9600, 
                         'train_loss': 0.0099, 'val_loss': 0.0110, 'test_loss': 0.0111},
        'MLP': {'train_acc': 0.992500, 'val_acc': 0.980000, 'test_acc': 0.990000, 
                'train_loss': 0.001103, 'val_loss': 0.002126, 'test_loss': 0.001697}
    },
    'BBBP': {
        'Decision Tree': {'train_acc': 0.9726, 'val_acc': 0.9207, 'test_acc': 0.9573, 
                          'train_loss': 0.0075, 'val_loss': 0.0384, 'test_loss': 0.0360},
        'Linear Model': {'train_acc': 0.8095, 'val_acc': 0.7683, 'test_acc': 0.8476, 
                         'train_loss': 0.2341, 'val_loss': 0.2240, 'test_loss': 0.1901},
        'MLP': {'train_acc': 0.985518, 'val_acc': 0.945122, 'test_acc': 0.963415, 
                'train_loss': 0.004238, 'val_loss': 0.028525, 'test_loss': 0.016676}
    },
    'mutag': {
        'Decision Tree': {'train_acc': 0.9611, 'val_acc': 0.9009, 'test_acc': 0.8571, 
                          'train_loss': 0.0097, 'val_loss': 0.0608, 'test_loss': 0.0796},
        'Linear Model': {'train_acc': 0.6114, 'val_acc': 0.6014, 'test_acc': 0.5899, 
                         'train_loss': 0.2749, 'val_loss': 0.3022, 'test_loss': 0.2713},
        'MLP': {'train_acc': 0.906313, 'val_acc': 0.896313, 'test_acc': 0.882488, 
                'train_loss': 0.048089, 'val_loss': 0.066551, 'test_loss': 0.056283}
    },
    'Benzen': {
        'Decision Tree': {'train_acc': 0.9664, 'val_acc': 0.9341, 'test_acc': 0.9347, 
                          'train_loss': 0.0182, 'val_loss': 0.0584, 'test_loss': 0.0605},
        'Linear Model': {'train_acc': 0.8056, 'val_acc': 0.7711, 'test_acc': 0.7773, 
                         'train_loss': 0.3550, 'val_loss': 0.3481, 'test_loss': 0.3425},
        'MLP': {'train_acc': 0.969431, 'val_acc': 0.960694, 'test_acc': 0.951060, 
                'train_loss': 0.019040, 'val_loss': 0.035711, 'test_loss': 0.034985}
    },
    'AlkaneCarbonyl': {
        'Decision Tree': {'train_acc': 0.9989, 'val_acc': 1.0000, 'test_acc': 1.0000, 
                          'train_loss': 0.0003, 'val_loss': 0.0007, 'test_loss': 0.0014},
        'Linear Model': {'train_acc': 0.6089, 'val_acc': 0.6339, 'test_acc': 0.6460, 
                         'train_loss': 0.1557, 'val_loss': 0.1555, 'test_loss': 0.1595},
        'MLP': {'train_acc': 0.998889, 'val_acc': 1.000000, 'test_acc': 0.982301, 
                'train_loss': 0.000463, 'val_loss': 0.001067, 'test_loss': 0.001597}
    }
}



def calculate_decline_acc(acc, ref_acc):
    return (1 - acc / ref_acc) * 100

def calculate_decline_loss(loss, ref_loss):
    try:
        return (loss / ref_loss - 1) * 100
    except:
        return "Nan"

def generate_decline_tables(results, ref_acc, ref_loss):
    acc_table = []
    loss_table = []
    for dataset, models in results.items():
        for model, metrics in models.items():
            acc_declines = [
                calculate_decline_acc(metrics['train_acc'], ref_acc[dataset]['train']),
                calculate_decline_acc(metrics['val_acc'], ref_acc[dataset]['val']),
                calculate_decline_acc(metrics['test_acc'], ref_acc[dataset]['test'])
            ]
            loss_declines = [
                calculate_decline_loss(metrics['train_loss'], ref_loss[dataset]['train']),
                calculate_decline_loss(metrics['val_loss'], ref_loss[dataset]['val']),
                calculate_decline_loss(metrics['test_loss'], ref_loss[dataset]['test'])
            ]
            acc_table.append([model, dataset] + acc_declines)
            loss_table.append([model, dataset] + loss_declines)
    return acc_table, loss_table

def save_table_as_latex(table, file_path, caption, label):
    df = pd.DataFrame(table, columns=['Model', 'Dataset', 'Train Decline (%)', 'Valid Decline (%)', 'Test Decline (%)'])
    with open(file_path, 'w') as file:
        file.write(df.to_latex(index=False, caption=caption, label=label, escape=False))

def generate_tables(results, ref_acc, ref_loss):
    acc_table = []
    loss_table = []
    for dataset, models in results.items():
        for model, metrics in models.items():
            acc_values = [
                metrics['train_acc'],
                metrics['val_acc'],
                metrics['test_acc']
            ]
            loss_values = [
                metrics['train_loss'],
                metrics['val_loss'],
                metrics['test_loss']
            ]
            acc_table.append([model, dataset] + acc_values)
            loss_table.append([model, dataset] + loss_values)
        # Add reference values (GNN)
        acc_table.append(['GNN', dataset, ref_acc[dataset]['train'], ref_acc[dataset]['val'], ref_acc[dataset]['test']])
        loss_table.append(['GNN', dataset, ref_loss[dataset]['train'], ref_loss[dataset]['val'], ref_loss[dataset]['test']])
    return acc_table, loss_table

acc_table, loss_table = generate_tables(results, ref_acc, ref_loss)

def save_table_as_latex_multirow(acc_table, loss_table, acc_file_path, loss_file_path):
    acc_df = pd.DataFrame(acc_table, columns=['Model', 'Dataset', 'Train Accuracy (%)', 'Valid Accuracy (%)', 'Test Accuracy (%)'])
    loss_df = pd.DataFrame(loss_table, columns=['Model', 'Dataset', 'Train Loss', 'Valid Loss', 'Test Loss'])

    with open(acc_file_path, 'w') as acc_file:
        acc_file.write("\\begin{table}[ht]\n\\centering\n\\caption{Accuracy Report}\n\\label{tab:accuracy}\n")
        acc_file.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
        acc_file.write("Dataset & Model & Train Accuracy (\\%) & Valid Accuracy (\\%) & Test Accuracy (\\%) \\\\\n\\hline\n")
        for dataset in acc_df['Dataset'].unique():
            models = acc_df[acc_df['Dataset'] == dataset]
            acc_file.write(f"\\multirow{{{len(models)}}}{{*}}{{{dataset}}} ")
            for i, row in models.iterrows():
                if i != models.index[0]:
                    acc_file.write(" & ")
                acc_file.write(f"& {row['Model']} & {float(row['Train Accuracy (%)']):.2f} & {float(row['Valid Accuracy (%)']):.2f} & {float(row['Test Accuracy (%)']):.2f} \\\\\n")
            acc_file.write("\\hline\n")
        acc_file.write("\\end{tabular}\n\\end{table}")

    with open(loss_file_path, 'w') as loss_file:
        loss_file.write("\\begin{table}[ht]\n\\centering\n\\caption{Loss Report}\n\\label{tab:loss}\n")
        loss_file.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
        loss_file.write("Dataset & Model & Train Loss & Valid Loss & Test Loss \\\\\n\\hline\n")
        for dataset in loss_df['Dataset'].unique():
            models = loss_df[loss_df['Dataset'] == dataset]
            loss_file.write(f"\\multirow{{{len(models)}}}{{*}}{{{dataset}}} ")
            for i, row in models.iterrows():
                if i != models.index[0]:
                    loss_file.write(" & ")
                loss_file.write(f"& {row['Model']} & {float(row['Train Loss']):.4f} & {float(row['Valid Loss']):.4f} & {float(row['Test Loss']):.4f} \\\\\n")
            loss_file.write("\\hline\n")
        loss_file.write("\\end{tabular}\n\\end{table}")

save_table_as_latex_multirow(acc_table, loss_table, 'accuracy_report_multirow.tex', 'loss_report_multirow.tex')
