import ast
import os
import shutil

from utiles import read_from_beam_search_files, read_from_mcts_files, read_rules_from_file
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift


def plot_interestingness(dataset_name):
    exhuast_file = f"{dataset_name.cpitalize()}_activatio.out"
    for layer in range(3):
        mcts_file_name = f"{dataset_name}_l{layer}_single.txt"
        beam_search_file_name = f"{dataset_name}_l{layer}_bean.csv"


def modify_exhds(dataset_name):
    '''file_name = f"{dataset_name}_activation.out"
    lines = []
    i = 0
    with open(file_name, "r") as f:
        for i,line in enumerate(f.readlines()):
            lines.append(line.replace("#", f"{i},# {i}").replace("= ", "="))
    #os.mkdir(f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}")
    os.remove(f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}/{dataset_name}_activation_encode_motifs.csv")
    with open(f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}/{dataset_name}_activation_encode_motifs.csv", "w") as f:
        for line in lines:
            f.write(line)'''
    shutil.move(f"mcts.py_modifed/{dataset_name}_activation.csv",
                f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}/{dataset_name}_activation_encode_motifs.csv")


def convert_rule_from_dict(rule, layer):
    components = ast.literal_eval(rule).keys()
    components = list(map(lambda x: f"l_{layer}c_{x}", components))
    rule = " ".join(components)
    return rule


def convert_from_beam_to_exh(dataset_name):
    rules = []
    i = 0
    for layer in range(3):
        for suffix in ["", "_neg"]:
            file_name = f"{dataset_name}_l{layer}_beam{suffix}.csv"
            rule_for_layer = read_from_beam_search_files(file_name)
            target = 0 if suffix == "_neg" else 1
            scores = rule_for_layer["score"].to_numpy()
            rule_for_layer = list(map(lambda x: convert_rule_from_dict(x, layer), rule_for_layer["pattern"]))
            rule_for_layer = list(map(lambda
                                          x: f"{x[0]},# {x[0]} {x[1][0]} target:{target} c+:0 c-:0 score:{scores[x[1][0]]} score+:0 score-:0 nb:{x[1][1].count('l')} ={x[1][1]}\n",
                                      list(enumerate(enumerate(rule_for_layer), i))))
            i += len(rule_for_layer)
            rules.extend(rule_for_layer)
    try:
        os.mkdir("beam_modifed")
    except FileExistsError:
        pass
    with open(f"beam_modifed/{dataset_name}_activation.csv", "w") as f:
        for line in rules:
            f.write(line)


def convert_from_mcts_to_exh(dataset_name):
    rules = []
    i = 0
    for layer in range(3):
        for suffix in ["", "_neg"]:
            file_name = f"{dataset_name}_l{layer}_single{suffix}.txt"
            rule_for_layer = read_from_mcts_files(file_name)
            target = 0 if suffix == "_neg" else 1
            scores = list(map(lambda x: x["score"], rule_for_layer))
            rule_for_layer = list(map(lambda x: convert_rule_from_dict(x["pattern"], layer), rule_for_layer))
            rule_for_layer = list(map(lambda
                                          x: f"{x[0]},# {x[0]} {x[1][0]} target:{target} c+:0 c-:0 score:{scores[x[1][0]]} score+:0 score-:0 nb:{x[1][1].count('l')} ={x[1][1]}\n",
                                      list(enumerate(enumerate(rule_for_layer), i))))
            i += len(rule_for_layer)
            rules.extend(rule_for_layer)
    try:
        os.mkdir("mcts.py_modifed")
    except FileExistsError:
        pass
    with open(f"mcts.py_modifed/{dataset_name}_activation.csv", "w") as f:
        for line in rules:
            f.write(line)


def read_datas_from_file(policy, methods):
    res = []
    for method in methods:
        file_name = f"/home/mike/{policy}_p/fidinfidspars_{method}"
        if not os.path.exists(file_name):
            file_name = f"/home/mike/{policy}_p/findinfidspars_{method}"
        current_dataset = None
        with open(file_name, "r") as f:
            dataset_dicts = dict()
            for line in f.readlines():
                if not len(line.split()):
                    continue
                if line.split()[0] == "Loading":
                    current_dataset = line.split()[1]
                elif line[:2] == "[{":
                    dataset_dicts[current_dataset] = ast.literal_eval(line.strip("\n")[1:-1])
        res.append(dataset_dicts)
    return res


def build_layer_dict(file):
    layer_dict = dict()
    with open(file, "r") as f:
        for line in f.readlines():
            target, score, pattern = int(line.split()[3].split(":")[1]), float(line.split()[6].split(":")[1]), line[
                                                                                                               line.index(
                                                                                                                   "=") + 1:].split()
            layer = int(pattern[0][pattern[0].index("l") + 2: pattern[0].index("c")])
            if (layer, target) not in layer_dict.keys():
                layer_dict[(layer, target)] = [{"score": score, "pattern": pattern}]
            else:
                layer_dict[(layer, target)].append({"score": score, "pattern": pattern})

    return layer_dict


def clustering(layer_dict):
    for (layer, target), rules in layer_dict.items():
        scores = np.array(list(map(lambda rule: rule["score"], rules))).reshape(-1, 1)
        labels = MeanShift().fit_predict(scores)
        number_of_clusters = len(set(labels))
        if number_of_clusters == 1:
            continue
        cluster_to_delete = labels[np.argmin(scores)]
        cnt = 0
        for i, score in enumerate(scores):
            if labels[i] == cluster_to_delete:
                rules.pop(i - cnt)
                cnt += 1
        layer_dict[(layer, target)] = rules
    return layer_dict

def filter_si_less_ten(layer_dict):
    for (layer, target) in layer_dict.keys():
        rules = []
        for rule in layer_dict[(layer, target)]:
            rules.append(rule)
            if rule["score"] < 10:
                break
        layer_dict[(layer, target)] = rules
    return layer_dict

def rewrite_to_the_file(layer_dict, file):
    with open(file, "w") as f:
        cnt = 0
        for layer, target in sorted(layer_dict.keys()):
            for i, rule in enumerate(layer_dict[(layer, target)]):
                string_to_write = f"{cnt},# {cnt} {i} target:{target} c+:0 c-:0 score:{rule['score']} score+:0 score-:0 nb:{len(rule['pattern'])} ={' '.join(rule['pattern'])}\n"
                f.write(string_to_write)
                cnt += 1


def preprocess(dataset_name, method):
    method = "_beam" if method == "beam" else ""
    file_to_read = f"/home/mike/activations/{dataset}/{dataset}_activation_encode_motifs{method}.csv"
    file_to_write = f"/home/mike/internship_project/inter-compres/GNN-explain/codebase/ExplanationEvaluation/datasets/activations/{dataset_name}/{dataset_name}_activation_encode_motifs{method}.csv"
    layer_dict = build_layer_dict(file_to_read)
    #layer_dict = clustering(layer_dict)
    layer_dict = filter_si_less_ten(layer_dict)
    rewrite_to_the_file(layer_dict, file_to_write)


datasets = ["ba2", "Bbbp", "Aids"]
methods = ["beam", "mcts"]
'''for dataset in datasets:
    for method in methods:
        preprocess(dataset, method)
'''
#fidinfid charts
policies = ["node", "ego", "decay", "top5"]
methods = ["ex", "mcts", "beam"]
metrics = ["fid", "infid", "sparsity"]
dict_name = {"Bbbp": "BBBP", "Aids": "aids", "ba2": "ba2"}
for policy in policies:
    chart_data = read_datas_from_file(policy, methods)
    for metric in metrics:
        bar_width = 0.25
        index = np.arange(len(datasets))
        fig, ax = plt.subplots()
        method_dict = dict()
        for i in range(len(methods)):
            x = chart_data[i]
            data_to_draw = [
                x[dict_name[ds]][metric][0] if isinstance(x[dict_name[ds]][metric], tuple) else x[dict_name[ds]][metric]
                for ds in datasets]
            ax.bar(index + (i * bar_width), data_to_draw, bar_width, label=methods[i])
        title_suffix = "_prob" if metric != "sparsity" else ""
        ax.set_title(f"{policy} {metric}{title_suffix}")
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(datasets)
        ax.legend(methods)
        #plt.show()
        plt.savefig(f"{policy} {metric}{title_suffix}10")
        

'''
#csi
for dataset in datasets:
    method_layer_dict = dict()
    for method in methods:
        scores = []
        suffix = f"_{method}" if method != "mcts" else ""
        file_name = f"/home/mike/activations/{dataset}/{dataset}_activation_encode_motifs{suffix}.csv"
        layer_dict = build_layer_dict(file_name)
        if method != "ex":
            for (layer, target) in layer_dict.keys():
                rules = []
                for rule in layer_dict[(layer, target)]:
                    rules.append(rule)
                    if rule["score"] < 10:
                        break
                layer_dict[(layer, target)] = rules
        method_layer_dict[method] = layer_dict
    for (layer, target) in method_layer_dict["ex"].keys():
        plt.clf()
        acm = dict()
        for method in methods:
            acm[method] = np.cumsum(list(map(lambda x:x["score"], method_layer_dict[method][(layer, target)])))
            plt.plot(np.arange(len(acm[method])), acm[method], label=method)
        plt.legend()
        plt.title(f"layer: {layer} target: {target} dataset: {dataset}")
        plt.savefig(f"layer: {layer} target: {target} dataset: {dataset}")
        '''