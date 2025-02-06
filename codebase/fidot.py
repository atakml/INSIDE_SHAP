import functools
import json
import operator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from tqdm import tqdm
import numpy as np

from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.explainers.MCTS_explainer import MCTSExplainer
from ego_mask import graph_dset, get_embs, get_fidelity
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch import tensor


def load_masks(dataset_name, method):
    file_name = f"median_{dataset_name}_{method}.json"
    with open(file_name) as file:
        data = json.load(file)
        data = {key: list(map(lambda x: x[1:], value)) for key, value in data.items()}
        data = functools.reduce(operator.iconcat, [data[key] for key in sorted(data.keys())])
        data = list(map(lambda x:[np.array(x[0]), np.array(x[1])], data))
        return data


Number_of_rules = dict([("aids", 35), ('mutag', 60), ("BBBP", 36), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 25)])

dataset = "ba2"
method = "mcts"
egos = load_masks(dataset, method)
metric = "cosine"
unlabeld = True if dataset != "ba2" else False
graphs, features, labels, _, _, _ = load_dataset(dataset)
model, checkpoint = model_selector("GNN",
                                   dataset,
                                   pretrained=True,
                                   return_checkpoint=True)
atoms = get_atoms(dataset, features)
egos = graph_dset(zip(*egos), atoms)
egos = {i: graph for i, graph in enumerate(egos)}
gdset = graph_dset([graphs, features], atoms)
f = "results/dataframes/" + dataset + ".csv"

x12 = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=0,
                    target_metric=None, uid=0, edge_probs=None,
                    real_ratio=(0.5, 0.5))
print("precompute fidelity")
#fid_precompute = [[(get_embs(x12, g, lay).detach(), x12.rule_evaluator.get_output(g).detach().softmax(dim=1)[0, 1])
#                   for g in tqdm(gdset)] for lay in range(3)]
embs = [[get_embs(x12, g, lay).detach()  for lay in range(3)
         ]for g in tqdm(gdset)]
preds = [x12.rule_evaluator.get_output(g).detach().softmax(dim=1)[0, 1] for g in tqdm(gdset)]
print("precompute fidelity finished")

outdf = pd.DataFrame(
    columns=["dataset", "metric", "graphid", "rule", "pred", "fidelity", "infidelity", "mags", "spars"])

masks = dict()
for rule in range(Number_of_rules[dataset]):
    evaluator = RuleEvaluator(model, dataset, (graphs, features, labels), rule, unlabeled=unlabeld)
    masks[rule] = evaluator.get_output(egos[rule]).detach().softmax(dim=1)[0, 1]
for i, (adj, f, g) in tqdm(
        list(enumerate(zip(graphs, features, gdset)))):
    emb, pred = embs[i], preds[i]
    df = get_fidelity(evaluator, metric, dataset, i, emb, pred, adj, f, g, egos, masks)
    outdf = outdf.append(df)
    # xx = outdf.groupby("graphid").apply(get_best_fid_infid)
    outdf.to_csv("results/dataframes/egofidsinfids " + "_" + dataset + "_" + method + ".csv")
