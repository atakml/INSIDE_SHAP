
import torch
import pandas as pd
from tqdm import tqdm

from ExplanationEvaluation.models.model_selector import model_selector
import pickle

from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms
import networkx


from ExplanationEvaluation.explainers.utils import get_edge_distribution
from subgraph_metrics import *

from ExplanationEvaluation.explainers.XGNN_explainer import gnn_explain
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.explainers.MCTS_explainer import MCTSExplainer
from ExplanationEvaluation.gspan_mine.gspan_mining.gspan import GSpanMiner
from ExplanationEvaluation.explainers.VAE import GVAE
from ExplanationEvaluation.explainers import VAE
from ExplanationEvaluation.gspan_mine.gspan_mining.gspan import GSpanMiner
from ExplanationEvaluation.explainers.RandomExplainers import RandomExplainer_subgraph, get_activation_score
from networkx.algorithms.isomorphism import ISMAGS

import matplotlib.pyplot as plt
import seaborn as sns

Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])


def get_children_values(tree, fun):
    nodes = tree.as_list()
    nodes = [el for el in nodes if el.emb is not None]

    vals = list()
    for node in nodes:
        if node.emb is not None:
            v1 = fun(node)
            for el in node.children:
                if el.emb is not None:
                    vals.append(np.abs(v1-fun(el)))
    return np.mean(vals)

def get_distribution_values(tree,evaluator, metr, rules, rule):
    nodes = tree.as_list()

    nodelist =  [el for el in nodes if el.emb is not None]
    best_nodes = sorted(nodelist, key = (lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule])))[-10:]

    vals =list()

    for r in rules:
        rulenb = evaluator.rules[r]
        rr= list()
        for node in best_nodes:
            vv = evaluator.funcs[metr](node.emb, rulenb)
            rr.append(vv)
        vals.append((np.mean(rr),np.std(rr)))
    return vals

def get_distribution_realism(tree,evaluator, metr, rules, rule):
    nodes = tree.as_list()

    nodelist = [el for el in nodes if el.emb is not None]
    #top = sorted(nodelist, key=(lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule])))[-20:]
    top = sorted(nodelist, key=(lambda el: el.value/el.visit))[-100:]

    #evaluator.funcs[metr](node.emb, rulenb)
    top_vals = list(map(lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule]), top))
    top_real = list(map(lambda el:evaluator.real_score(el.graph), top))


    values = list(map(lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule]), nodelist))
    real = list(map(lambda el:evaluator.real_score(el.graph), nodelist))


    #values = sorted(values, )
    #sns.displot(list(values),kind="kde", color='blue')


    df=pd.DataFrame(list(zip(values, real, ["all" for _ in values]))+ list(zip(top_vals, top_real, ["top" for _  in top_vals ])), columns=["score", "realism","type"])
    #d2= pd.DataFrame(), columns=["score", "realism","type"])

    #xx = plt.subplots(1)
    plt.figure(figsize=(20, 20))

    #g = sns.JointGrid(data=df, x="score", y="realism",hue="type")
    #g.plot(sns.kdeplot, sns.histplot,kde=True, stat="count", bins=20)

    g = sns.jointplot(data=df, x="score", y="realism",hue="type")
    g.plot_joint(sns.kdeplot, zorder=0, levels=20)
    g.plot_marginals(sns.histplot,kde=True, clip_on=False)

    g.figure.suptitle("rule " + str(rule) +" metric "+ metr +" elements " + str(len(nodelist)))

    #sns.jointplot(data=df, x="score", y="realism",hue="type")

    #g.show()
    #plt.savefig("results/plts/rule" + str(rule) +"_metric_"+ metr +".png",format='png')
    plt.show()


def get_mcts_vals(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)

    if dataset =="ba2":
        edge_probs=None
    else :
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,30)
        else :
            edge_probs = get_edge_distribution(graphs, features)
        #degre_distrib = get_degre_distribution(graphs, features)

    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    metrics = ["cosine","likelyhood_max", "entropy"]
    #metrics = []

    real_ratio = {"cosine": (0, 1),
                  "entropy" : (-70,-2.014),
                  "lin" : (-1,0.1),
                  "likelyhood_max": (-35,20.04)
                  }
    #metrics = ["likelyhood_max"]
    eval_metrics = []#metrics[0]
    #metrics = ["likelyhood"]
    rules = range(Number_of_rules[dataset])

    rules = [list(range(0,20)),list(range(20,40)), list(range(40,60))]



    scores = list()
    nsteps = 2000
    nxp = 1
    print( dataset+ " "+ str(nsteps) + " " + str(nxp)+ " " + str(metrics))
    r=1
    vals = list()
    for rulelist in rules:
        for rule in tqdm(rulelist):

            for metric in metrics:
                for x in (range(nxp)):
                    explainer = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=rule,
                                                  target_metric=metric, eval_metrics=[metric], uid=x, edge_probs=edge_probs,
                                                  real_ratio= (real_ratio[metric],r))
                    #explainer.train(nsteps)
                    #rv = partial(rule_value, explainer.rule_evaluator, metric, rule)
                    #dd= get_distribution_values(explainer.tree.root,rv )
                    #valtree = get_children_values(explainer.tree.root, rv)
                    #vals = [get_distribution_values(explainer.tree.root,partial(rule_value, explainer.rule_evaluator, metric, rn)) for rn in tqdm(range(20*(rule//60), 20*(rule//60)+20))]
                    #with open("results/mcts_dumps/dataset_" +"valtree dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(nxp)+".pkl", 'wb') as f1:
                    #    explainer.tree.root.clean_tree()

                    #    pickle.dump(explainer.tree.root, f1)
                    dir = "mcts_dumps_old"
                    old_nodes = get_nodes("results/"+dir+"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(1)+".pkl",
                              explainer)
                    dir= "mcts_dumps_new"
                    new_nodes = get_nodes("results/" + dir + "/dataset_" + dataset + "_rule_" + str(
                        rule) + "metric_" + metric + "xp_" + str(x) + "steps_" + str(nsteps) + "nxp_" + str(1) + ".pkl",
                                          explainer)
                    get_dis(old_nodes, new_nodes, rule, metric)
                    """with open("results/"+dir+"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(1)+".pkl", 'rb') as f:
                        base = pickle.load(f)
                        
                        x=tt
                        l_node=[el for el in tt.as_list() if el.emb is not None]
                        scores=[explainer.compute_score(el.graph)[1][0] for el in l_node]
                        sns.displot(scores)
                        plt.show()
                        #vv = get_distribution_values(tt ,explainer.rule_evaluator, metric, rulelist,rule)
                        #get_distribution_realism(tt ,explainer.rule_evaluator, metric, rulelist,rule)
                        #vals.append(get_distribution_values(tt ,explainer.rule_evaluator, metric, rulelist,rule))
                    #with open("results/mcts_dumps/dataset_" + dataset +"metric_"+metric+"confusion.pkl", 'wb') as f:
                    #    pickle.dump(np.array(vals), f)"""
    print(0)

def get_dis(old,new,rule, metr):
    old = sorted(old, key=lambda x :x[0])[-100:]
    olds = [el[1][0] for el in old]
    oldr = [el[1][1] for el in old]

    new = sorted(new, key=lambda x :x[0])[-100:]

    news = [el[1][0] for el in new]
    newr = [el[1][1] for el in new]

    df=pd.DataFrame(list(zip(olds, oldr,  ["old" for _ in oldr]))+ list(zip(news,newr, ["new" for _ in newr])), columns=["score","realism","type"])
    g = sns.jointplot(data=df, x="score", y="realism",hue="type")
    g.plot_joint(sns.kdeplot, zorder=0, levels=20)
    g.plot_marginals(sns.histplot,kde=True, clip_on=False)

    g.figure.suptitle("rule " + str(rule) +" metric "+ metr )
    plt.show()

def get_nodes(filename, explainer):
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
        l_node = [el for el in tree.as_list() if (el.emb is not None or len(el.graph)>1)]
        scores = [explainer.compute_score(el.graph) for el in l_node]
        return scores
#for dataset in ["aids"]:#', "","ba2"]:

#    get_mcts_vals(dataset)



"""
En gros pour les figures 2, 3, 4 et 5 on a besoin en ligne des patterns et en colonne 1) la méthode, 2) le numéro de layer, 3) le nombre de "supporting nodes", 4) le nombre de composantes activées, 5) SI_SG, 6) coverage positive class, 7) coverage negative class.
11:37
je ne sais pas si les informations nécessaires à la figure 7 peuvent être ajoutées au fichier précédent ou pas, si oui en colonne à la suite
"""

from ExplanationEvaluation.explainers.utils import RuleEvaluator

def statistical_data(dataset):
    names = {"ba2": ("ba2"),
             "aids": ("Aids"),
             "BBBP": ("Bbbp"),
             "mutag": ("Mutag"),
             "DD": ("DD"),
             "PROTEINS_full": ("Proteins")
             }
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
    # file = "/home/ata/ENS de Lyon/Internship/Projects/MCTS/inter-compres/INSIDE-GNN/data/Aids/Aids_activation_encode_motifs.csv"
    rules = list()
    datas = load_dataset(dataset)[:2]
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    embs = get_embs(datas, model)
    with open(file, "r") as f:

        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            dd=dict()
            for el in l.split(" "):
                s = el.split(":")
                if len(s) ==2:
                    dd[s[0]] = float(s[1])


            label = int(l.split(" ")[3].split(":")[1])

            if len(rules)<47:
                rules.append((label, r, dd))
    out = list()


    for i, (_, r, dd) in enumerate(tqdm(rules)):
        c = r.split(" ")
        layer = int(c[0][2])
        components = list()
        for el in c:
            components.append(int(el[5:]))
            (layer, components, components, layer)
        acts = count_activation(embs, (layer, components))
        out.append((i, layer, acts, int(dd["nb"]) , format(dd["score"] ,'.3f'), int(dd["c+"]), int(dd["c-"]), int(dd["target"])))


    return out

def get_embs(dataset, model):
    embs = list()
    for g, f in tqdm(list(zip(*dataset))):
        g = torch.tensor(g)
        f = torch.tensor(f)
        #adj = dense_to_sparse(g)[0]
        max = int(f.sum())
        embs.append([el[:max] for el in model.embeddings(f, g)])
    return embs

def count_activation(embs, rule):
    count=0
    for e in embs:
        for node in e[rule[0]]:
            count += RuleEvaluator.activate_static(rule, node)
    return count



""" des patterns et en colonne 1) la méthode, 2) le numéro de layer, 3) le nombre de "supporting nodes", 4) le nombre de composantes activées, 5) SI_SG, 6) coverage positive class, 7) coverage negative class.
"""
print("looo")
import csv

for dataset in ["DD"]:
    data = statistical_data(dataset)

    with open("results/stat_data_"+dataset+".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        header = ['pattern number', 'layer', 'supporting nodes', "activated components","SI_SG", "pos graphs", "neg graphs", "target class "]
        datawriter.writerow(header)
        for row in data :
            datawriter.writerow(row)
