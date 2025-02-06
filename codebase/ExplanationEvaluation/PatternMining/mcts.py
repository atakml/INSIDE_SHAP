import copy
import json
import math
import os
import random

import scipy as sp
from scipy import stats
from math import inf
import numpy as np
import pandas as pd
from tqdm import tqdm

total_calls = 0
zero_calls = 0

# from activation_pattern_with_pysub import compute_SI as csi
from utiles import compute_SI_single, build_model, read_log_probs_from_file, compute_SI_double


# from activation_pattern_with_pysub import SI_UB


class NodePattern:
    def __init__(self, si_function, value=0, parent=None, fixed_value_dict={}, free_indices=None, c=100, si_ub=0,
                 ub_function=None):
        self.value = value
        self.parent = parent
        self.fixed_value_dict = fixed_value_dict
        self.free_indices = free_indices
        self.visits = 0
        self.c = c
        self.children = []
        self.si_function = si_function
        self.best_simulation = None
        self.si_ub = si_ub
        self.ub_function = ub_function
        self.si = None
        self.flag = True

    def is_terminal(self):
        return len(self.free_indices) == 0

    def compute_UCB1(self):
        if not self.visits:
            return inf
        # if not self.flag:
        #    return -inf
        return self.value / self.visits + self.c * (np.log2(self.parent.visits) / self.visits) ** 0.5

    def compute_UCB2(self):
        if not self.visits:
            return inf
        # if not self.flag:
        #    return -inf
        if self.si is None:
            self.si = self.si_function(self.fixed_value_dict)
        return self.si + self.value / self.visits + self.c * (
                np.log2(self.parent.visits) * (len(self.free_indices)) / self.visits) ** 0.5

    def propagate_value(self, value, simulated_node):
        self.value += value
        # self.value = max(value, self.value)
        if not self.best_simulation or value > self.best_simulation[1]:
            self.best_simulation = (simulated_node, value)
        if self.parent:
            self.parent.propagate_value(value, simulated_node)

    def simulate(self, pcol=None):
        # idea: using a probability rather than uniform
        # flags = np.random.binomial(n=2, p=[0.5] * len(self.free_indices))  # uniform
        flags = np.random.binomial(n=2, p=1 - pcol[self.free_indices])  # marginal
        tmp_dict = dict(zip(self.free_indices, flags))
        # begin of random selection of indices
        '''size = random.randint(0, min(len(self.free_indices), 3))
        indices = np.random.choice(self.free_indices, size=size)
        tmp_dict = {index: 0 for index in self.free_indices if index not in indices}
        for index in indices:
            tmp_dict[index] = np.random.binomial(n=2, p=[0.5])'''
        # end of random selection of indices

        new_fixed_value_dict = {**self.fixed_value_dict, **tmp_dict}
        value = self.si_function(new_fixed_value_dict)
        self.propagate_value(value, NodePattern(self.si_function, value, fixed_value_dict=new_fixed_value_dict))

    def propagate_value2(self, value):
        self.value -= value
        if self.parent:
            self.parent.propagate_value2(-value / 2)

    def select_child2(self, depth=0, besti_si=0):
        number_of_children = len(self.children)
        total_number_of_children = 2 * len(self.free_indices)
        #total_number_of_children = len(self.free_indices)
        if not total_number_of_children:
            self.propagate_value2(-self.value)
            return self
        if total_number_of_children == number_of_children:
            try:
                best_child = max(self.children, key=lambda node: node.compute_UCB2())
            except:
                pass
            return best_child
        component_index = number_of_children // 2
        component_stat = number_of_children % 2
        #component_index = number_of_children
        #component_stat = 1
        new_dict = copy.copy(self.fixed_value_dict)
        new_dict[self.free_indices[component_index]] = component_stat
        for i in range(component_index):
            new_dict[self.free_indices[i]] = 2
        new_free_indices = np.setdiff1d(self.free_indices, list(new_dict.keys()))
        new_child = NodePattern(self.si_function, free_indices=new_free_indices, fixed_value_dict=new_dict,
                                parent=self, ub_function=self.ub_function)
        self.children.append(new_child)
        return new_child

    def select_child(self, depth, best_si=None):
        number_of_children = len(self.children)
        total_number_of_children = 3
        if total_number_of_children == number_of_children:
            # print(self.children[0].compute_UCB1(), self.children[1].compute_UCB1())
            best_child = max(self.children, key=lambda node: node.compute_UCB1())
            if depth > 10 and best_child.si_ub < best_si:
                self.flag = False
            return best_child
        else:
            key = min(self.free_indices)
            value = number_of_children % 3
            new_dict = copy.copy(self.fixed_value_dict)
            new_dict[key] = value
            ub = 0
            if depth > 7:
                ub = self.ub_function(new_dict)
            new_free_indices = np.delete(self.free_indices, np.where(self.free_indices == key))
            new_child = NodePattern(self.si_function, free_indices=new_free_indices, fixed_value_dict=new_dict,
                                    parent=self, si_ub=ub, ub_function=self.ub_function)
            self.children.append(new_child)
            return new_child


def evaluate(root, value=0):
    best_result, best_pattern = root.best_simulation[1], root.best_simulation[0].fixed_value_dict
    for child in root.children:
        # SI_SG = value if set(key for key, value in child.fixed_value_dict.items() if value < 2) == set(
        # key for key, value in root.fixed_value_dict.items() if value < 2) else child.si if child.si is not None else child.si_function(
        # child.fixed_value_dict)
        SI_SG = child.si if child.si is not None else value if set(
            key for key, value in child.fixed_value_dict.items() if value < 2) == set(
            key for key, value in root.fixed_value_dict.items() if value < 2) else child.si_function(
            child.fixed_value_dict)
        # SI_SG = child.si_function(child.fixed_value_dict)
        if SI_SG > best_result:
            best_result, best_pattern = SI_SG, {key: value for key, value in child.fixed_value_dict.items() if
                                                value < 2}
        sub_best_result, sub_best_pattern = evaluate(child, SI_SG)
        if sub_best_result > best_result:
            best_result, best_pattern = sub_best_result, sub_best_pattern
    return best_result, best_pattern


class MCTSPatternMiner:

    def __init__(self, model, ds, graph_inds, labels, w0, w1, exceptions):
        # self.model = model
        self.ds = ds
        self.w0, self.w1 = w0, w1
        self.labels = labels
        # self.graph_inds = graph_inds
        self.positive_data = ds.data[labels]
        self.positive_graph_inds = graph_inds[labels]
        self.positive_probs = model.proba[labels]
        self.negative_data = ds.data[~labels]
        self.negative_graph_inds = graph_inds[~labels]
        self.negative_probs = model.proba[~labels]
        self.root = NodePattern(
            si_function=lambda pattern: -(self.w1 * compute_SI_double(self.positive_probs, self.positive_data, pattern,
                                                                    self.positive_graph_inds) - (
                                                self.w0 * compute_SI_double(self.negative_probs, self.negative_data,
                                                                            pattern,
                                                                            self.negative_graph_inds))),
            free_indices=np.arange(ds.data.shape[1]),
            ub_function=lambda pattern: SI_UB(self.positive_probs, self.positive_data, pattern,
                                              self.positive_graph_inds, w1=self.w1))
        # self.mapping = sorted(range(20), key=lambda x: -self.root.si_function({x: 0}))
        # self.mapping = sorted(range(14), key=lambda x: ds.pcol[x])
        '''self.positive_data = self.positive_data[:, self.mapping]
        self.positive_probs = self.positive_probs[:, self.mapping]
        self.negative_data = self.negative_data[:, self.mapping]
        self.negative_probs = self.negative_probs[:, self.mapping]'''

    def iterate(self, budget=lambda: total_calls == 0 or zero_calls / total_calls < 0.9):
        current_node = self.root
        rule_enum = {}
        # print("iteration:")
        simul_cnt = 0
        best_si = 0
        while simul_cnt < 100000:
            current_node = self.root
            simul_cnt += 1
            print("simulation:", simul_cnt)
            depth_cnt = 0
            while current_node.children:
                current_node = current_node.select_child2(depth_cnt, best_si)
                depth_cnt += 1
                '''if depth_cnt > 7:
                    if current_node.si == 0:
                        current_node.si = current_node.si_function(current_node.fixed_value_dict)
                    best_si = max(best_si, current_node.si)'''
                current_node.visits += 1
            if current_node.is_terminal():
                SI_SG = current_node.si_function(current_node.fixed_value_dict)
                current_node.propagate_value(SI_SG, current_node)
                current_node.visits += 1
                '''if current_node.si == 0:
                    current_node.si = current_node.si_function(current_node.fixed_value_dict)
                best_si = max(best_si, current_node.si)'''
                # break
            if current_node.best_simulation is not None:
                current_node = current_node.select_child2(depth_cnt,
                                                          best_si)  # This function acts both as expander and selector
                current_node.visits += 1
                '''if current_node.si == 0:
                    current_node.si = current_node.si_function(current_node.fixed_value_dict)'''
                #                best_si = max(best_si, current_node.si)
                depth_cnt += 1
                print("depth:", depth_cnt)
            current_node.simulate(self.ds.pcol)
            if current_node.visits > 200:
                break
            print("depth:", depth_cnt)
            print("pattern:", current_node.fixed_value_dict)
            print("%%%%%%%%%%%%%%%%%")
            print("best:", self.root.best_simulation[1])
            # print(f"empty ratio: {zero_calls / total_calls:2f}")
            print("%%%%%%%%%%%%%%%%%")
        print("Evaluating:")
        SI_SG = current_node.si_function(current_node.fixed_value_dict)
        current_node.propagate_value(SI_SG, current_node)
        current_node.visits += 1
        return evaluate(self.root), simul_cnt, 0


if __name__ == "__main__":
    dataset_name = "Mutag"
    for idLayer in [0]:
        _, ds, data, Tclasse, Tmolecule, Tatome, Tnames, pcol, pligne, nb_mol, nb_mol_plus = build_model(
            f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv", idLayer)
        model = read_log_probs_from_file(f"{dataset_name}_activation_lay_{idLayer}_proba.txt")
        n_samples0 = nb_mol - nb_mol_plus
        n_samples1 = nb_mol_plus
        exceptions = []
        with open(f"{dataset_name}_l{idLayer}_double_neg.txt", "a") as file:
            for i in tqdm(range(10)):
                pattern = []
                while not len(pattern):
                    pattern_miner = MCTSPatternMiner(model, ds, Tmolecule.to_numpy(), Tclasse,
                                                     w0=max(1, n_samples1 / n_samples0),
                                                     w1=max(1, n_samples0 / n_samples1), exceptions=exceptions)
                    pattern, sim_cnt, f_rate = pattern_miner.iterate()
                file.write(
                    f"Score: {pattern[0]} Pattern: {pattern[1]}  Number of Simulations: {sim_cnt}, Failure Rate{f_rate}\n")
                file.write("_" * 100 + "\n")
                # print(pattern, score)
                model.proba[:, list(filter(lambda x: pattern[1][x] < 2, pattern[1].keys()))] = 0
                exceptions += list(filter(lambda x: pattern[1][x] < 2, pattern[1].keys()))
                if set(exceptions) == set(np.arange(ds.data.shape[1])):
                    break
                if pattern[0] <= 0:
                    break
