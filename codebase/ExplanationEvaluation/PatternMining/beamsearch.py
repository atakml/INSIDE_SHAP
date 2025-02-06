import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time 
from utiles import read_log_probs_from_file, compute_SI_double, lecture_fichier, build_model
import pysubgroup as ps

positive_data, positive_probs, negative_data, negative_probs = None, None, None, None


class MyQualityFunction:
    def calculate_constant_statistics(self, task):
        """ calculate_constant_statistics
            This function is called once for every execution,
            it should do any preparation that is necessary prior to an execution.
        """
        pass

    def calculate_statistics(self, subgroup, data=None):
        """ calculates necessary statistics
            this statistics object is passed on to the evaluate
            and optimistic_estimate functions
        """
        pass

    @staticmethod
    def evaluate(subgroup, target, data, statistics_or_data=None):
        """ return the quality calculated from the statistics """
        pattern = {data.columns.get_loc(key.attribute_name): key.attribute_value for key in subgroup.selectors}
        if len(subgroup.selectors) != len(pattern.keys()):
            return -math.inf
        # res = w1 * compute_SI(positive_probs, positive_data, pattern, positive_graph_inds) - (
        # w0 * compute_SI(negative_probs, negative_data, pattern, negative_graph_inds))
        # res = w1 * compute_SI(positive_probs, positive_data, pattern, positive_graph_inds) - (
        # w0 * compute_SI(negative_probs, negative_data, pattern, negative_graph_inds))
        # print(pattern, res + 10 ** 5)
        res = w1 * compute_SI_double(positive_probs, positive_data, pattern, positive_graph_inds) - (
                w0 * compute_SI_double(negative_probs, negative_data, pattern, negative_graph_inds))
        return -res + 10 ** 5

    def optimistic_estimate(self, subgroup, statistics=None):
        """ returns optimistic estimate
            if one is available return it otherwise infinity"""
        return math.inf
        # pattern = {data.columns.get_loc(key.attribute_name): key.attribute_value for key in subgroup.selectors}
        # res = SI_UB(model.proba, data, pattern, positive_graph_inds, negative_graph_inds, w1, w0)
        # return res + 10 ** 5


dataset_name = "Mutag"
for idLayer in range(3):
    _, ds, data, Tclasse, Tmolecule, Tatome, Tnames, pcol, pligne, nb_mol, nb_mol_plus = build_model(
        f"/home/ata/inside/GNN-explain/codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation.csv",
        idLayer)
    model = read_log_probs_from_file(f"{dataset_name}_activation_lay_{idLayer}_proba.txt")
    Tnames = sorted(Tnames, key=lambda x: (int(x[x.index('_') + 1: x.index('c')]), int(x[x.index('c') + 2:])))
    layer_columns = list(filter(lambda col: f"l_{idLayer}c" in col, Tnames))
    dataset = pd.DataFrame(ds.data, columns=layer_columns)
    dataset = dataset.astype(bool)
    dataset["label"] = Tatome
    n_samples0 = nb_mol - nb_mol_plus
    n_samples1 = nb_mol_plus
    w0 = max(1, n_samples1 / n_samples0)
    w1 = max(1, n_samples0 / n_samples1)
    graph_inds = Tmolecule.to_numpy()
    labels = Tclasse
    positive_data = ds.data[labels]
    positive_graph_inds = graph_inds[labels]
    positive_probs = model.proba[labels]
    negative_data = ds.data[~labels]
    negative_graph_inds = graph_inds[~labels]
    negative_probs = model.proba[~labels]
    target = ps.BinaryTarget('label', False)
    for i in tqdm(range(10)):
        start_time = time()
        searchspace = ps.create_nominal_selectors(dataset, ignore="label")
        c = []
        for sel in searchspace:
            if not sel.attribute_value:
                c.append(sel)
        for sel in c:
            searchspace.remove(sel)
        task = ps.SubgroupDiscoveryTask(
            dataset,
            target,
            searchspace,
            result_set_size=1,
            depth=9,
            qf=MyQualityFunction)
        result = ps.BeamSearch().execute(task)
        columns_to_delete = [key.attribute_name for key in result.results[0][1].selectors]
        result = result.to_dataframe()
        result['quality'] -= 10 ** 5
        result['time'] = time() - start_time
        result.head(1).to_csv(f"{dataset_name}_l{idLayer}_beam_double_neg.csv", mode="a")
        if result['quality'][0] < 10:
            break
        indices_of_columns_to_delete = [dataset.columns.get_loc(column) for column in columns_to_delete]
        model.proba[:, indices_of_columns_to_delete] = 0
        positive_probs[:, indices_of_columns_to_delete] = 0
        negative_probs[:, indices_of_columns_to_delete] = 0
