import os 
import pickle 
import numpy as np
from method.diffnaps import test_bin, test
from method.my_loss import weightedXorCover
import math
import torch
from tqdm import tqdm
from utils_base import TrainConfig, compile_new_pat_by_class, get_positional_patterns
from method.diffnaps import *
from patternmod.diffversify_utils import load_diffversify_patterns, convert_patterns_to_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()
    dataset = args.dataset_name

    methods = ["diffversify"]
    lambs = [0.0, 1., 5., 10., 15., 25., 100.]
    lambcs = [0.0, 1., 5., 10., 15., 25., 50.0, 100.]
    lrs= [0.01, 0.005]
    alpha = None
    
    for method in methods:
        best_loss, best_lamb, best_lr, best_lambc = math.inf, None, None, None
        for lr in lrs:
            for lamb in lambs:
                for lambc in lambcs: 
                    print(f"{method=} {dataset=} {lr=} {lamb=} {lambc=}")
                    config = TrainConfig(hidden_dim =-1 , epochs=200, weight_decay=lamb, elb_k=lamb, elb_lamb=lamb, class_elb_k=lamb, class_elb_lamb=lamb,
                                    lambda_c = lambc, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                                log_interval=1000, sparse_regu=0, test=False, lr=lr, model=DiffnapsNet,seed=14401360119984179300,
                                save_xp=True, loss=weightedXorCover)
                    command = f"python patternmod/pattern_mining.py {dataset} --method {method} --lr {lr} --lamb {lamb} --lambc {lambc}"
                    exit_code = os.system(command)
                    assert exit_code == 0
                    with open(f"patternmod/res/{method}_{dataset}_{lr}_{lamb}_{lambc}.pkl", "rb") as file:
                        model, new_weights, trainDS = pickle.load(file)
                    if alpha is None:
                        alpha = trainDS.labels.sum().cpu().item()/len(trainDS.labels)
                    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=64, shuffle=True)
                    weight = trainDS.getSparsity()
                    lossFun = weightedXorCover(weight, lamb, "cuda:0", label_decay = 0, labels=2, alpha=alpha)
                    lossFun.config = config
                    try:
                        loss = test(model, 0, "cpu", "cuda:0", train_loader, lossFun)
                    except:
                        print(trainDS.ncol())
                        data, target = next(iter(train_loader))
                        print(data.shape)
                        print(target)
                        raise
                    if loss < best_loss:
                        best_loss, best_lamb, best_lr, best_lambc = loss, lamb, lr, lambc

        with open(f"patternmod/res/train_hyperparam_{method}_{dataset}.pkl", "wb") as file:
            pickle.dump((best_lamb, best_lr, best_lambc, best_loss), file)#second """
        with open(f"patternmod/res/train_hyperparam_{method}_{dataset}.pkl", "rb") as file:
            best_lamb, best_lr, best_lambc, best_loss = pickle.load(file)
        with open(f"patternmod/res/{method}_{dataset}_{best_lr}_{best_lamb}_{best_lambc}.pkl", "rb") as file:
        #with open(f"patternmod/res/{method}_{dataset}_{lr}_{lamb}.pkl", "rb") as file:
            model, new_weights, trainDS = pickle.load(file)
        config = TrainConfig(hidden_dim =-1 , epochs=200, weight_decay=best_lamb, elb_k=best_lamb, elb_lamb=best_lamb, class_elb_k=best_lamb, class_elb_lamb=best_lamb,
                                lambda_c = best_lambc, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                            log_interval=1000, sparse_regu=0, test=False, lr=best_lr, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True, loss=weightedXorCover)
        alpha = trainDS.labels.sum().cpu().item()/len(trainDS.labels)
        weight = trainDS.getSparsity()
        lossFun = weightedXorCover(weight, best_lamb, "cuda:0", label_decay = 0, labels=2, alpha=alpha)
        lossFun.config = config
        train_loader = torch.utils.data.DataLoader(trainDS, batch_size=64, shuffle=True)
        print(test(model, 0, "cpu", "cuda:0", train_loader, lossFun, verbose=True, writer=None))

        min_value = new_weights.min().cpu().numpy()
        max_value = new_weights.max().cpu().numpy()
        num_points = 20
        x = np.linspace(min_value, max_value, num_points)
        y = np.linspace(min_value, max_value, num_points)
        X, Y = np.meshgrid(x, y)
        best_t1, best_t2, best_loss = None, None, math.inf
        t1_dict = {}
        for i in tqdm(range(num_points)):
            best_recon_loss = math.inf
            for j in range(num_points):
                t1 = X[i, j]
                t2 = Y[i, j]
                recon_loss = test_bin(model, "cpu", "cuda:0", train_loader, weight, t1, t2, 0)
                #print(f"{t1=}, {recon_loss=}")
                if recon_loss < best_recon_loss:
                    best_recon_loss, t1_dict[t2] = recon_loss, t1
            class_loss = test_bin(model, "cpu", "cuda:0", train_loader, weight, t1_dict[t2], t2, 1)
            #print(f"{t2=}, {class_loss=}")
            if class_loss < best_loss:
                best_t1, best_t2, best_loss = t1_dict[t2], t2, class_loss
        with open(f"patternmod/res/pattern_hyperparam_{method}_{dataset}.pkl", "wb") as file:
            pickle.dump((best_t1, best_t2, best_loss), file)
        command = f"python patternmod/pattern_mining.py {dataset} --method {method} --t1 {best_t1} --t2 {best_t2} --lr {best_lr} --lamb {best_lamb} --lambc {best_lambc} --pmining"
        #command = f"python patternmod/pattern_mining.py {dataset} --method {method} --t1 {best_t1} --t2 {best_t2} --lr {lr} --lamb {lamb} --pmining"
        exit_code = os.system(command)
        assert exit_code == 0
        patterns = load_diffversify_patterns(dataset, method="diffversify", t1=best_t1, t2=best_t2, lamb=best_lamb, lr=best_lr)
        print(len(convert_patterns_to_dict(patterns)))
        print(best_t1, best_t2, best_loss, min_value, max_value)

