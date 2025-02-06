from utils_base import TrainConfig, compile_new_pat_by_class, get_positional_patterns
from modelmod.gnn_utils import fill_embeddings
import torch
import pickle 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import method.dataLoader as mydl
import method.my_layers as myla
import method.my_loss as mylo
from method.diffnaps import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime, time
from method.my_loss import weightedXorCover, weightedXor
from datasetmod.data_utils import select_mask
from datasetmod.datasetloader import load_dataset_gnn, load_splited_data
from modelmod.gnn import load_gnn
from modelmod.gnn_utils import concat_embeddings, fill_embeddings
import math
from copy import deepcopy 
from measures import mean_compute_metric
from experiment_utils import *
import pandas as pd

device = "cuda:0"

def learn_diffnaps_net(dataset_name, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels, config, ret_test=False, verbose=True, writer=None):
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.thread_num)
    device_cpu = torch.device("cpu")

    if not torch.cuda.is_available():
        device_gpu = device_cpu
        print("WARNING: Running purely on CPU. Slow.")
    else:
        device_gpu = torch.device("cuda")
    alpha = config.alpha #train_labels.sum().cpu().item()/len(train_labels)
    trainDS = mydl.DiffnapsDatDataset("file", 0, False, device_cpu, data=train_embeddings.detach().clone().cpu(), labels = train_labels.detach().clone().cpu())

    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=config.batch_size, shuffle=True)

    valDS = mydl.DiffnapsDatDataset("file", 0, False, device_cpu, data=val_embeddings.detach().clone().cpu(), labels = val_labels.detach().clone().cpu())
    val_loader = torch.utils.data.DataLoader(valDS, batch_size=config.batch_size, shuffle=True)
    testDS = mydl.DiffnapsDatDataset("file", 0, False, device_cpu, data=test_embeddings.detach().clone().cpu(), labels = test_labels.detach().clone().cpu())
    test_loader = torch.utils.data.DataLoader(testDS, batch_size=config.batch_size, shuffle=True)

    
    if config.hidden_dim == -1:
        config.hidden_dim = trainDS.ncol()
        
    new_weights = torch.zeros(config.hidden_dim, trainDS.ncol(), device=device_gpu)
    initWeights(new_weights, trainDS.data)
    new_weights.clamp_(1/(trainDS.ncol()), 1)
    bInit = torch.zeros(config.hidden_dim, device=device_gpu)
    init.constant_(bInit, -1)
    model = config.model(new_weights, int(np.max(train_labels.detach().clone().cpu().numpy())+1), bInit, trainDS.getSparsity(), device_cpu, device_gpu, config=config).to(device_gpu)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    cnt = 0 
    # original line from diffnaps: change to have loss as a parameter
    # lossFun = mylo.weightedXocr(trainDS.getSparsity(), config.weight_decay, device_gpu, label_decay = 0, labels=2)
    lossFun = config.loss(trainDS.getSparsity(), config.weight_decay, device_gpu, label_decay = 0, labels=2, alpha= alpha)
    scheduler = MultiStepLR(optimizer, [5,7], gamma=config.gamma)
    best_loss = math.inf
    print_gpu()
    for epoch in range(1, config.epochs + 1):
        #print(model.fc0_enc.weight.data.mean())
        model.learn(device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, config.log_interval, config, verbose=verbose, writer=writer)
        test_loss = test(model, epoch,  device_cpu, device_gpu, test_loader, lossFun,verbose=verbose, writer=writer)
        test(model, epoch,  device_cpu, device_gpu, test_loader, lossFun,verbose=verbose, writer=writer)

        print(f"{test_loss=}")
        if test_loss < best_loss:
            cnt = 0 
            best_loss = test_loss 
            with open(f"patternmod/tmpmodel_{dataset_name}.pkl", "wb") as file:
                pickle.dump(model, file)          
        else:
            cnt += 1
        if cnt >= 30:
            break
        scheduler.step()
        update_elb(config)
    with open(f"patternmod/tmpmodel_{dataset_name}.pkl", "rb") as file:
        best_model = pickle.load(file) 
    print("Best:")
    test_loss = test(best_model, epoch,  device_cpu, device_gpu, test_loader, lossFun,verbose=verbose, writer=writer)
    print(f"{test_loss=}, {best_loss=}")
    test_loss = test(best_model, epoch,  device_cpu, device_gpu, test_loader, lossFun,verbose=verbose, writer=writer)
    print(f"{test_loss=}, {best_loss=}")
    if ret_test:
        return best_model, model.fc0_enc.weight.data, trainDS, test_loader
    else:
        return best_model, model.fc0_enc.weight.data, trainDS






def build_dataset(dataset_name, device="cuda:0"):
    train_features, train_graphs, val_features, val_graphs, test_features, test_graphs = load_splited_data(dataset_name, device)
    gnn_model = load_gnn(dataset_name, device=device)
    gnn_model.eval()
    train_embeddings = concat_embeddings(dataset_name, fill_embeddings(None, train_features, train_graphs, gnn_model), train_features)
    val_embeddings = concat_embeddings(dataset_name, fill_embeddings(None, val_features, val_graphs, gnn_model), val_features)
    test_embeddings = concat_embeddings(dataset_name, fill_embeddings(None, test_features, test_graphs, gnn_model), test_features)
    train_labels = torch.concat(list(map(lambda x, edge_index:torch.argmax(gnn_model(x, edge_index)[0].softmax(-1)).int().repeat(x.shape[0]), train_features, train_graphs)))
    val_labels = torch.concat(list(map(lambda x, edge_index:torch.argmax(gnn_model(x, edge_index)[0].softmax(-1)).int().repeat(x.shape[0]), val_features, val_graphs)))
    test_labels = torch.concat(list(map(lambda x, edge_index:torch.argmax(gnn_model(x, edge_index)[0].softmax(-1)).int().repeat(x.shape[0]), test_features, test_graphs)))
    print(len(train_labels) + len(val_labels) + len(test_labels))
    return train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels




def run_dyfnapps(dataset_name, args):
    train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = build_dataset(dataset_name)
    method = args.method
    if method == "diffversify":
        """conf = TrainConfig(hidden_dim =-1 , epochs=200, weight_decay=args.lamb, elb_k=args.lamb, elb_lamb=args.lamb, class_elb_k=args.lamb, class_elb_lamb=args.lamb,
                                lambda_c = args.lambc, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                            log_interval=100, sparse_regu=0, test=False, lr=args.lr, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True, loss=weightedXorCover)"""
        conf = TrainConfig(hidden_dim = 500, epochs=50, weight_decay=8.0, elb_k=1, elb_lamb=1, class_elb_k=5, class_elb_lamb=5,
                                lambda_c = 10.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=20000, log_interval=1000, sparse_regu=0, test=False, lr=args.lr, model=DiffnapsNet,seed=14401360119984179300, loss=weightedXorCover, alpha=args.alpha,save_xp=True,k_w=10, k_f=7)
    else:
        conf = TrainConfig(hidden_dim =-1 , epochs=25, weight_decay=8.0, elb_k=1, elb_lamb=1, class_elb_k=5, class_elb_lamb=5,
                                lambda_c = 100.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.1, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True, loss=weightedXor)

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","results","real_results")
    log_dir = os.path.join(root, r"runs/")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag =  f'{current_time}_expR_{method}_db_{dataset_name}_a{conf.alpha}'
    run_dir = log_dir +  tag
    writer = SummaryWriter(log_dir=run_dir)
    writer.add_text('Run info', 'Hyperparameters:' + conf.to_str())
    #save_file_name = f"patternmod/res/{method}_{dataset_name}_{args.lr}_{args.lamb}_{args.lambc}.pkl"
    save_file_name = f"patternmod/res/{method}_{dataset_name}_default.pkl"
    if True or not os.path.isfile(save_file_name):
        model, new_weights, trainDS = learn_diffnaps_net(dataset_name, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels, conf, ret_test=False,verbose=True, writer=writer)
        with open(save_file_name, "wb") as file:
            pickle.dump((model, new_weights, trainDS), file)
    else:
        with open(save_file_name, "rb") as file:
            model, new_weights, trainDS = pickle.load(file)
    if args.pmining:
        translator = None
        label_dict = {0:"0",1:"1"}
        c_w = model.classifier.weight.detach().cpu()
        enc_w = model.fc0_enc.weight.data.detach().cpu()
        file = os.path.join(run_dir, "classif_weight.pt")
        torch.save(c_w, file)
        _,_,_,num_pat,res_dict, gen_patterns = get_positional_patterns(enc_w,c_w, general=True, t_mean=1, t1=args.t1,t2=args.t2, device=device)
        print(sum(list(map(len, res_dict.values()))))
        metric_result = {'method':method, 'db':dataset_name, 'ktop':0, 'NMF':'_'}
        metric_result.update(mean_compute_metric(train_embeddings, train_labels, res_dict, device=device))
        res_dict_int = {int(k):v for k,v in res_dict.items()}
        line_x, line_y, auc = roc(res_dict_int, train_embeddings.detach().cpu().numpy(),train_labels.detach().cpu().numpy(),label_dict,translator,verbose=False)
        metric_result['roc_auc'] = auc
        df_metric = pd.DataFrame(metric_result, index=[0])
        for key, value in metric_result.items():
            if key != "method" and key != 'db' and key != 'NMF':
                writer.add_scalar(key, value, 0)
        dr = pd.DataFrame({'x':line_x, 'y':line_y})
        dr.to_csv(os.path.join(run_dir, 'auc_roc_data_noNMF.csv'))

        res_to_csv(method, dataset_name, res_dict_int, train_embeddings.detach().cpu().numpy(), train_labels.detach().cpu().numpy(), label_dict, translator, output=run_dir)
        class_pat = [res_dict]
        if method == 'diffversify':
            new_p = compile_new_pat_by_class(labels=train_labels, patterns=res_dict, data=train_embeddings, n=[2,3], device=device, max_iter=200, rank=conf.k_f)
            for keyk, val in new_p.items():
                print(sum(list(map(len, val.values()))))
                metric_result = {'method':method,'db':dataset_name, 'ktop':keyk, 'NMF':'filter'}
                metric_result.update(mean_compute_metric(train_embeddings.detach().cpu().numpy(), train_labels.detach().cpu().numpy(), val, device=device))
                res_dict_int = {int(float(k)):v for k,v in val.items()}
                line_x, line_y, auc = roc(res_dict_int, train_embeddings.detach().cpu().numpy(),train_labels.detach().cpu().numpy(),label_dict,translator,verbose=False)
                metric_result['roc_auc'] = auc
                df_metric = pd.concat([df_metric, pd.DataFrame(metric_result, index=[0])], ignore_index=True)
                
                for key, value in metric_result.items():
                    if key != "method" and key != 'db' and key != 'NMF':
                                writer.add_scalar(key, value, keyk)

                dr = pd.DataFrame({'x':line_x, 'y':line_y})
                dr.to_csv(os.path.join(run_dir, f'auc_roc_data_NMF_filter_{keyk}.csv'))
        file = rf"expR_{dataset_name}_{method}.csv"
        df = pd.DataFrame()
        df_metric.to_csv(os.path.join(run_dir, file),index=False)
        df = pd.concat([df, df_metric], ignore_index=True)
        print(df[['ktop','NMF','pat_count','cov','supp','purity','wf1_quant','roc_auc']])
        class_pat.append(new_p)
        #with open(f"patternmod/res/{args.method}_{args.dataset_name}_{args.t1=}_{args.t2}_patterns_{args.lr}_{args.lamb}_{args.lambc}.pkl", "wb") as file:
        with open(f"patternmod/res/{args.method}_{args.dataset_name}_patterns_default.pkl", "wb") as file:
            pickle.dump((num_pat, class_pat, gen_patterns), file)

