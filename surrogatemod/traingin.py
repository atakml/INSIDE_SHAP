from datetime import datetime
import pickle 
import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax, relu
from torch.optim import Adam, AdamW, RAdam, SGD, ASGD, SparseAdam, Adadelta, LBFGS
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from surrogatemod.surrogate_utils import prepare_data, load_gin_dataset
from GModel import GModel
from modelmod.gnn_utils import evaluate_metric_on_loader
from modelmod.train_gnn import initialize_weights_he
import math

def kl_loss(model_log_probs, gnn_probs):
    #print(model_log_probs.shape, gnn_probs.shape)
    loss = -gnn_probs[:, 0] * (model_log_probs[:, 0] - torch.log(gnn_probs[:, 0]+1e-8)) - gnn_probs[:, 1] * (
        	model_log_probs[:, 1] - torch.log(gnn_probs[:, 1]+1e-8))
    return loss


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    plt.clf()
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=4, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=4, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.1) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig("grad.png")

def train_gin(training_loader, model, optimizer, loss_fn, epoch_index, tb_writer):
    """
    GIN training
    :param tb_writer:
    :param epoch_index:
    :param optimizer:
    :param loss_fn:
    :param training_loader: data loader with the following inside:
    graphs: graph structures obtained by initial dataset, in the form of adjacency matrix
    pattern_features: the output of GINUtil.generate_features
    labels: predicted probability of each instance obtained by the original model
    :param model: model to train
    :return:
    """
    running_loss = 0.
    last_loss = 0.
    loss = 0
    optimizer.zero_grad()
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        #edge_index, features, labels = data
        labels = softmax(data.y, -1)
        #labels = softmax(labels[0], -1)
        #print(labels.shape)
        #outputs = log_softmax(model(features[0].cuda(), edge_index[0].cuda()), -1)
        outputs = log_softmax(model(data=data), -1)
        loss = loss_fn(outputs, labels).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        optimizer.step()
        optimizer.zero_grad()




def train_for_dataset(dataset_name, method="inside", random=False, device="cuda:0"):
    training_loader, validation_loader, test_data_loader, _ = load_gin_dataset(dataset_name, method, random=random, device=device, batch_size=(64,1,1))
    feature_size = next(iter(training_loader)).x.shape[-1]
    print(feature_size)
    lr = 0.001
    loss_function = kl_loss  
    model = GModel(feature_size, 20, 2)
    model = model.to(device)

    initialize_weights_he(model)
    #optimizer_function = SGD(model.parameters(), lr=lr, momentum=0.01, dampening=0, nesterov=True)
    optimizer_function = Adam(model.parameters(), lr=lr, weight_decay=0.001)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    num_epochs = 200

    best_vloss = math.inf
    best_train_acc = 0 
    best_test_acc = 0
    early_stop = 20
    triger = 0 
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train()
        avg_loss = train_gin(training_loader, model, optimizer_function, loss_function, epoch_number, writer)
        model.eval()
        writer.flush()
        if True or epoch%5 == 0:

            loss_val = evaluate_metric_on_loader(validation_loader, model, lambda model_output, labels: kl_loss(
                model_output.log(), labels).sum(), batch=True)
            print("valid loass:", loss_val, len(validation_loader))

            if best_vloss > loss_val:
                triger = 0 
                model_path = 'models/{}/model_GIN_{}_{}_{}_{}_{}'.format(dataset_name, timestamp, epoch_number, dataset_name, loss_val, method)
                if random:
                    model_path = 'models/random features/{}/model_GIN_{}_{}_{}_{}_{}'.format(dataset_name, timestamp, epoch_number, dataset_name, loss_val, method)
                torch.save(model.state_dict(), model_path)
                best_vloss = loss_val
                loss_train = evaluate_metric_on_loader(training_loader, model, lambda model_output, labels: kl_loss(
                model_output.log(), labels).sum(), batch=True)
                
                loss_test = evaluate_metric_on_loader(test_data_loader, model, lambda model_output, labels: kl_loss(
                model_output.log(), labels).sum(), batch=True)
				
                acc_val = evaluate_metric_on_loader(validation_loader, model, lambda model_output, labels: torch.argmax(
                model_output) == torch.argmax(labels), batch=True)
                
                acc_test = evaluate_metric_on_loader(test_data_loader, model, lambda model_output, labels: torch.argmax(
                model_output) == torch.argmax(labels), batch=True)

                acc_train = evaluate_metric_on_loader(training_loader, model, lambda model_output, labels: (torch.argmax(
                model_output,-1) == torch.argmax(labels, -1)).sum(), batch=True, verbos=False)
                #best_vacc, best_train_acc, best_test_acc = acc_validation, acc_train, acc_test
                best_vacc, best_train_acc, best_test_acc, best_vloss, best_train_loss, best_test_loss  = acc_val, acc_train, acc_test, loss_val, loss_train, loss_test
                print(f"{best_train_acc=}, {best_vacc=}, {best_test_acc=}, {best_train_loss=}, {best_vloss=}, {best_test_loss=}")
            else:
                triger += 1
        epoch_number += 1
        if triger == early_stop:
            break
    random_string = "random features/" if random else ""
    result_file = f"models/{random_string}{dataset_name}_{method}_results.txt"	
    with open(result_file, "a") as file:
        file.write(f"{best_train_acc=}, {best_vacc=}, {best_test_acc=}, {best_train_loss=}, {best_vloss=}, {best_test_loss=}")
    print(f"{best_train_acc=}, {best_vacc=}, {best_test_acc=}")
    model_positive_label, true_positive_label = evaluate_metric_on_loader(test_data_loader, model,
                                                                          lambda model_output, labels: torch.tensor((
                                                                              torch.argmax(
                                                                                  model_output),
                                                                              torch.argmax(
                                                                                  labels))), batch=True)
    print(
        f"total: {len(test_data_loader)}, true_positive: {true_positive_label}, model_positive: {model_positive_label}")

		

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    parser.add_argument("--random", action="store_true", help="Use pretrained model (default: False)")
    args = parser.parse_args()
    train_for_dataset(args.dataset_name, args.method, args.random, args.device)
