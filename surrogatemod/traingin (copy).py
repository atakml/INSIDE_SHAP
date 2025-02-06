from codebase import config
from torch.nn.functional import log_softmax, softmax, relu
from torch.optim import Adam, AdamW, RAdam, SGD, ASGD, SparseAdam, Adadelta, LBFGS
from torch.utils.tensorboard import SummaryWriter
import torch
from datetime import datetime
from codebase.GINUtil import prepare_data, init_gin, load_gin_dataset
from codebase.GIN import GIN
import pickle 
from torchviz import make_dot
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from GModel import GModel


feature_dict = {}


def kl_loss(model_log_probs, gnn_probs):
    # return -model_log_probs[torch.argmax(gnn_probs)]

    #if gnn_probs[0] == 0:
    #    gnn_probs[0] = torch.exp(torch.tensor(-100))
    #if gnn_probs[1] == 0:
    #    gnn_probs[1] = torch.exp(torch.tensor(-100))
    loss = -gnn_probs[0] * (model_log_probs[0] - torch.log(gnn_probs[0]+1e-8)) - gnn_probs[1] * (
            model_log_probs[1] - torch.log(gnn_probs[1]+1e-8))
    #print(loss.grad)
    return loss
    #if loss < 0.01 :
    #    print(model_log_probs.exp(), gnn_probs, loss)
    return loss *(1 + relu((gnn_probs[0] - 0.5)/(0.5 - model_log_probs[0].exp())))
    return -gnn_probs[0] * (model_log_probs[0] - torch.log(gnn_probs[0]+1e-8)) - gnn_probs[1] * (
            model_log_probs[1] - torch.log(gnn_probs[1]+1e-8))
    return -torch.exp(model_log_probs[0]) * (model_log_probs[0] - torch.log(gnn_probs[0])) - torch.exp(model_log_probs[1]) * (
            model_log_probs[1] - torch.log(gnn_probs[1]))
flag = True


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
    print(f"{ave_grads=}")
    print(f"{max_grads=}")
    """" plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=4, color="c")
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



    plt.savefig("grad.png")"""

def train_gin(training_loader, model, optimizer, loss_fn, epoch_index, tb_writer, train_mask):
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
    global feature_dict
    global flag
    random_key = "train"
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    cnt = 0
    loss = 0
    optimizer.zero_grad()
    for i, data in enumerate(training_loader):
        if False:#i not in train_mask:
            continue
        # Every data instance is an input + label pair
        edge_index, features, labels = data
        if random_key not in feature_dict.keys():#random
        #    print("?"*100)
            feature_dict[random_key] = []#random
        while len(feature_dict[random_key]) <= i:#random
            feature_dict[random_key].append(torch.rand_like(features.float()))#random
        
        features = feature_dict[random_key][i]#random
        #print(labels[0][0], labels[0][0].shape, softmax(labels[0][0], -1))
        labels = softmax(labels[0][0], -1)
        # Zero your gradients for every batch!
        cnt += 1
        # Make predictions for this batch
        #print(features[0].type())
        
        outputs = log_softmax(model(features[0].cuda(), edge_index[0].cuda())[0], -1)#remove zero while coming back to GIN
        #t = model(features[0], edge_index[0])
        #print(features[0].shape, edge_index[0].shape, t, t.shape, log_softmax(t, -1), log_softmax(t, -1).shape)
        #input()
        # print("output:", outputs[1].reshape([1, -1]))
        # print(labels[1].reshape([1, -1]))
        # Compute the loss and its gradients
        #loss += loss_fn(outputs, labels)
        # print(loss)
        loss = loss_fn(outputs, labels)
        # print(outputs.exp(), labels.exp())
        # target_class = torch.argmax(labels)
        # loss = loss_fn(torch.tensor(float(outputs[1].exp()>0.5), requires_grad=True), torch.tensor(float(labels[1].exp() > 0.5), requires_grad=True))
        #running_loss += loss
        if True: # cnt == 5:
            loss.backward()
            #print(loss.grad)
            #plot_grad_flow(model.named_parameters())
            #input()
            if not flag:
                grad_map = make_dot(loss, params = model.named_parameters(), show_saved=False)
                grad.format = 'png'
                grad.render('model_arch.png')
                flag = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            # print("loss:", loss.grad)
            # Adjust learning weights
            optimizer.step()
            optimizer.zero_grad()
            cnt = 0
            #running_loss = 0
            # Gather data and report

        # if i % 1000 == 999:
        #    last_loss = running_loss / 1000  # loss per batch
        #    print('  batch {} loss: {}'.format(i + 1, last_loss))
        #    tb_x = epoch_index * len(training_loader) + i + 1
        #    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #    # running_loss = 0.
    """loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
    # print("loss:", loss.grad)
    # Adjust learning weights
    optimizer.step()
    optimizer.zero_grad()"""
    #input()
    #return running_loss / len(training_loader)


def generate_train_masks(train_loader, number_of_epoches, ratio):
    positive_indices = []
    negative_indices = []
    for i, data in enumerate(train_loader):
        _, _, label = data
        if label[0][0][0] > label[0][0][1]:
            negative_indices.append(i)
        else:
            positive_indices.append(i)
    positive_indices = torch.tensor(positive_indices)
    negative_indices = torch.tensor(negative_indices)

    masks = []
    for _ in range(number_of_epoches):
        selected_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:int(ratio*len(positive_indices))]]
        selected_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:int(ratio*len(negative_indices))]]
        selected_indices = torch.sort(torch.concat((selected_positive_indices, selected_negative_indices)))[0]
        masks.append(selected_indices)
    return masks

def evaluate_metric_on_loader(dataloader, model, metric, random=False, random_key=None):
    global feature_dict
    model.eval()
    res = 0
    t = 0 
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            edge_index, features, labels = data
            if random:
                if random_key not in feature_dict.keys():
                    feature_dict[random_key] = []
                while len(feature_dict[random_key]) <= i:
                    feature_dict[random_key].append(torch.rand_like(features.float()))
                features = feature_dict[random_key][i]
            labels = softmax(labels[0][0], -1)
            outputs = model(features[0].cuda(), edge_index[0].cuda())[0]#remove zero while coming back to GIN
            outputs = softmax(outputs)
            #print(f"{labels=}, {outputs=}, {metric(outputs, labels)=}")
            res += metric(outputs, labels)
        return res / len(dataloader)

import torch.nn.init as init

# Define a function to apply He initialization to model layers
def initialize_weights_he(model):
    # Iterate through all layers in the model
    for layer in model.modules():
        # Check if the layer has a `weight` attribute (e.g., Conv layers, Linear layers)
        if hasattr(layer, 'weight') and layer.weight is not None:
            if layer.weight.dim() >= 2:
                print("!"*100)
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
        # Check if the layer has a `bias` attribute and initialize it to zeros
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)

def train_for_dataset(dataset):
    lr = 0.01

    loss_function = kl_loss  # KLDivLoss(reduction="none")
    # loss_function = BCELoss()
    #model = init_gin(dataset)
    model = GModel(52, 20, 2)
    model = model.to("cuda:0")

    initialize_weights_he(model)
    optimizer_function = SGD(model.parameters(), lr=lr, momentum=0.01, dampening=0, nesterov=True)
    #optimizer_function = LBFGS(model.parameters(), lr=lr)
    #optimizer_function = Adam(model.parameters(), lr=lr)#, amsgrad=True)#, betas=(0.7, 0.8)) #, momentum=0.1, dampening=0, nesterov=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    num_epochs = 100

    best_vacc = 0
    training_loader, validation_loader, test_data_loader, _ = load_gin_dataset(dataset)
    print(len(training_loader))
    #print(evaluate_metric_on_loader(training_loader, model, lambda x, y: torch.tensor((int(y[0]>=0.5), int(y[1]>0.5)))))
    #print(evaluate_metric_on_loader(test_data_loader, model, lambda x, y: torch.tensor((int(y[0]>=0.5), int(y[1]>0.5)))))
    #print(evaluate_metric_on_loader(validation_loader, model, lambda x, y: torch.tensor((int(y[0]>=0.5), int(y[1]>0.5)))))
   
    #train_masks = generate_train_masks(training_loader, num_epochs, 0.2)
    #with open(f"{dataset}_train_masks.pkl", "wb") as file:
    #    pickle.dump(train_masks, file)
    #with open(f"{dataset}_train_masks.pkl", "rb") as file:
    #    train_masks = pickle.load(file)
    train_masks = list(range(len(training_loader)))
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train()
        avg_loss = train_gin(training_loader, model, optimizer_function, loss_function, epoch_number, writer, train_masks[epoch])
        #for param_group in optimizer_function.param_groups:
        #    print(f"Epoch {epoch+1}, Learning Rate: {param_group['lr']}")
        model.eval()
        #print(avg_loss.retain_grad())
        #avg_validation_loss = evaluate_metric_on_loader(validation_loader, model,
        #                                                lambda model_output, labels: kl_loss(torch.log(model_output),
        #                                                                                     labels), random=False, random_key="valid")
        #print('LOSS train {} valid {}'.format(avg_loss, avg_validation_loss))

        #writer.add_scalars('Training vs. Validation Loss',
        #                   {'Training': avg_loss, 'Validation': avg_validation_loss},
        #                   epoch_number + 1)
        writer.flush()
        if epoch%5 == 0:
            acc_test = evaluate_metric_on_loader(test_data_loader, model, lambda model_output, labels: torch.argmax(
                model_output) == torch.argmax(labels), random=True, random_key="test")
            print("test acc:", acc_test)

            acc_validation = evaluate_metric_on_loader(validation_loader, model, lambda model_output, labels: torch.argmax(
                model_output) == torch.argmax(labels), random=True, random_key="valid")
            print("valid acc:", acc_validation, len(validation_loader))

            if best_vacc < acc_validation:
                model_path = 'models/random features/{}/model_GIN_{}_{}_{}_{}'.format(dataset, timestamp, epoch_number, dataset, acc_validation)
                #model_path = 'models/{}/model_GIN_{}_{}_{}_{}'.format(dataset, timestamp, epoch_number, dataset, acc_validation)
                torch.save(model.state_dict(), model_path)
                best_vacc = acc_validation

            acc_train = evaluate_metric_on_loader(training_loader, model, lambda model_output, labels: torch.argmax(
                model_output) == torch.argmax(labels), random=True, random_key="train")
            #loss_train = evaluate_metric_on_loader(training_loader, model,
            #                                       lambda model_output, labels: kl_loss(torch.log(model_output),
            #                                                                            labels), random=False, random_key="train")
            print("train acc:", acc_train, len(training_loader))
        epoch_number += 1

    model_positive_label, true_positive_label = evaluate_metric_on_loader(test_data_loader, model,
                                                                          lambda model_output, labels: torch.tensor((
                                                                              torch.argmax(
                                                                                  model_output),
                                                                              torch.argmax(
                                                                                  labels))), random=True, random_key='test')
    print(
        f"total: {len(test_data_loader)}, true_positive: {true_positive_label}, model_positive: {model_positive_label}")


if __name__ == "__main__":
    train_for_dataset("AlkaneCarbonyl")
