import torch 
from torch.nn.functional import log_softmax, softmax, relu
def evaluate_metric_on_loader(dataloader, model, metric, batch=False, verbos=False):
    model.eval()
    res = 0
    t = 0 
    cnt = 0 
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if not batch:
                edge_index, features, labels = data
                labels = softmax(labels[0], -1)
                outputs = model(features[0].cuda(), edge_index[0].cuda())
                outputs = softmax(outputs)
            else:
                labels = softmax(data.y, -1)
                outputs = softmax(model(data=data), -1)

            res += metric(outputs, labels)
            if verbos:
                print(labels.shape)
                print(outputs.shape)
                print(metric(outputs, labels))
                print(labels.shape[0], res)
            cnt += labels.shape[0]
        return res / cnt#len(dataloader)

def concat_embeddings(dataset_name, embeddings, features, stack=False):
    embeddings = list(map(torch.hstack, embeddings))
    if stack:
    	embeddings = torch.stack(embeddings, axis=0).bool().float()
    else:
    	embeddings = torch.vstack(embeddings).bool().float()
    shape = embeddings.shape
    if dataset_name not in ["ba2"]:
        if not stack:
            atoms = torch.vstack(features).float()
            #atoms = atoms.reshape((atoms.shape[0]*atoms.shape[1], -1))
            embeddings = torch.hstack((atoms.float(), embeddings))
        else:
            #atoms = torch.stack(features).float()
            embeddings = torch.cat([embeddings, features], dim=2)
    return embeddings



def fill_embeddings(embeddings, features, graphs, model):
    if embeddings is None:
        embeddings = list(map(lambda edge_index, x: model.embeddings(x, edge_index), graphs, features))
    return embeddings
