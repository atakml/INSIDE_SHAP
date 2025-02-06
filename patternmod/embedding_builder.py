from modelmod.gnn_utils import *
from modelmod.gnn import load_gnn
from datasetmod.datasetloader import load_dataset_gnn
import numpy as np 

def save_file(file_name, data):
    data = data.cpu().detach().numpy()
    np.savetxt(file_name, data, delimiter=" ")

for dataset_name in ["aids", "ba2", "BBBP", "mutag", "AlkaneCarbonyl", "Benzen"]:
    print(f"{dataset_name=}")
    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name)
    model = load_gnn(dataset_name)
    labels = list(map(lambda feature, graph: torch.ones(features.shape[1])*torch.argmax(model(feature, graph)[0].softmax(-1)).item(), features, graphs))
    labels = torch.concat(labels).float()
    print(labels.shape)
    embeddings = fill_embeddings(None, features, graphs, model)
    concated_embeddings  = concat_embeddings(dataset_name, embeddings, features)
    save_file(f"data/{dataset_name}_pred.txt", labels)
    save_file(f"data/{dataset_name}_act_mat.txt", concated_embeddings)

