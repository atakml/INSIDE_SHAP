from representationmod.visualise import visualize_graph, visualize_shap_inside, visualize_inside
from shapmod.rulemask import generate_node_values, find_support
from patternmod.inside_utils import read_patterns
from torch_geometric.utils import to_networkx
from modelmod.gnn import load_gnn
from datasetmod.datasetloader import load_dataset_gnn
import argparse
import pickle
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--i', type=int, default=0, help='Instance to visualize')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model = load_gnn(dataset_name, 'cuda:0')
    pattern_dict = read_patterns(dataset_name)
    train_loader, graphs, features, labels, train_mask, val_mask, test_mask = load_dataset_gnn(dataset_name)
    for i, data in enumerate(train_loader):
        if i == args.i:
            data = data
            break
    with open(f"{dataset_name} inside shap explanations gcn.pkl", "rb") as file:
        shap_dict = pickle.load(file)
    target_class = torch.argmax(model(data.x, data.edge_index)[0])
    colors = []
    for i, pattern in enumerate(pattern_dict): 
        if pattern['layer'] == 3:
            continue
        node_support = find_support(model.embeddings(data.x, data.edge_index)[pattern['layer']] if pattern['layer'] < 3 else data.x, pattern["components"])
        if not node_support.shape[0]:
            continue
        shap_values = ({0: (shap_dict[(args.i,0)][0][i].clone(),)}, {0: (shap_dict[(args.i,1)][0][i].clone(),)})
        node_values = generate_node_values(data.x, data.edge_index, model, shap_values, [pattern], device="cuda:0")
        colors.extend(node_values.tolist())
        if i == 1:
            print(shap_values)
            print(args.i)

    colors = list(set(colors))
    for i, pattern in enumerate(pattern_dict): 
        node_support = find_support(model.embeddings(data.x, data.edge_index)[pattern['layer']] if pattern['layer'] < 3 else data.x, pattern["components"])
        if not node_support.shape[0] or pattern['layer'] == 3:
            continue
        shap_values = ({0: (shap_dict[(args.i,0)][0][i],)}, {0: (shap_dict[(args.i,1)][0][i],)})
        node_values = generate_node_values(data.x, data.edge_index, model, shap_values, [pattern], device="cuda:0")
        try:
            node_values = list(map(lambda x: (x - min(colors))/(max(colors) - min(colors)), node_values.tolist()))
        except:
            print(i)
            print(shap_values)
            raise
        data.node_values = node_values
        print(node_values)
        visualize_graph(dataset_name, data, f"representationmod/figs/{dataset_name}/instance", args.i, "inside", f"{args.i}_{i}_inside.png", f"Pattern:{i} Shap value:{shap_values[target_class][0][0].item():.4f}", num_colors=len(colors))
    shap_values = (shap_dict[(i,0)], shap_dict[(i,1)])
    node_values = generate_node_values(data.x, data.edge_index, model, shap_values, pattern_dict, device="cuda:0")
    """node_values = torch.tensor([0.0167, 0.0000, 0.0306, 0.0159, 0.0309, 0.0298, 0.0297, 0.0298, 0.0306,
    0.0248, 0.0399, 0.0232, 0.0360, 0.0411, 0.0472, 0.0459, 0.0486, 0.0491,
    0.0432, 0.0471, 0.0522, 0.0848, 0.0835, 0.0929, 0.0956, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]).cuda()"""
    data.node_values = list(map(lambda x: ((x - node_values.min())/(node_values.max() - node_values.min())).item(), node_values))
    visualize_graph(dataset_name, data, f"representationmod/figs/{dataset_name}/instance", args.i, "inside", f"{args.i}_total_inside.png", f"SHAP-INSIDE mask")
    visualize_graph(dataset_name, data, f"representationmod/figs/{dataset_name}/instance", args.i, "svx", f"{args.i}_svx.png", f"svx mask")
