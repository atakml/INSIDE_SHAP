


def select_mask(features, graphs, mask):
    masked_graphs = [graphs[i] for i in range(len(graphs)) if mask[i]]
    masked_features = [features[i] for i in range(len(features)) if mask[i]]
    return masked_features, masked_graphs

