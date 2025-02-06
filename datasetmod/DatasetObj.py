from torch.utils.data import Dataset
import torch 
from graphxai.datasets import GraphDataset
from torch_geometric.data import Data, InMemoryDataset
class GinDataset(Dataset):
    def __init__(self, graphs, features, labels, mask):
        self.indices = torch.nonzero(torch.tensor(mask)).unsqueeze(-1)
        self.graphs = graphs
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        if self.graphs[self.indices[idx]].device != self.features[self.indices[idx]].device:
            self.graphs[self.indices[idx]] = self.graphs[self.indices[idx]].to(self.features[self.indices[idx]].device)

        return self.graphs[self.indices[idx]], self.features[self.indices[idx]], self.labels[self.indices[idx]]


class GinDatasetBatch(InMemoryDataset):
    def __init__(self, graphs, features, labels, mask, transform=None, pre_transform=None):
        self.graphs = graphs
        self.features = features
        self.labels = labels
        self.mask = torch.tensor(mask)
        
        # Initialize the parent class
        super().__init__(None, transform, pre_transform)
        
        # Create the dataset
        self.data, self.slices = self._process()

    def _process(self):
        # Filter data based on the mask
        indices = torch.nonzero(self.mask).squeeze(-1)
        
        data_list = []
        for idx in indices:
            graph = self.graphs[idx]
            feature = self.features[idx]
            label = self.labels[idx]
            
            # Ensure devices match
            if graph.device != feature.device:
                graph = graph.to(feature.device)
            
            # Create a PyG Data object
            data = Data(x=feature, edge_index=graph, y=label)
            data_list.append(data)
        
        # Use InMemoryDataset utilities to collate the data
        return self.collate(data_list)

    """def len(self):
        return self.slices['x'].size(0)"""

    """def get(self, idx):
        # Access the data using PyTorch Geometric's slicing
        return self._get_data(idx)"""



class SPMDataSet(GinDataset):
    def __getitem__(self, idx):
        index = self.indices[idx]
        if self.graphs[self.indices[idx]].device != self.features[self.indices[idx]].device:
            self.graphs[self.indices[idx]] = self.graphs[self.indices[idx]].to(self.features[self.indices[idx]].device)

        data = Data(edge_index=self.graphs[index],
                    x=self.features[index][0][0],
                    label=self.labels[index])
        return data




class ClassicDataset(GraphDataset):
    def __init__(self, name, graphs, features, labels, train_mask, val_mask, test_mask, remove_paddings=True, device="cuda:0"):
        if name not in ["Benzen", "AlkaneCarbonyl"]:
            self.graphs = [Data(x=features[i], edge_index=graphs[i], y=torch.argmax(labels[i])) for i in range(len(graphs))]
        else:
            self.graphs = [Data(x=features[i], edge_index=graphs[i], y=labels[i]) for i in range(len(graphs))]

        if remove_paddings:
            self.remove_padding()
        super().__init__(name = name, seed = None, split_sizes = (1,0,0), device = device)
        self.train_index = torch.where(torch.tensor(train_mask))[0]
        self.test_index = torch.where(torch.tensor(test_mask))[0]
        self.val_index = torch.where(torch.tensor(val_mask))[0]
        self.explanations = [None for _ in range(len(graphs))]

    def remove_padding(self):
        cnt = 0 
        for i in range(len(self.graphs)):
            data = self.graphs[i]
            wh = torch.where(data.edge_index[0] == data.edge_index[1])[0]
            if wh.shape[0]:
                last_index = wh[0]
            else: 
                continue
            number_of_nodes = data.edge_index[0][last_index]
            while number_of_nodes < data.x.shape[0]:
                if (data.edge_index[0] == number_of_nodes).sum() > 1:
                    number_of_nodes += 1
                else: 
                    break
            if number_of_nodes == data.x.shape[0]: 
                continue
            cnt += 1
            self.graphs[i].x = self.graphs[i].x[:number_of_nodes]
            self.graphs[i].edge_index = self.graphs[i].edge_index[:, :last_index]




