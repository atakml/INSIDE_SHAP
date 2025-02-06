from graphxai.gnn_models.graph_classification.gcn import GCN_3layer
import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from dgl import DGLGraph
from torch_geometric.utils import from_dgl

class GModel(GCN_3layer):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GModel, self).__init__(in_channels, hidden_channels, out_channels)
        self.lin = torch.nn.Linear(2*hidden_channels, out_channels)
        """self.conv1.normalize = False
        self.conv2.normalize = False
        self.conv3.normalize = False
        self.conv1.add_self_loops = False
        self.conv2.add_self_loops = False
        self.conv3.add_self_loops = False"""

    def embeddings(self, x, edge_index, edge_weights=None, batch=None): 
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), device=x.device)
        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = out1.relu()
        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = out2.relu()
        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = out3.relu() 
        return [out1]+[out2]+[out3]

    def forward(self, x=None, edge_index=None, batch=None, edge_weights=None, data=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        if isinstance(x, DGLGraph):
            data = from_dgl(x)
            edge_index = data.edge_index
            x = data.x
        embed = self.embedding(x, edge_index, edge_weights)

        return self.decoder(embed, batch)

    def decoder(self,embed, batch=None  ):
        if batch is None:  # No batch given
            batch = torch.zeros(embed.size(0), dtype=torch.long, device=embed.device)
        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)
        #print(f"{input_lin.shape=}, {out1.shape=}, {out2.shape}")
        return self.lin(input_lin)


    def embedding(self, x, edge_index, edge_weights=None):
        return self.embeddings(x, edge_index, edge_weights)[-1]
