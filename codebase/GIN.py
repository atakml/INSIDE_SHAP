from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GINConv
from torch.nn.functional import relu
from torch.nn import Linear
import torch
from torch_geometric.nn import global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, input_features, num_layers, out_channels):
        super(GIN, self).__init__()
        self.input_features = input_features
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.first_conv_layer = GINConv(MLP([input_features, out_channels, out_channels]), train_eps=True)
        self.convs.append(self.first_conv_layer)
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(MLP([out_channels, out_channels, out_channels], dropout=0.2)))
        #self.lin = Linear(out_channels*num_layers, out_channels * num_layers)
        #self.final_linear = Linear(out_channels * num_layers, 2)
        self.lin = Linear(out_channels, out_channels)
        self.final_linear = Linear(out_channels, 2)

    def forward(self, x, edge_index, batch=None):
        #xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i:
                xs = torch.concat((xs, torch.sum(x, dim=0)))
            else:
                xs = torch.sum(x, dim=0)
            #xs.append(torch.sum(x, dim=0))
        #xs = self.lin(xs)
        #xs = relu(xs)
        #final = self.final_linear(xs)#torch.concat(xs))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        x = relu(x)
        final = self.final_linear(x)
        return final
