import torch
import torch_geometric
from torch_geometric.nn import HypergraphConv
class myHGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_layers,
        dropout,
    ):
        super(myHGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            HypergraphConv(in_channels, hidden_channels)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                HypergraphConv(hidden_channels, hidden_channels)
            )
        self.lin = torch.nn.Linear(hidden_channels*num_layers, out_channels)

    def forward(self, data):
        x,edge_index,edge_attr = data.x,data.edge_index,data.edge_attr
        # message passing
        all_layer=[]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            if i != self.num_layers - 1:
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            all_layer.append(x)
        # readout layer
        # 对每层输出进行全局平均池化
        pooled_outputs = [torch_geometric.nn.global_mean_pool(output, data.batch) for output in all_layer]
        # 拼接所有层的池化输出
        x = torch.cat(pooled_outputs, dim=1)

        # 分类器
        x = self.lin(x)
        return x
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.dropout = 0.5
    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_layers}, {self.dropout})'
    
class myGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
    ):
        super(myGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
            )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x,edge_index = data.x,data.edge_index
        # message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        # readout layer
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        # classifier
        x = self.lin(x)
        return x
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.dropout = 0.5
    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_layers}, {self.dropout})'
    def __str__(self):
        return f'{self.__class__.__name__}({self.num_layers}, {self.dropout})'
    def __len__(self):
        return self.num_layers
    def __getitem__(self, index):
        if index < 0 or index >= self.num_layers:
            raise IndexError("Index out of range")
        return self.convs[index]
