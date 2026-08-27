"""DAGCN model (ported from official DAGCN code, tao2025dagcn).

DAGCNModel: dual ResNet50 (cnn 2048-d features + dsa 1000-class scores for the graph)
+ torch_geometric GCN (in=1000, hidden=256, layers=3, out=64) -> concat 2048+64 = 2112-d.
Returns (features, scores). GCN output is 64-d in the official weights (not 150), hence 64.

Requires torch_geometric (pip install torch_geometric).
"""
import torch
import torch.nn as nn
from torch_geometric import nn as geometric_nn
from torch_geometric.data import Data as geometric_data
from torch_geometric.utils import dense_to_sparse
from torchvision.models import resnet50, ResNet50_Weights


class DAGCNModel(nn.Module):
    def __init__(self, gcn_hidden_channels=256, gcn_layers=3, gcn_out_channels=64,
                 gcn_dropout=0.2):
        super(DAGCNModel, self).__init__()
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        features = self.cnn.fc.in_features
        self.combined_features = features + gcn_out_channels  # 2048 + 64 = 2112

        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # -> 2048
        self.dsa = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # full 1000-class output

        gcn_in_channels = 1000  # DSA module output
        self.gcn = geometric_nn.GCN(in_channels=gcn_in_channels,
                                    hidden_channels=gcn_hidden_channels,
                                    num_layers=gcn_layers,
                                    out_channels=gcn_out_channels,
                                    dropout=gcn_dropout)
        self.gcn_out_channels = gcn_out_channels

    def forward(self, x):
        features = self.cnn.forward(x)
        scores = self.dsa.forward(x)

        transposed_scores = torch.transpose(scores, 0, 1)
        adjacency_matrix = torch.matmul(scores, transposed_scores)
        sparse_adj_matrix = dense_to_sparse(adjacency_matrix)
        edge_index = sparse_adj_matrix[0]
        graph = geometric_data(scores, edge_index=edge_index)

        gcn_features = self.gcn(graph.x, graph.edge_index)
        gcn_features = gcn_features.view(-1, self.gcn_out_channels, 1, 1)

        concat_features = torch.cat([features, gcn_features], dim=1)
        concat_features = concat_features.view(-1, self.combined_features)
        return concat_features, scores


class Classifier(nn.Module):
    def __init__(self, features, num_classes, prob=0.3):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(features, 1024),
            nn.ReLU(),
            nn.Dropout(p=prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=prob),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        pseudo_label = self.classifier(x)
        mid_out = self.classifier[:-1](x)
        return pseudo_label, mid_out


class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_dims=500, output_dims=2):
        super(Discriminator, self).__init__()
        self.restored = False
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, input):
        out = self.layer(input)
        out = torch.sigmoid(out)
        return out
