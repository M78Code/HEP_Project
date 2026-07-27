"""GravNet classifier augmented with a 172-D TOF paddle energy branch."""
import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_mean_pool


class GravNetTOFClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 128,
        graph_feat_dim: int = 45,
        tof_paddle_dim: int = 172,
        space_dimensions: int = 4,
        propagate_dimensions: int = 22,
        k: int = 8,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.pre_linears = nn.ModuleList()
        self.gravnet_layers = nn.ModuleList()
        self.post_norms = nn.ModuleList()

        current_dim = in_channels
        for _ in range(num_blocks):
            self.pre_linears.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
            ))
            self.gravnet_layers.append(GravNetConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                space_dimensions=space_dimensions,
                propagate_dimensions=propagate_dimensions,
                k=k,
            ))
            self.post_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        self.skip_linear = nn.Linear(in_channels, hidden_dim)
        graph_dim = hidden_dim * num_blocks + hidden_dim

        self.tof_branch = nn.Sequential(
            nn.Linear(tof_paddle_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(graph_dim + graph_feat_dim + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, graph_feat, tof_paddle_energy):
        x_skip = self.skip_linear(x)
        block_outputs = []
        x_cur = x
        for pre_linear, gravnet, norm in zip(
                self.pre_linears, self.gravnet_layers, self.post_norms):
            x_cur = pre_linear(x_cur)
            x_cur = gravnet(x_cur, batch=batch)
            x_cur = norm(x_cur).relu()
            block_outputs.append(x_cur)

        graph_embedding = global_mean_pool(
            torch.cat(block_outputs + [x_skip], dim=1), batch)
        tof_embedding = self.tof_branch(tof_paddle_energy)
        fused = torch.cat(
            [graph_embedding, graph_feat, tof_embedding], dim=1)
        return self.classifier(fused)
