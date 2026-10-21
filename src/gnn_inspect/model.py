"""GNN node scorer: message passing encoder + MLP scoring head."""

import copy

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv


class NodeScorer(nn.Module):
    def __init__(self, in_dim: int, kind: str = "gcn", hidden: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kind = kind
        dims = [in_dim] + [hidden] * layers
        if kind == "gcn":
            self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims, dims[1:]))
        elif kind == "gat":
            self.convs = nn.ModuleList(GATConv(a, b // 4, heads=4) for a, b in zip(dims, dims[1:]))
        elif kind == "mlp":  # ablation: same capacity, no message passing
            self.convs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        else:
            raise ValueError(kind)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x) if self.kind == "mlp" else conv(x, edge_index)
            x = self.dropout(torch.relu(x))
        return self.head(x).squeeze(-1)


def train(model, train_data, val_data, epochs=150, lr=3e-3, batch_size=32, seed=0):
    """MSE regression of true severity; keeps the weights with the best val loss."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data))
    best, best_state = np.inf, None
    for _ in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(b.x, b.edge_index), b.y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vb = next(iter(val_loader))
            val = nn.functional.mse_loss(model(vb.x, vb.edge_index), vb.y).item()
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, data) -> np.ndarray:
    model.eval()
    return model(data.x, data.edge_index).numpy()
