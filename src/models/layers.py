import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer supporting PyG edge_index and sparse tensors.
    """
    def __init__(self, in_ft: int, out_ft: int, act=nn.PReLU(), bias: bool = True):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_ft, out_ft, bias=bias)
        self.act = act

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        if self.act is not None:
            out = self.act(out)
        return out


class GATv2Layer(nn.Module):
    """
    Graph Attention v2 Layer with multi-head attention and residual connections.
    GATv2 uses dynamic attention that is strictly more expressive than GAT.
    """
    def __init__(self, in_ft: int, out_ft: int, heads: int = 4, act=nn.PReLU(), dropout: float = 0.1):
        super(GATv2Layer, self).__init__()
        assert out_ft % heads == 0, f"out_ft ({out_ft}) must be divisible by heads ({heads})"
        self.conv = GATv2Conv(in_ft, out_ft // heads, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_ft)
        self.act = act
        # Residual projection if dimensions don't match
        self.residual = nn.Linear(in_ft, out_ft) if in_ft != out_ft else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        out = self.norm(out + self.residual(x))  # Residual + LayerNorm
        if self.act is not None:
            out = self.act(out)
        return out


class AvgReadout(nn.Module):
    """Computes global/subgraph average readout."""
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.mean(seq, dim=0, keepdim=True)


class MaxReadout(nn.Module):
    """Computes global/subgraph max readout."""
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.max(seq, dim=0, keepdim=True).values


class MinReadout(nn.Module):
    """Computes global/subgraph min readout."""
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.min(seq, dim=0, keepdim=True).values


class WSReadout(nn.Module):
    """Weighted sum attention readout matching GraphNC specification."""
    def __init__(self, hidden_dim: int):
        super(WSReadout, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(seq), dim=0)
        return torch.sum(seq * weights, dim=0, keepdim=True)


class BilinearDiscriminator(nn.Module):
    """
    Bilinear Discriminator for contrastive score discrimination.
    """
    def __init__(self, hidden_dim: int, negsamp_round: int = 1):
        super(BilinearDiscriminator, self).__init__()
        self.f_k = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.negsamp_round = negsamp_round
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f_k.weight.data)
        if self.f_k.bias is not None:
            self.f_k.bias.data.fill_(0.0)

    def forward(self, c: torch.Tensor, h_pl: torch.Tensor) -> torch.Tensor:
        # Positive score
        scs = [self.f_k(h_pl, c.expand_as(h_pl))]

        # Negative samples via circular shift
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1:], c_mi[:-1]), dim=0)
            scs.append(self.f_k(h_pl, c_mi.expand_as(h_pl)))

        return torch.cat(tuple(scs), dim=0)
