import torch
import torch.nn as nn

from metal_predictor.constants import RESIDUES
from metal_predictor.models.layers import MetalMPNNLayer
from metal_predictor.models.attention import MetalPredictionHead


class MetalPredictionModel(nn.Module):
    def __init__(self, input_dim=42, node_dim=64, edge_dim=4, hidden_dim=64,
                 attn_mode='softmax', top_k=10, res_emb_dim=16,
                 temperature=1.0, learn_temperature=False,
                 use_plm=False, esm_dim=480, num_hypotheses=1):
        super().__init__()
        self.use_plm       = use_plm
        self.res_embedding = nn.Embedding(len(RESIDUES), node_dim)
        self.node_embed    = nn.Linear(input_dim, node_dim)
        self.layers = nn.ModuleList([
            MetalMPNNLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(3)
        ])
        self.head = MetalPredictionHead(
            node_dim, attn_mode=attn_mode, top_k=top_k, res_emb_dim=res_emb_dim,
            temperature=temperature, learn_temperature=learn_temperature,
            use_plm=use_plm, esm_dim=esm_dim, num_hypotheses=num_hypotheses,
        )

    def forward(self, x, edge_index, edge_attr, batch, res_type, pos, pos_distal,
                return_attn=False, esm=None):
        x = self.node_embed(x) + self.res_embedding(res_type)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.head(x, pos, pos_distal, batch, res_type,
                         return_attn=return_attn, esm=esm)
