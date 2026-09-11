import torch
import torch.nn as nn

from metal_predictor.constants import RESIDUES
from metal_predictor.models.layers import MetalMPNNLayer
from metal_predictor.models.attention import MetalPredictionHead


class MetalPredictionModel(nn.Module):
    def __init__(self, input_dim=42, node_dim=64, edge_dim=4, hidden_dim=64,
                 attn_mode='softmax', top_k=10, res_emb_dim=16,
                 temperature=1.0, learn_temperature=False,
                 use_plm=False, esm_dim=480, num_hypotheses=1, num_mpnn_rounds=3,
                 use_node_features=None):
        super().__init__()
        self.use_plm         = use_plm
        self.num_mpnn_rounds = num_mpnn_rounds
        if use_node_features is None:
            use_node_features = num_mpnn_rounds > 0
        self.res_embedding  = nn.Embedding(len(RESIDUES), node_dim)
        self.node_embed     = nn.Linear(input_dim, node_dim)
        self.layers = nn.ModuleList([
            MetalMPNNLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_mpnn_rounds)
        ])
        self.head = MetalPredictionHead(
            node_dim, attn_mode=attn_mode, top_k=top_k, res_emb_dim=res_emb_dim,
            temperature=temperature, learn_temperature=learn_temperature,
            use_plm=use_plm, esm_dim=esm_dim, num_hypotheses=num_hypotheses,
            use_node_features=use_node_features,
        )

    def forward(self, x, edge_index, edge_attr, batch, res_type, pos, pos_distal,
                return_attn=False, esm=None):
        if self.num_mpnn_rounds > 0:
            x = self.node_embed(x) + self.res_embedding(res_type)
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
        else:
            x = None
        return self.head(x, pos, pos_distal, batch, res_type,
                         return_attn=return_attn, esm=esm)
