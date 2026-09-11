import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax

from metal_predictor.constants import RESIDUES


def graph_sparsemax(logits, batch):
    z = logits.squeeze(1)
    num_graphs = batch.max().item() + 1
    out = torch.zeros_like(z)
    for g in range(num_graphs):
        mask = batch == g
        z_g = z[mask]
        n_g = z_g.shape[0]
        z_sorted, _ = torch.sort(z_g, descending=True)
        z_cumsum = torch.cumsum(z_sorted, dim=0)
        k = torch.arange(1, n_g + 1, device=z.device, dtype=z.dtype)
        k_hat = (1 + k * z_sorted > z_cumsum).sum().clamp(min=1)
        tau = (z_cumsum[k_hat - 1] - 1.0) / k_hat
        out[mask] = torch.clamp(z_g - tau, min=0.0)
    return out.unsqueeze(1)


def graph_topk_attn(logits, batch, k):
    z = logits.squeeze(1)
    num_graphs = batch.max().item() + 1
    out = torch.zeros_like(z)
    for g in range(num_graphs):
        mask = batch == g
        z_g = z[mask]
        k_g = min(k, z_g.shape[0])
        top_vals, top_idx = torch.topk(z_g, k_g)
        vals = torch.zeros_like(z_g)
        vals[top_idx] = torch.softmax(top_vals, dim=0)
        out[mask] = vals
    return out.unsqueeze(1)


class MetalPredictionHead(nn.Module):
    """
    attn_mode        : 'softmax' | 'sparsemax' | 'topk'
    top_k            : nodes kept per graph when attn_mode='topk'
    res_emb_dim      : bypass residue embedding concatenated to attention MLP input (0 = off)
    temperature      : initial τ value; logits are divided by τ before softmax (τ < 1 → sharper)
    learn_temperature: if True, τ is a learned nn.Parameter; otherwise fixed buffer
    use_plm          : if True, expects an esm=(N, esm_dim) tensor and uses it instead of res_bypass
    esm_dim          : embedding dimension of the ESM-2 model (480 for esm2_t12_35M_UR50D)
    num_hypotheses   : 1 = standard single head (self.attention_mlp, checkpoint-compatible with all v8 runs)
                       >1 = K parallel heads (self.attention_mlps ModuleList); train with wta_loss()
    """
    def __init__(self, node_dim, attn_mode='softmax', top_k=10, res_emb_dim=16,
                 temperature=1.0, learn_temperature=False, use_plm=False, esm_dim=480,
                 num_hypotheses=1, use_node_features=True):
        super().__init__()
        self.attn_mode         = attn_mode
        self.top_k             = top_k
        self.res_emb_dim       = res_emb_dim
        self.use_plm           = use_plm
        self.use_node_features = use_node_features
        self.num_hypotheses    = num_hypotheses

        log_tau = torch.tensor(float(temperature)).log()
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_tau)
        else:
            self.register_buffer('log_temperature', log_tau)

        if use_plm:
            self.esm_proj = nn.Linear(esm_dim, node_dim)
            attn_in_dim = node_dim + node_dim if use_node_features else node_dim
        else:
            if res_emb_dim > 0:
                self.res_bypass = nn.Embedding(len(RESIDUES), res_emb_dim)
            attn_in_dim = (node_dim if use_node_features else 0) + res_emb_dim

        def _make_mlp():
            return nn.Sequential(
                nn.Linear(attn_in_dim, attn_in_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(attn_in_dim // 2, 1),
            )

        if num_hypotheses == 1:
            self.attention_mlp  = _make_mlp()
        else:
            self.attention_mlps = nn.ModuleList([_make_mlp() for _ in range(num_hypotheses)])

    def _apply_attn(self, logits, batch):
        tau = self.log_temperature.exp()
        logits = logits / tau
        if self.attn_mode == 'softmax':
            return pyg_softmax(logits, batch, dim=0)
        elif self.attn_mode == 'sparsemax':
            return graph_sparsemax(logits, batch)
        elif self.attn_mode == 'topk':
            return graph_topk_attn(logits, batch, self.top_k)
        else:
            raise ValueError(f'Unknown attn_mode: {self.attn_mode!r}')

    def _predict_one(self, mlp, attn_input, centroid, centered_distal, batch):
        weights = self._apply_attn(mlp(attn_input), batch)
        num_graphs = centroid.shape[0]
        offset = torch.zeros(num_graphs, 3, device=centroid.device)
        offset.scatter_add_(0, batch.unsqueeze(1).expand(-1, 3), weights * centered_distal)
        return centroid + offset, weights

    def forward(self, x, pos, pos_distal, batch, res_type, return_attn=False, esm=None):
        num_graphs = batch.max().item() + 1

        device = pos.device
        counts = torch.zeros(num_graphs, dtype=torch.float, device=device) \
                      .scatter_add(0, batch, torch.ones(batch.shape[0], dtype=torch.float, device=device))

        centroid = torch.zeros(num_graphs, 3, device=device)
        centroid.scatter_add_(0, batch.unsqueeze(1).expand(-1, 3), pos)
        centroid /= counts.unsqueeze(1)

        centered_distal = pos_distal - centroid[batch]

        if self.use_plm:
            if esm is None:
                raise RuntimeError(
                    'use_plm=True but no ESM embeddings on this batch. '
                    'Run the ESM-2 cache builder cell first, then set '
                    'esm_cache_dir=ESM_CACHE_DIR when constructing the dataset.'
                )
            esm_proj = self.esm_proj(esm)
            attn_input = torch.cat([x, esm_proj], dim=-1) if self.use_node_features else esm_proj
        elif self.res_emb_dim > 0:
            rb = self.res_bypass(res_type)
            attn_input = torch.cat([x, rb], dim=-1) if self.use_node_features else rb
        else:
            attn_input = x

        if self.num_hypotheses == 1:
            pred, weights = self._predict_one(self.attention_mlp, attn_input,
                                              centroid, centered_distal, batch)
            if return_attn:
                return pred, weights.squeeze(1)
            return pred

        preds, all_weights = [], []
        for mlp in self.attention_mlps:
            pred_k, w_k = self._predict_one(mlp, attn_input, centroid, centered_distal, batch)
            preds.append(pred_k)
            all_weights.append(w_k.squeeze(1))
        stacked = torch.stack(preds, dim=1)  # (N_graphs, K, 3)
        if return_attn:
            return stacked, all_weights
        return stacked
