import torch

from metal_predictor.models.model import MetalPredictionModel


def load_model(checkpoint, device=None, **model_kwargs):
    """Load a MetalPredictionModel from a checkpoint file."""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')
    model = MetalPredictionModel(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model.eval()
    return model


def neighborhood_predict(model, batch, use_plm=False, device=None):
    """
    Alternative to the soft-attention weighted sum: find the node whose k-NN
    neighborhood carries the most total attention mass, then return the CA centroid
    of that neighborhood as the predicted metal position.

    Works with any existing checkpoint — no retraining needed.

    Neighborhood score:  s_i = α_i + Σ_{j ∈ N(i)} α_j
    This is the total probability mass in node i's local neighborhood, and equals
    the first-order approximation of P(at least one binding residue in N(i)).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        esm_feat = batch.esm.to(device) if (use_plm and hasattr(batch, 'esm')) else None
        _, attn  = model(batch.x, batch.edge_index, batch.edge_attr,
                         batch.batch, batch.res_type, batch.pos, batch.pos_distal,
                         return_attn=True, esm=esm_feat)

    # If multi-hypothesis, attn is a list of (N,) tensors — average them
    if isinstance(attn, list):
        attn = torch.stack(attn, dim=0).mean(dim=0)

    # For each edge src→dst, add α[dst] to the score of src
    # Result: neigh_score[i] = α_i + Σ α_j for all j ∈ k-NN(i)
    neigh_score = attn.clone()
    neigh_score.scatter_add_(0, batch.edge_index[0], attn[batch.edge_index[1]])

    num_graphs = batch.batch.max().item() + 1
    preds = []
    for g in range(num_graphs):
        mask      = batch.batch == g
        node_idx  = mask.nonzero(as_tuple=True)[0]
        scores_g  = neigh_score[mask]
        best_local  = scores_g.argmax().item()
        best_global = node_idx[best_local].item()

        # k-NN neighbors of the best node (from edge_index)
        neigh_mask  = batch.edge_index[0] == best_global
        neigh_nodes = batch.edge_index[1][neigh_mask]
        all_nodes   = torch.cat([
            torch.tensor([best_global], device=batch.pos.device),
            neigh_nodes,
        ])
        centroid = batch.pos[all_nodes].mean(dim=0)
        preds.append(centroid)

    return torch.stack(preds, dim=0)
