import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from metal_predictor.constants import DONOR_SLOTS, RESIDUES, RES_TO_IDX

# (num_residue_types, 14) bool: which atom14 slots hold sidechain metal-donor atoms
DONOR_MASK = torch.zeros(len(RESIDUES), 14, dtype=torch.bool)
for _res, _slots in DONOR_SLOTS.items():
    DONOR_MASK[RES_TO_IDX[_res], list(_slots)] = True


def donor_cluster_predict(batch, radius=4.0, coord_frame='absolute'):
    """
    Model-free baseline. Per graph: find the HIS/CYS/ASP/GLU sidechain donor atom
    with the most donor atoms from *other* residues within `radius`, and predict
    the mean of that atom and its neighbours. Ties are broken by the number of
    distinct residues in the neighbourhood. Falls back to the CA centroid when the
    graph has no donor pair within `radius`.

    Returns (pred (G, 3), fallback (G,) bool).
    """
    x14     = batch.x.view(-1, 14, 3)
    present = (x14 != 0).any(-1)                      # before un-reframing
    if coord_frame == 'residue':
        x14 = x14 + batch.pos.unsqueeze(1)
    elif coord_frame == 'centroid':
        x14 = x14 + global_mean_pool(batch.pos, batch.batch)[batch.batch].unsqueeze(1)
    node, slot    = (DONOR_MASK.to(x14.device)[batch.res_type] & present).nonzero(as_tuple=True)
    coords, graph = x14[node, slot], batch.batch[node]

    pred     = global_mean_pool(batch.pos, batch.batch)
    fallback = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    for g in range(pred.shape[0]):
        sel  = graph == g
        c, n = coords[sel], node[sel]
        if len(c) == 0:
            continue
        near  = (torch.cdist(c, c) <= radius) & (n[:, None] != n[None, :])
        count = near.sum(1)
        if count.max() == 0:
            continue
        n_res = (near.float() @ F.one_hot(n).float()).clamp(max=1).sum(1)
        best  = torch.where(count == count.max(), n_res, torch.full_like(n_res, -1.0)).argmax()
        near[best, best] = True
        pred[g]     = c[near[best]].mean(0)
        fallback[g] = False
    return pred, fallback
