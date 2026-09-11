import torch
from torch_geometric.data import Data

from metal_predictor.data.dataset import MetalBindingDataset


def random_rotation_matrix(device='cpu'):
    """Sample a uniformly random 3D rotation matrix via QR decomposition."""
    Z = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(Z)
    Q = Q * torch.sign(torch.linalg.det(Q))
    return Q


def augment_data(data: Data) -> Data:
    """
    Random SO(3) rotation applied to x (14 atoms), pos, pos_distal, and y.
    Edge attributes and ESM embeddings are invariant (distance-based / sequence-based).
    """
    R          = random_rotation_matrix()
    n          = data.x.shape[0]
    num_metals = data.num_metals if hasattr(data, 'num_metals') else data.y.shape[0]
    pos_distal = data.pos_distal if hasattr(data, 'pos_distal') else data.pos
    augmented  = Data(
        x=(data.x.reshape(n, 14, 3) @ R.T).reshape(n, 42),
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        y=data.y @ R.T,
        num_metals=num_metals,
        res_type=data.res_type,
        pos=data.pos @ R.T,
        pos_distal=pos_distal @ R.T,
    )
    if hasattr(data, 'esm'):
        augmented.esm = data.esm
    if hasattr(data, 'closest_residue_idx'):
        augmented.closest_residue_idx = data.closest_residue_idx
    return augmented


class AugmentedDataset(MetalBindingDataset):
    """Applies a fresh random SO(3) rotation on every __getitem__ call."""
    def __getitem__(self, idx):
        return augment_data(super().__getitem__(idx))
