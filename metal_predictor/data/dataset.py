import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from torch.utils.data import Dataset
from torch_geometric.data import Data

from metal_predictor.constants import (
    BARE_METAL_RESNAMES,
    DISTAL_CACHE_DIR,
    RES_TO_IDX,
    _ATOM14_COLS,
)


def _compute_distal(coords14_raw):
    """(N, 14, 3) with NaN for missing atoms → (N, 3) last-finite atom per residue."""
    finite_mask = np.isfinite(coords14_raw).all(axis=2)   # (N, 14)
    last_idx    = (finite_mask * np.arange(14)).argmax(axis=1)
    last_idx[~finite_mask.any(axis=1)] = 1                # CA fallback for all-NaN residues
    return coords14_raw[np.arange(len(coords14_raw)), last_idx, :]


def cluster_aware_split(clusters_parquet, seed=42, val_frac=0.1, test_frac=0.1):
    """
    Load cluster assignments and return (train_ids, val_ids, test_ids) split at
    the cluster level to prevent sequence-homology leakage.
    """
    df = pd.read_parquet(clusters_parquet, columns=['structure_id', 'cluster_id'])
    cluster_to_structures = (
        df.drop_duplicates()
          .groupby('cluster_id')['structure_id']
          .apply(list)
          .to_dict()
    )
    clusters = list(cluster_to_structures.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n = len(clusters)
    n_val  = int(val_frac * n)
    n_test = int(test_frac * n)

    train_clusters = clusters[:n - n_val - n_test]
    val_clusters   = clusters[n - n_val - n_test : n - n_test]
    test_clusters  = clusters[n - n_test:]

    train_ids = [s for c in train_clusters for s in cluster_to_structures[c]]
    val_ids   = [s for c in val_clusters   for s in cluster_to_structures[c]]
    test_ids  = [s for c in test_clusters  for s in cluster_to_structures[c]]

    return train_ids, val_ids, test_ids, cluster_to_structures


def filter_single_metal(ids, cache_dir):
    """Return only structure IDs whose cached graph has exactly 1 metal."""
    result = []
    for sid in ids:
        path = os.path.join(cache_dir, f'{sid}.pt')
        if not os.path.exists(path):
            continue
        d = torch.load(path, weights_only=False)
        if d.y.shape[0] == 1:
            result.append(sid)
    return result


def single_metal_resplit(cluster_to_structures, cache_dir, seed=42):
    """
    Re-run the cluster-level split over single-metal structures only.

    Multi-metal structures are dropped and clusters left empty are removed, so the
    split boundaries land differently than cluster_aware_split(). Every entry point
    (train.py, evaluate.py, train_classifier.py) must call this rather than filtering
    the base split, or their train/val/test sets drift apart.
    """
    all_ids = [s for structs in cluster_to_structures.values() for s in structs]
    sm_set  = set(filter_single_metal(all_ids, cache_dir))

    sm_cluster_to_structs = {
        c: [s for s in structs if s in sm_set]
        for c, structs in cluster_to_structures.items()
    }
    sm_cluster_to_structs = {c: s for c, s in sm_cluster_to_structs.items() if s}
    sm_clusters = list(sm_cluster_to_structs.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(sm_clusters)
    n = len(sm_clusters)

    train_ids = [s for c in sm_clusters[:int(0.8*n)]           for s in sm_cluster_to_structs[c]]
    val_ids   = [s for c in sm_clusters[int(0.8*n):int(0.9*n)] for s in sm_cluster_to_structs[c]]
    test_ids  = [s for c in sm_clusters[int(0.9*n):]           for s in sm_cluster_to_structs[c]]

    return train_ids, val_ids, test_ids


def build_metal_resname_index(chunk_glob):
    """Returns {structure_id: set(metal_res_names)} by scanning parquet files."""
    index = {}
    for fpath in glob.glob(chunk_glob):
        df = pd.read_parquet(fpath, columns=['structure_id', 'res_name', 'is_metal'])
        metal_df = df[df['is_metal']]
        for sid, grp in metal_df.groupby('structure_id'):
            index[str(sid)] = set(grp['res_name'].str.strip().str.upper())
    return index


def filter_by_metal_resnames(ids, metal_index, allowed):
    """Keep only structures whose every metal res_name is in `allowed`."""
    allowed_upper = {r.upper() for r in allowed}
    return [sid for sid in ids
            if sid in metal_index and metal_index[sid].issubset(allowed_upper)]


class MetalBindingDataset(Dataset):
    def __init__(self, chunk_glob, structure_ids, k=10,
                 cache_dir='data/pt_cache_v3', esm_cache_dir=None):
        self.k             = k
        self.ids           = list(structure_ids)
        self.cache_dir     = cache_dir
        self.esm_cache_dir = esm_cache_dir

        self.index = {}
        for fpath in glob.glob(chunk_glob):
            df = pd.read_parquet(fpath, columns=['structure_id'])
            for sid in df['structure_id'].unique():
                self.index[sid] = fpath

    def __len__(self):
        return len(self.ids)

    def _process(self, idx):
        sid   = self.ids[idx]
        fpath = self.index[sid]
        df    = pd.read_parquet(fpath)
        df    = df[df['structure_id'] == sid]

        protein = df[~df['is_metal']].drop(columns='is_metal').reset_index(drop=True)
        metals  = df[ df['is_metal']].drop(columns='is_metal').reset_index(drop=True)

        coords14_raw = protein[_ATOM14_COLS].values.reshape(-1, 14, 3)

        valid        = np.isfinite(coords14_raw[:, 1, :]).all(axis=1)
        coords14_raw = coords14_raw[valid]

        pos_distal = torch.tensor(_compute_distal(coords14_raw), dtype=torch.float)
        coords14   = np.nan_to_num(coords14_raw, nan=0.0)

        res_names = protein['res_name'].values[valid]
        res_type  = torch.tensor(
            [RES_TO_IDX.get(r, RES_TO_IDX['UNK']) for r in res_names],
            dtype=torch.long,
        )

        pos = torch.tensor(coords14[:, 1, :], dtype=torch.float)
        x   = torch.tensor(coords14.reshape(len(coords14), -1), dtype=torch.float)

        ca   = coords14[:, 1, :]
        tree = KDTree(ca)
        _, neighbors = tree.query(ca, k=self.k + 1)
        neighbors = neighbors[:, 1:]

        src = np.repeat(np.arange(len(coords14)), self.k)
        dst = neighbors.flatten()

        ca_src  = ca[src];              ca_dst  = ca[dst]
        dis_np  = pos_distal.numpy()
        dis_src = dis_np[src];          dis_dst = dis_np[dst]

        d_ca_ca   = np.linalg.norm(ca_src  - ca_dst,  axis=1)
        d_dis_dis = np.linalg.norm(dis_src - dis_dst, axis=1)
        d_cross   = (np.linalg.norm(ca_src  - dis_dst, axis=1)
                   + np.linalg.norm(dis_src - ca_dst,  axis=1)) / 2

        v_src = dis_src - ca_src;  v_dst = dis_dst - ca_dst
        n_src = np.linalg.norm(v_src, axis=1, keepdims=True).clip(1e-8)
        n_dst = np.linalg.norm(v_dst, axis=1, keepdims=True).clip(1e-8)
        sc_dot = ((v_src / n_src) * (v_dst / n_dst)).sum(axis=1)

        edge_attr  = np.stack([d_ca_ca, d_dis_dis, d_cross, sc_dot], axis=1)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        y = torch.tensor(metals[['x0', 'y0', 'z0']].values, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=y,
            num_metals=torch.tensor([y.shape[0]], dtype=torch.long),
            res_type=res_type,
            pos=pos,
            pos_distal=pos_distal,
        )

    def _attach_distal(self, d, sid):
        if hasattr(d, 'pos_distal'):
            return d
        distal_path = os.path.join(DISTAL_CACHE_DIR, f'{sid}.pt')
        if os.path.exists(distal_path):
            try:
                pd_ = torch.load(distal_path, weights_only=True)
                if (pd_ is not None and isinstance(pd_, torch.Tensor)
                        and pd_.shape[0] == d.pos.shape[0]):
                    d.pos_distal = pd_
                    return d
            except Exception:
                pass
        d.pos_distal = d.pos  # last-resort: CA
        return d

    def _attach_esm(self, d, sid):
        if self.esm_cache_dir is None:
            return d
        esm_path = os.path.join(self.esm_cache_dir, f'{sid}.pt')
        if os.path.exists(esm_path):
            emb = torch.load(esm_path, weights_only=True)
            if emb.shape[0] == d.x.shape[0]:
                d.esm = emb
        return d

    def _attach_closest_residue(self, d):
        if hasattr(d, 'closest_residue_idx'):
            return d
        metal_pos = d.y[0].numpy()                                         # (3,)
        dists = np.linalg.norm(d.pos_distal.numpy() - metal_pos, axis=1)  # (N,)
        d.closest_residue_idx = torch.tensor(int(dists.argmin()), dtype=torch.long)
        return d

    def __getitem__(self, idx):
        sid        = self.ids[idx]
        cache_path = os.path.join(self.cache_dir, f'{sid}.pt')
        if os.path.exists(cache_path):
            d = torch.load(cache_path, weights_only=False)
            d.num_metals = torch.tensor([d.y.shape[0]], dtype=torch.long)
            d = self._attach_distal(d, sid)
        else:
            d = self._attach_distal(self._process(idx), sid)
        d = self._attach_closest_residue(d)
        return self._attach_esm(d, sid)

    def build_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        for i, sid in enumerate(self.ids):
            cache_path = os.path.join(self.cache_dir, f'{sid}.pt')
            if os.path.exists(cache_path):
                continue
            try:
                torch.save(self._process(i), cache_path)
            except Exception:
                continue
            if i % 500 == 0:
                print(f'{i}/{len(self.ids)}')
