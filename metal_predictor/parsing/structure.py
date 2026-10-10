"""
Raw structure → model-ready graph, without the data/ cache.

Reproduces the training pipeline for a single file: the gemmi atom14 extraction from
notebooks/parse_cifs.ipynb, then MetalBindingDataset._process for the graph (via a
one-structure temporary parquet, so no graph-construction logic is duplicated), and the
on-the-fly ESM-2 embedding recipe from the model.ipynb cache builder.
"""
import os
import tempfile

import gemmi
import numpy as np
import pandas as pd
import requests
import torch

from metal_predictor.constants import IDX_TO_RES, _ATOM14_COLS
from metal_predictor.data.dataset import MetalBindingDataset

# Same slot layout the parquet cache was built with (notebooks/parse_cifs.ipynb).
ATOM14_NAMES = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
}
# Elements treated as metal ions. Training used the first eight; Ca/Cd/V/W were patched in later.
METAL_ELEMENTS = {'Zn', 'Co', 'Mg', 'Fe', 'Cu', 'Mn', 'Mo', 'Ni', 'Ca', 'Cd', 'V', 'W'}
THREE_TO_ONE   = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'UNK': 'X',
}
COLS        = ['structure_id', 'chain', 'res_name', 'res_num', 'is_metal'] + _ATOM14_COLS
RCSB_CIF    = 'https://files.rcsb.org/download/{}.cif'
ESM_LAYER   = 12   # last layer of esm2_t12_35M_UR50D


def fetch_cif(pdb_id, dest_dir):
    """Download <ID>.cif from RCSB into dest_dir and return its path."""
    pdb_id = pdb_id.upper()
    path   = os.path.join(dest_dir, f'{pdb_id}.cif')
    if not os.path.exists(path):
        r = requests.get(RCSB_CIF.format(pdb_id), timeout=60)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
    return path


def parse_structure(path, structure_id=None):
    """
    gemmi atom14 extraction, identical to the parquet builder: first model only, waters and
    non-standard residues dropped, any altloc accepted, missing atoms NaN. Metal ions are kept
    as is_metal rows (they never become graph nodes; _process routes them to the target y).
    """
    structure_id = structure_id or os.path.basename(path)
    model = gemmi.read_structure(path)[0]
    rows  = []
    for chain in model:
        for residue in chain:
            info     = gemmi.find_tabulated_residue(residue.name)
            is_metal = len(residue) > 0 and residue[0].element.name in METAL_ELEMENTS
            if info is not None and info.is_water():
                continue
            if not (info is not None and info.is_amino_acid()) and not is_metal:
                continue
            coords = [np.nan] * 42
            if is_metal:
                a = residue[0]
                coords[0], coords[1], coords[2] = a.pos.x, a.pos.y, a.pos.z
            else:
                atom14 = ATOM14_NAMES.get(residue.name)
                if atom14 is None:
                    continue
                for slot, name in enumerate(atom14):
                    if not name:
                        continue
                    a = residue.find_atom(name, '*')
                    if a:
                        coords[slot * 3:slot * 3 + 3] = [a.pos.x, a.pos.y, a.pos.z]
            rows.append([structure_id, chain.name, residue.name, str(residue.seqid), is_metal] + coords)
    return pd.DataFrame(rows, columns=COLS)


def structure_to_graph(df, k=10, coord_frame='absolute'):
    """
    Build the model's graph from a parsed atom14 table by running MetalBindingDataset._process
    on a one-structure temporary parquet — the exact code path the training cache came from.
    k=10 matches the cached graphs every checkpoint was trained on.
    """
    sid = df['structure_id'].iloc[0]
    with tempfile.TemporaryDirectory() as tmp:
        df.to_parquet(os.path.join(tmp, 'chunk_0.parquet'))
        ds = MetalBindingDataset(os.path.join(tmp, 'chunk_*.parquet'), [sid], k=k,
                                 cache_dir=os.path.join(tmp, 'cache'), coord_frame=coord_frame)
        return ds[0]


_esm = {}


def esm_embeddings(res_type, device):
    """
    (N, 480) ESM-2 embeddings for the graph's residues, same recipe as the training cache:
    esm2_t12_35M_UR50D, layer 12, BOS/EOS stripped, unknown residues as X. Weights download
    to the torch hub cache on first use.
    """
    import esm as esm_pkg
    if 'model' not in _esm:
        model, alphabet = esm_pkg.pretrained.esm2_t12_35M_UR50D()
        _esm['model'], _esm['convert'] = model.eval().to(device), alphabet.get_batch_converter()
    seq = ''.join(THREE_TO_ONE.get(IDX_TO_RES[int(t)], 'X') for t in res_type)
    _, _, tokens = _esm['convert']([('query', seq)])
    with torch.no_grad():
        out = _esm['model'](tokens.to(device), repr_layers=[ESM_LAYER], return_contacts=False)
    return out['representations'][ESM_LAYER][0, 1:len(seq) + 1, :].float()
