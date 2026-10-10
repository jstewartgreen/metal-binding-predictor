"""
Predict the metal-binding site of a protein structure.

Usage:
    python predict.py 1ZAA                              # fetch from RCSB by PDB ID
    python predict.py path/to/structure.cif             # local CIF or PDB file
    python predict.py 1ZAA --heuristic                  # also print the donor-cluster baseline
    python predict.py 1ZAA --checkpoint data/best_model_v9_zn_sparsemax_rel.pt

Prints one line "x y z" (Å, in the input file's frame). Any metal ions present in the input
are excluded from the graph; if one is present, its deposited position and the prediction
error are reported on stderr. Model flags default to values inferred from the checkpoint
filename and state dict; pass them explicitly if the name does not follow the v9 scheme.
"""
import argparse
import os
import re
import sys
import tempfile

import torch

from metal_predictor.heuristics import donor_cluster_predict
from metal_predictor.inference import load_model
from metal_predictor.parsing.structure import esm_embeddings, fetch_cif, parse_structure, structure_to_graph

DEFAULT_CHECKPOINT = 'data/best_model_v9_zn_sparsemax_plm.pt'


def infer_from_checkpoint(path):
    """attn_mode / use_plm / coord_frame from the filename; mpnn rounds and head width from weights."""
    name = os.path.basename(path)
    attn = re.search(r'_(softmax|sparsemax|topk)', name)
    sd   = torch.load(path, weights_only=True, map_location='cpu')
    rounds = len({k.split('.')[1] for k in sd if k.startswith('layers.')})
    attn_w = next((v for k, v in sd.items() if 'attention_mlp' in k and k.endswith('.0.weight')), None)
    use_plm = 'head.esm_proj.weight' in sd
    return dict(
        attn_mode=attn.group(1) if attn else None,
        use_plm=use_plm,
        coord_frame='residue' if '_rel' in name else 'centroid' if '_cen' in name else 'absolute',
        num_mpnn_rounds=rounds,
        use_node_features=(attn_w.shape[1] > 64) if (attn_w is not None and use_plm) else rounds > 0,
    )


def main():
    p = argparse.ArgumentParser(description='Predict metal ion coordinates for a protein structure')
    p.add_argument('structure', help='Path to a .cif/.pdb file, or a 4-character PDB ID to fetch from RCSB')
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--attn-mode', choices=['softmax', 'sparsemax', 'topk'], help='Default: inferred from checkpoint name')
    p.add_argument('--use-plm', action='store_true', default=None, help='Default: inferred from checkpoint weights')
    p.add_argument('--coord-frame', choices=['absolute', 'residue', 'centroid'], help='Default: inferred from checkpoint name')
    p.add_argument('--heuristic', action='store_true', help='Also print the donor-cluster heuristic prediction')
    p.add_argument('--device', default=None, help='mps | cuda | cpu (default: auto)')
    args = p.parse_args()

    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else
                                          'cuda' if torch.cuda.is_available() else 'cpu'))
    cfg = infer_from_checkpoint(args.checkpoint)
    for key in ('attn_mode', 'use_plm', 'coord_frame'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg['attn_mode'] is None:
        p.error('--attn-mode could not be inferred from the checkpoint name; pass it explicitly')

    # ── structure → graph ───────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        path = args.structure
        if re.fullmatch(r'[0-9A-Za-z]{4}', path) and not os.path.exists(path):
            path = fetch_cif(path, tmp)
        df = parse_structure(path)
    if df.empty or not (~df['is_metal']).any():
        sys.exit(f'no standard amino-acid residues found in {args.structure}')
    graph = structure_to_graph(df, coord_frame=cfg['coord_frame'])
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    graph = graph.to(device)
    esm   = esm_embeddings(graph.res_type, device) if cfg['use_plm'] else None

    # ── model ───────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device, attn_mode=cfg['attn_mode'], use_plm=cfg['use_plm'],
                       num_mpnn_rounds=cfg['num_mpnn_rounds'], use_node_features=cfg['use_node_features'])
    with torch.no_grad():
        pred = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.res_type,
                     graph.pos, graph.pos_distal, esm=esm)[0]
    print(' '.join(f'{v:.3f}' for v in pred.tolist()))

    if args.heuristic:
        hpred, fallback = donor_cluster_predict(graph, coord_frame=cfg['coord_frame'])
        tag = '  (no donor pair within 4 Å; CA-centroid fallback)' if fallback[0] else ''
        print(' '.join(f'{v:.3f}' for v in hpred[0].tolist()) + tag)

    # ── report against any deposited metal (stderr, so stdout stays machine-readable) ──
    metals = df[df['is_metal']]
    if len(metals):
        y = torch.tensor(metals[['x0', 'y0', 'z0']].values, dtype=torch.float, device=device)
        err = torch.norm(y - pred, dim=-1).min().item()
        names = ', '.join(metals['res_name'].str.strip().tolist())
        print(f'[deposited metal: {names} at {" ".join(f"{v:.3f}" for v in y[0].tolist())}; '
              f'model error {err:.2f} Å'
              + (f'; heuristic error {torch.norm(y - hpred[0], dim=-1).min().item():.2f} Å' if args.heuristic else '')
              + ']', file=sys.stderr)


if __name__ == '__main__':
    main()
