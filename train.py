"""
CLI entry point for training the MetalPredictionModel.

Usage:
    python train.py [options]

Examples:
    python train.py --epochs 25 --single-metal-only --attn-mode softmax
    python train.py --metal-filter ZN --attn-mode sparsemax --use-plm
    python train.py --resume --checkpoint data/best_model_v9_zn_softmax.pt
"""
import argparse
import torch

from metal_predictor.constants import BARE_METAL_RESNAMES
from metal_predictor.data.dataset import (
    build_metal_resname_index,
    cluster_aware_split,
    filter_by_metal_resnames,
    single_metal_resplit,
)
from metal_predictor.training.train import run_training

CHUNK_GLOB = 'data/gemmi_atom14/chunk_*.parquet'
CACHE_DIR  = 'data/pt_cache_v3'
CLUSTERS   = 'data/clusters/clusters.parquet'


def build_checkpoint_name(args):
    if args.metal_filter == 'ZN':
        filter_tag = '_zn'
    elif args.metal_filter == 'bare':
        filter_tag = '_bm'
    elif args.single_metal_only:
        filter_tag = '_sm'
    else:
        filter_tag = ''
    plm_tag  = '_plm' if args.use_plm else ''
    mpnn_tag = f'_m{args.num_mpnn_rounds}' if args.num_mpnn_rounds != 3 else ''
    eps_tag  = f'_e{args.eps_wta}' if args.eps_wta > 0.0 else ''
    wta_tag  = f'_wta{args.num_hypotheses}' if args.num_hypotheses > 1 else ''
    frame_tag = {'residue': '_rel', 'centroid': '_cen'}.get(args.coord_frame, '')
    return f'data/best_model_v9{filter_tag}_{args.attn_mode}{plm_tag}{mpnn_tag}{eps_tag}{wta_tag}{frame_tag}.pt'


def main():
    parser = argparse.ArgumentParser(description='Train MetalPredictionModel')

    # Data
    parser.add_argument('--single-metal-only', action='store_true', default=True,
                        help='Train on structures with exactly 1 metal (default: True)')
    parser.add_argument('--all-metals', dest='single_metal_only', action='store_false',
                        help='Train on all structures with >= 1 metal')
    parser.add_argument('--metal-filter', default=None, choices=[None, 'bare', 'ZN'],
                        help='Filter training set: None | bare | ZN')
    parser.add_argument('--k-neighbors', type=int, default=16,
                        help='k for k-NN graph construction (default: 16)')
    parser.add_argument('--coord-frame', default='absolute',
                        choices=['absolute', 'residue', 'centroid'],
                        help='Frame for atom14 node coords: absolute crystal frame, relative to each '
                             'residue CA, or relative to the CA centroid (default: absolute)')
    parser.add_argument('--relative-coords', action='store_const', const='residue',
                        dest='coord_frame', help=argparse.SUPPRESS)   # alias for --coord-frame residue

    # Model
    parser.add_argument('--attn-mode', default='softmax',
                        choices=['softmax', 'sparsemax', 'topk'],
                        help='Attention mechanism (default: softmax)')
    parser.add_argument('--num-hypotheses', type=int, default=1,
                        help='Number of WTA attention heads (default: 1)')
    parser.add_argument('--num-mpnn-rounds', type=int, default=3,
                        help='MPNN message-passing rounds; 0 = GNN ablation (default: 3)')
    parser.add_argument('--diversity-weight', type=float, default=0.05,
                        help='Diversity penalty weight for WTA loss (default: 0.05)')
    parser.add_argument('--eps-wta', type=float, default=0.0,
                        help='ε-WTA exploration rate: prob of training a random non-winner (default: 0.0)')
    parser.add_argument('--learn-temperature', action='store_true', default=True,
                        help='Learn attention temperature τ (default: True)')
    parser.add_argument('--fixed-temperature', dest='learn_temperature', action='store_false')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Initial attention temperature τ (default: 1.0)')
    parser.add_argument('--use-plm', action='store_true', default=False,
                        help='Inject ESM-2 embeddings into attention head')
    parser.add_argument('--esm-cache-dir', default='data/esm2_cache')
    parser.add_argument('--esm-dim', type=int, default=480)

    # Training
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint path (auto-generated from flags if not set)')

    args = parser.parse_args()

    checkpoint = args.checkpoint or build_checkpoint_name(args)
    ckpt_full  = checkpoint.replace('.pt', '_full.pt')

    device = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    # ── Build splits ──────────────────────────────────────────────────────────
    print('Loading cluster-aware splits...')
    train_ids, val_ids, test_ids, cluster_to_structures = cluster_aware_split(CLUSTERS)

    if args.metal_filter in ('ZN', 'bare') or args.single_metal_only:
        print('Filtering to single-metal structures...')
        train_ids, val_ids, test_ids = single_metal_resplit(
            cluster_to_structures, CACHE_DIR)

    if args.metal_filter in ('ZN', 'bare'):
        print(f'Building metal res_name index for filter={args.metal_filter}...')
        metal_index = build_metal_resname_index(CHUNK_GLOB)
        allowed = {'ZN'} if args.metal_filter == 'ZN' else BARE_METAL_RESNAMES
        train_ids = filter_by_metal_resnames(train_ids, metal_index, allowed)
        val_ids   = filter_by_metal_resnames(val_ids,   metal_index, allowed)

    print(f'Split: train={len(train_ids)} val={len(val_ids)}')
    print(f'Config: attn={args.attn_mode} | filter={args.metal_filter} | '
          f'hypotheses={args.num_hypotheses} | plm={args.use_plm}')
    print(f'Checkpoint: {checkpoint}')

    run_training(
        train_ids=train_ids,
        val_ids=val_ids,
        chunk_glob=CHUNK_GLOB,
        cache_dir=CACHE_DIR,
        checkpoint=checkpoint,
        ckpt_full=ckpt_full,
        attn_mode=args.attn_mode,
        single_metal_only=args.single_metal_only,
        metal_filter=args.metal_filter,
        num_hypotheses=args.num_hypotheses,
        num_mpnn_rounds=args.num_mpnn_rounds,
        diversity_weight=args.diversity_weight,
        eps_wta=args.eps_wta,
        learn_temperature=args.learn_temperature,
        temperature=args.temperature,
        use_plm=args.use_plm,
        esm_cache_dir=args.esm_cache_dir,
        esm_dim=args.esm_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        k_neighbors=args.k_neighbors,
        coord_frame=args.coord_frame,
        resume=args.resume,
        device=device,
    )


if __name__ == '__main__':
    main()
