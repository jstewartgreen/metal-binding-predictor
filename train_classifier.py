"""
CLI entry point for training the closest-residue classification model.

Usage:
    python train_classifier.py --attn-mode sparsemax --metal-filter ZN
    python train_classifier.py --attn-mode sparsemax --metal-filter ZN --use-plm
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
from metal_predictor.training.train_classifier import run_classifier_training

CHUNK_GLOB = 'data/gemmi_atom14/chunk_*.parquet'
CACHE_DIR  = 'data/pt_cache_v3'
CLUSTERS   = 'data/clusters/clusters.parquet'


def build_checkpoint_name(args):
    if args.metal_filter == 'ZN':
        filter_tag = '_zn'
    elif args.metal_filter == 'bare':
        filter_tag = '_bm'
    else:
        filter_tag = '_sm'
    plm_tag = '_plm' if args.use_plm else ''
    return f'data/best_model_v9{filter_tag}_{args.attn_mode}{plm_tag}_clf.pt'


def main():
    parser = argparse.ArgumentParser(description='Train closest-residue classifier')

    parser.add_argument('--metal-filter', default='ZN', choices=['bare', 'ZN'])
    parser.add_argument('--k-neighbors', type=int, default=16)
    parser.add_argument('--attn-mode', default='sparsemax',
                        choices=['softmax', 'sparsemax', 'topk'])
    parser.add_argument('--use-plm', action='store_true', default=False)
    parser.add_argument('--esm-cache-dir', default='data/esm2_cache')
    parser.add_argument('--esm-dim', type=int, default=480)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--checkpoint', default=None)

    args = parser.parse_args()

    checkpoint = args.checkpoint or build_checkpoint_name(args)
    ckpt_full  = checkpoint.replace('.pt', '_full.pt')

    device = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    print('Loading cluster-aware splits...')
    train_ids, val_ids, test_ids, cluster_to_structures = cluster_aware_split(CLUSTERS)

    print('Filtering to single-metal structures...')
    train_ids, val_ids, _ = single_metal_resplit(cluster_to_structures, CACHE_DIR)

    print(f'Building metal res_name index for filter={args.metal_filter}...')
    metal_index = build_metal_resname_index(CHUNK_GLOB)
    allowed = {'ZN'} if args.metal_filter == 'ZN' else BARE_METAL_RESNAMES
    train_ids = filter_by_metal_resnames(train_ids, metal_index, allowed)
    val_ids   = filter_by_metal_resnames(val_ids,   metal_index, allowed)

    print(f'Split: train={len(train_ids)} val={len(val_ids)}')
    print(f'Config: attn={args.attn_mode} | filter={args.metal_filter} | plm={args.use_plm}')
    print(f'Checkpoint: {checkpoint}')

    run_classifier_training(
        train_ids=train_ids,
        val_ids=val_ids,
        chunk_glob=CHUNK_GLOB,
        cache_dir=CACHE_DIR,
        checkpoint=checkpoint,
        ckpt_full=ckpt_full,
        attn_mode=args.attn_mode,
        metal_filter=args.metal_filter,
        use_plm=args.use_plm,
        esm_cache_dir=args.esm_cache_dir,
        esm_dim=args.esm_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        k_neighbors=args.k_neighbors,
        resume=args.resume,
        device=device,
    )


if __name__ == '__main__':
    main()
