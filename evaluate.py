"""
CLI entry point for evaluating a trained MetalPredictionModel on the test set.

Usage:
    python evaluate.py --checkpoint data/best_model_v9_zn_softmax.pt --metal-filter ZN
    python evaluate.py --checkpoint data/best_model_v9_sm_softmax_plm.pt --use-plm
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
from metal_predictor.training.evaluate import run_evaluation

CHUNK_GLOB = 'data/gemmi_atom14/chunk_*.parquet'
CACHE_DIR  = 'data/pt_cache_v3'
CLUSTERS   = 'data/clusters/clusters.parquet'


def main():
    parser = argparse.ArgumentParser(description='Evaluate MetalPredictionModel on test set')

    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--single-metal-only', action='store_true', default=True)
    parser.add_argument('--all-metals', dest='single_metal_only', action='store_false')
    parser.add_argument('--metal-filter', default=None, choices=[None, 'bare', 'ZN'])
    parser.add_argument('--attn-mode', default='softmax',
                        choices=['softmax', 'sparsemax', 'topk'])
    parser.add_argument('--num-hypotheses', type=int, default=1)
    parser.add_argument('--num-mpnn-rounds', type=int, default=3,
                        help='MPNN message-passing rounds; 0 = GNN ablation (default: 3)')
    parser.add_argument('--use-plm', action='store_true', default=False)
    parser.add_argument('--esm-cache-dir', default='data/esm2_cache')
    parser.add_argument('--esm-dim', type=int, default=480)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--no-log', action='store_true', default=False,
                        help='Skip writing results to runs.csv')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Also compute naive CA-centroid prediction as a baseline')

    args = parser.parse_args()

    device = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    # ── Build test split ──────────────────────────────────────────────────────
    print('Loading cluster-aware splits...')
    train_ids, val_ids, test_ids, cluster_to_structures = cluster_aware_split(CLUSTERS)

    if args.metal_filter in ('ZN', 'bare') or args.single_metal_only:
        print('Filtering to single-metal structures...')
        _, _, test_ids = single_metal_resplit(cluster_to_structures, CACHE_DIR)

    if args.metal_filter in ('ZN', 'bare'):
        print(f'Building metal res_name index for filter={args.metal_filter}...')
        metal_index = build_metal_resname_index(CHUNK_GLOB)
        allowed  = {'ZN'} if args.metal_filter == 'ZN' else BARE_METAL_RESNAMES
        test_ids = filter_by_metal_resnames(test_ids, metal_index, allowed)

    print(f'Test set: {len(test_ids)} structures')

    run_evaluation(
        test_ids=test_ids,
        chunk_glob=CHUNK_GLOB,
        cache_dir=CACHE_DIR,
        checkpoint=args.checkpoint,
        attn_mode=args.attn_mode,
        single_metal_only=args.single_metal_only,
        metal_filter=args.metal_filter,
        num_hypotheses=args.num_hypotheses,
        num_mpnn_rounds=args.num_mpnn_rounds,
        use_plm=args.use_plm,
        esm_cache_dir=args.esm_cache_dir,
        esm_dim=args.esm_dim,
        batch_size=args.batch_size,
        device=device,
        log=not args.no_log,
        baseline=args.baseline,
    )


if __name__ == '__main__':
    main()
