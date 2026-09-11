import os
import random

import torch
from torch.utils.data import Sampler

from metal_predictor.constants import DISTAL_CACHE_DIR


def get_num_nodes(dataset, idx):
    """Return the number of residues for dataset[idx] without full __getitem__."""
    sid = dataset.ids[idx]
    cache_path = os.path.join(dataset.cache_dir, f'{sid}.pt')
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        return d.x.shape[0]
    # Fallback: load via dataset (slow, shouldn't happen if cache is warm)
    return dataset[idx].x.shape[0]


class BucketBatchSampler(Sampler):
    """
    Groups structures into fixed-size buckets by residue count, then shuffles
    bucket order and within-bucket order each epoch.

    On MPS, Metal kernels are compiled per unique tensor shape. With random
    shuffling, each epoch generates ~N unique batch shapes (because variable-
    length proteins combine differently), forcing a full recompilation every
    epoch. Bucket batching ensures batch shapes repeat across epochs so MPS
    can reuse compiled kernels, giving consistent per-epoch times.

    bucket_size: number of structures per bucket (larger → more shape variety
                 within a bucket; smaller → less shuffle diversity but faster).
                 Default 256 works well for ~4k ZN structures with batch_size=16.
    """
    def __init__(self, dataset, batch_size, bucket_size=256, drop_last=False):
        self.batch_size  = batch_size
        self.drop_last   = drop_last

        # Sort indices by protein length once at construction
        sizes = [get_num_nodes(dataset, i) for i in range(len(dataset))]
        self._sorted_indices = sorted(range(len(dataset)), key=lambda i: sizes[i])
        self._bucket_size    = bucket_size

    def __iter__(self):
        # Shuffle within each bucket
        indices = list(self._sorted_indices)
        buckets = [indices[i:i + self._bucket_size]
                   for i in range(0, len(indices), self._bucket_size)]
        for b in buckets:
            random.shuffle(b)

        # Flatten back, form batches, shuffle batch order
        flat = [idx for bucket in buckets for idx in bucket]
        batches = [flat[i:i + self.batch_size]
                   for i in range(0, len(flat), self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        n = len(self._sorted_indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
