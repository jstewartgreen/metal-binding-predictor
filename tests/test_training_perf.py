"""
Training performance benchmarks.

Three tests that progressively narrow down where per-epoch time growth comes from:

  test_synthetic_fwd_bwd   — pure model forward+backward, no DataLoader, no real data.
                             If growth appears here, it is in the model/autograd/MPS runtime.

  test_real_loader_loop    — full training loop inline (notebook-style) on a small ZN subset.
                             If growth appears here but not in test_synthetic, it involves the
                             DataLoader or dataset.

  test_run_training_wrapper — calls run_training() from the package with the same subset.
                              If growth appears here but not in test_real_loader_loop, the
                              package wrapper itself is the cause.

Run with:
    pytest tests/test_training_perf.py -s -v
"""

import time
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from metal_predictor.models.model import MetalPredictionModel
from metal_predictor.data.dataset import (
    MetalBindingDataset, cluster_aware_split,
    filter_single_metal, build_metal_resname_index, filter_by_metal_resnames,
)
from metal_predictor.data.augmentation import AugmentedDataset
from metal_predictor.training.train import run_training

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_GLOB = 'data/gemmi_atom14/chunk_*.parquet'
CACHE_DIR  = 'data/pt_cache_v3'
CLUSTERS   = 'data/clusters/clusters.parquet'

N_EPOCHS       = 8     # enough to see a trend
SYNTH_BATCH    = 16    # graphs per synthetic batch
SYNTH_N_NODES  = 200   # residues per synthetic graph
SYNTH_K        = 10    # edges per node
SYNTH_N_BATCHES = 20   # batches per synthetic epoch (≈ real 4285 / 16 / 13)
REAL_TRAIN_CAP  = 150  # structures to use from ZN train split
REAL_VAL_CAP    = 32

GROWTH_THRESHOLD = 2.0  # epoch N time must be < GROWTH_THRESHOLD × epoch 1 time


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture(scope='module')
def model_fresh(device):
    """Return a freshly initialised model on device."""
    m = MetalPredictionModel(attn_mode='softmax').to(device)
    return m


def _make_synth_batch(batch_size, n_nodes, k, device):
    """Build a single synthetic PyG Batch of batch_size graphs."""
    graphs = []
    for _ in range(batch_size):
        n = n_nodes
        e = n * k
        src = torch.arange(n).repeat_interleave(k)
        dst = torch.randint(n, (e,))
        graphs.append(Data(
            x          = torch.randn(n, 42),
            edge_index = torch.stack([src, dst]),
            edge_attr  = torch.randn(e, 4),
            res_type   = torch.randint(21, (n,)),
            pos        = torch.randn(n, 3),
            pos_distal = torch.randn(n, 3),
            y          = torch.randn(1, 3),
            num_metals = torch.tensor([1]),
        ))
    return Batch.from_data_list(graphs).to(device)


@pytest.fixture(scope='module')
def synth_batches(device):
    """Pre-built list of synthetic batches (re-used across epochs to save time)."""
    return [_make_synth_batch(SYNTH_BATCH, SYNTH_N_NODES, SYNTH_K, device)
            for _ in range(SYNTH_N_BATCHES)]


@pytest.fixture(scope='module')
def real_loaders(device):
    """Small ZN loaders capped for speed."""
    train_ids, val_ids, _, _ = cluster_aware_split(CLUSTERS)
    train_ids = filter_single_metal(train_ids, CACHE_DIR)
    val_ids   = filter_single_metal(val_ids,   CACHE_DIR)
    metal_index = build_metal_resname_index(CHUNK_GLOB)
    train_ids = filter_by_metal_resnames(train_ids, metal_index, {'ZN'})[:REAL_TRAIN_CAP]
    val_ids   = filter_by_metal_resnames(val_ids,   metal_index, {'ZN'})[:REAL_VAL_CAP]

    train_ds = AugmentedDataset(CHUNK_GLOB, train_ids, cache_dir=CACHE_DIR, k=10)
    val_ds   = MetalBindingDataset(CHUNK_GLOB, val_ids, cache_dir=CACHE_DIR, k=10)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ids, val_ids


# ── Helper ─────────────────────────────────────────────────────────────────────

def _sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def _report(label, times):
    print(f"\n{label}")
    print("  " + "  ".join(f"epoch {i+1}: {t:.1f}s" for i, t in enumerate(times)))
    ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
    print(f"  last/first ratio: {ratio:.2f}x  (threshold: {GROWTH_THRESHOLD}x)")


# ── Test 1: synthetic forward+backward — isolates model/autograd/MPS ──────────

def test_synthetic_fwd_bwd(synth_batches, device):
    """
    Pure model forward+backward on pre-built synthetic batches.
    No DataLoader, no real data, no val loop.
    If per-epoch time grows here, the issue is in the model, autograd, or MPS runtime.
    """
    model = MetalPredictionModel(attn_mode='softmax').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_times = []
    for epoch in range(N_EPOCHS):
        model.train()
        _sync(device)
        t0 = time.time()

        for batch in synth_batches:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr,
                         batch.batch, batch.res_type, batch.pos, batch.pos_distal)
            loss = F.mse_loss(pred, batch.y.view(-1, 3))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss.item()  # force sync each batch

        _sync(device)
        epoch_times.append(time.time() - t0)

    _report("test_synthetic_fwd_bwd", epoch_times)
    assert epoch_times[-1] < epoch_times[0] * GROWTH_THRESHOLD, (
        f"Synthetic epoch time grew {epoch_times[-1]/epoch_times[0]:.2f}x "
        f"(epoch1={epoch_times[0]:.1f}s → epoch{N_EPOCHS}={epoch_times[-1]:.1f}s)"
    )


# ── Test 2: real DataLoader, inline notebook-style loop ───────────────────────

def test_real_loader_loop(real_loaders, device):
    """
    Full training + validation loop, written inline exactly as in the notebook.
    Uses real ZN data (capped to REAL_TRAIN_CAP structures).
    If per-epoch time grows here but not in test_synthetic, the issue is in the
    DataLoader, dataset, or augmentation.
    """
    train_loader, val_loader, _, _ = real_loaders

    model = MetalPredictionModel(attn_mode='softmax').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6)

    epoch_times = []
    for epoch in range(N_EPOCHS):
        # ── train ──
        model.train()
        _sync(device)
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr,
                         batch.batch, batch.res_type, batch.pos, batch.pos_distal)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss.item()

        # ── val ──
        model.eval()
        val_sq = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr,
                             batch.batch, batch.res_type, batch.pos, batch.pos_distal)
                val_sq.append(torch.norm(pred - batch.y, dim=-1).pow(2))

        val_rmse = torch.cat(val_sq).mean().sqrt().item()
        scheduler.step(val_rmse)

        _sync(device)
        epoch_times.append(time.time() - t0)

    _report("test_real_loader_loop", epoch_times)
    assert epoch_times[-1] < epoch_times[0] * GROWTH_THRESHOLD, (
        f"Inline loop epoch time grew {epoch_times[-1]/epoch_times[0]:.2f}x "
        f"(epoch1={epoch_times[0]:.1f}s → epoch{N_EPOCHS}={epoch_times[-1]:.1f}s)"
    )


# ── Test 3: package run_training() wrapper ────────────────────────────────────

def test_run_training_wrapper(real_loaders, device, tmp_path, capsys):
    """
    Calls run_training() from the package on the same capped ZN subset.
    Parses per-epoch times from stdout to compare against test_real_loader_loop.
    If growth appears here but not in test_real_loader_loop, the package wrapper
    is the cause.
    """
    _, _, train_ids, val_ids = real_loaders
    ckpt = str(tmp_path / 'test_model.pt')

    # run_training prints epoch lines; capture them
    run_training(
        train_ids=train_ids,
        val_ids=val_ids,
        chunk_glob=CHUNK_GLOB,
        cache_dir=CACHE_DIR,
        checkpoint=ckpt,
        ckpt_full=str(tmp_path / 'test_model_full.pt'),
        num_epochs=N_EPOCHS,
        device=device,
    )

    captured = capsys.readouterr().out
    epoch_times = []
    for line in captured.splitlines():
        # line format: "Epoch   N | ... | 35.4s [load=... fwd=... bwd=...]"
        if line.startswith('Epoch'):
            parts = line.split('|')
            time_part = parts[-1].strip()          # "35.4s [load=...]  *"
            secs = float(time_part.split('s')[0])
            epoch_times.append(secs)

    assert len(epoch_times) == N_EPOCHS, (
        f"Expected {N_EPOCHS} epoch lines, got {len(epoch_times)}.\nCaptured:\n{captured}"
    )

    _report("test_run_training_wrapper", epoch_times)
    assert epoch_times[-1] < epoch_times[0] * GROWTH_THRESHOLD, (
        f"run_training epoch time grew {epoch_times[-1]/epoch_times[0]:.2f}x "
        f"(epoch1={epoch_times[0]:.1f}s → epoch{N_EPOCHS}={epoch_times[-1]:.1f}s)"
    )
