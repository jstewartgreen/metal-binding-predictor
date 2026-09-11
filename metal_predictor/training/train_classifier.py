import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from metal_predictor.data.augmentation import AugmentedDataset
from metal_predictor.data.dataset import MetalBindingDataset
from metal_predictor.data.sampler import BucketBatchSampler
from metal_predictor.models.model import MetalPredictionModel

RUNS_CSV = 'data/runs_clf.csv'


def log_clf_run(phase, checkpoint, attn_mode=None, metal_filter=None,
                use_plm=None, batch_size=None, num_epochs=None, lr=None,
                k=None, train_size=None, val_size=None,
                best_epoch=None, best_val_acc=None,
                test_size=None, test_acc=None):
    row = dict(
        checkpoint=os.path.basename(checkpoint),
        attn_mode=attn_mode, metal_filter=metal_filter,
        use_plm=use_plm, batch_size=batch_size,
        num_epochs=num_epochs, lr=lr, k=k,
        train_size=train_size, val_size=val_size,
        best_epoch=best_epoch, best_val_acc=best_val_acc,
        test_size=test_size, test_acc=test_acc,
        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )
    row_s = {k: ('None' if v is None else v) for k, v in row.items()}

    if os.path.exists(RUNS_CSV):
        df   = pd.read_csv(RUNS_CSV)
        mask = df['checkpoint'] == row_s['checkpoint']
        if mask.any():
            idx = df.index[mask][0]
            for col, val in row_s.items():
                if val != 'None':
                    df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([row_s])], ignore_index=True)
    else:
        df = pd.DataFrame([row_s])

    df.to_csv(RUNS_CSV, index=False)
    print(f'[log_clf_run] {phase} → {RUNS_CSV}  ({row["checkpoint"]})')


def clf_loss(weights, batch, closest_residue_idx):
    """
    BCE loss: train attention weights to peak on the closest residue.

    weights             : (N, 1) — per-node attention weights (post-normalization)
    batch               : (N,)   — graph index per node
    closest_residue_idx : (G,)   — graph-level label (index within each graph)
    """
    # Build (N,) binary target: 1.0 at the closest residue node, 0.0 elsewhere.
    # closest_residue_idx is a local index within each graph, so we offset by the
    # start of each graph's node block.
    num_graphs  = closest_residue_idx.shape[0]
    graph_start = torch.zeros(num_graphs, dtype=torch.long, device=batch.device)
    for g in range(1, num_graphs):
        graph_start[g] = (batch < g).sum()
    abs_idx    = graph_start + closest_residue_idx           # (G,) absolute node indices
    target     = torch.zeros(weights.shape[0], device=weights.device)
    target[abs_idx] = 1.0
    return F.binary_cross_entropy(weights.squeeze(1), target)


def run_classifier_training(
    train_ids, val_ids,
    chunk_glob, cache_dir,
    checkpoint, ckpt_full,
    attn_mode='sparsemax',
    metal_filter=None,
    use_plm=False,
    esm_cache_dir=None,
    esm_dim=480,
    batch_size=16,
    num_epochs=25,
    lr=1e-3,
    grad_clip=1.0,
    k_neighbors=16,
    resume=False,
    device=None,
):
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    esm_dir = esm_cache_dir if use_plm else None

    train_ds      = AugmentedDataset(chunk_glob, train_ids, cache_dir=cache_dir,
                                     esm_cache_dir=esm_dir, k=k_neighbors)
    val_ds        = MetalBindingDataset(chunk_glob, val_ids, cache_dir=cache_dir,
                                        esm_cache_dir=esm_dir, k=k_neighbors)
    train_sampler = BucketBatchSampler(train_ds, batch_size=batch_size)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    val_loader    = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # num_hypotheses=1 — classification uses a single attention head
    model = MetalPredictionModel(
        attn_mode=attn_mode,
        learn_temperature=True,
        use_plm=use_plm,
        esm_dim=esm_dim,
        num_hypotheses=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    start_epoch   = 1
    best_val_acc  = 0.0
    best_epoch    = 0

    if resume and os.path.exists(ckpt_full):
        ckpt = torch.load(ckpt_full, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        best_epoch   = ckpt.get('best_epoch', ckpt['epoch'])
        print(f'Resumed from epoch {ckpt["epoch"]} | best val acc: {best_val_acc:.1f}%')
    elif resume and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f'Loaded weights from {checkpoint} (no optimizer state)')
    else:
        nn.init.zeros_(model.head.attention_mlp[-1].weight)
        nn.init.zeros_(model.head.attention_mlp[-1].bias)
        print('Fresh start with zero-init attention head')

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch    = batch.to(device)
            optimizer.zero_grad()
            esm_feat = batch.esm.to(device) if (use_plm and hasattr(batch, 'esm')) else None
            _, weights = model(batch.x, batch.edge_index, batch.edge_attr,
                               batch.batch, batch.res_type, batch.pos, batch.pos_distal,
                               return_attn=True, esm=esm_feat)

            loss = clf_loss(weights.unsqueeze(1), batch.batch,
                            batch.closest_residue_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(device)
                esm_feat = batch.esm.to(device) if (use_plm and hasattr(batch, 'esm')) else None
                _, weights = model(batch.x, batch.edge_index, batch.edge_attr,
                                   batch.batch, batch.res_type, batch.pos, batch.pos_distal,
                                   return_attn=True, esm=esm_feat)

                # Argmax within each graph
                num_g = batch.batch.max().item() + 1
                for g in range(num_g):
                    mask    = batch.batch == g
                    pred    = weights[mask].argmax().item()
                    target  = batch.closest_residue_idx[g].item()
                    correct += int(pred == target)
                    total   += 1

        val_acc = 100.0 * correct / total if total > 0 else 0.0
        scheduler.step(-val_acc)  # scheduler minimizes; negate accuracy

        tau     = model.head.log_temperature.exp().item()
        flag    = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), checkpoint)
            torch.save({
                'epoch':        epoch,
                'model':        model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'scheduler':    scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_epoch':   best_epoch,
            }, ckpt_full)
            flag = '  *'

        print(f'Epoch {epoch:3d} | train {train_loss:.4f} | val acc {val_acc:.1f}%'
              f' | τ={tau:.3f} | {time.time()-t0:.1f}s{flag}')

    log_clf_run('train', checkpoint,
                attn_mode=attn_mode, metal_filter=metal_filter,
                use_plm=use_plm, batch_size=batch_size,
                num_epochs=num_epochs, lr=lr, k=k_neighbors,
                train_size=len(train_ids), val_size=len(val_ids),
                best_epoch=best_epoch, best_val_acc=round(best_val_acc, 2))
