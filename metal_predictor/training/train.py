import os
import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from metal_predictor.data.augmentation import AugmentedDataset
from metal_predictor.data.dataset import MetalBindingDataset
from metal_predictor.models.model import MetalPredictionModel

RUNS_CSV = 'data/runs.csv'


def wta_loss(preds, targets, diversity_weight=0.05):
    """
    Winner-takes-all loss for multi-hypothesis mode.

    preds  : (N, K, 3) — K hypotheses per structure
    targets: (N, 3)    — ground-truth metal position (single-metal mode)

    Returns WTA MSE (best hypothesis) + diversity regularization to prevent mode collapse.
    """
    sq_dists = ((preds - targets.unsqueeze(1)) ** 2).sum(-1)   # (N, K)
    wta = sq_dists.min(dim=1).values.mean()
    K   = preds.shape[1]
    div = sum(
        torch.exp(-(preds[:, i] - preds[:, j]).norm(dim=-1)).mean()
        for i in range(K) for j in range(i + 1, K)
    ) / (K * (K - 1) / 2)
    return wta + diversity_weight * div


def log_run(phase, checkpoint,
            attn_mode=None, metal_filter=None, num_hypotheses=None,
            diversity_weight=None, learn_temperature=None, use_plm=None,
            batch_size=None, num_epochs=None, lr=None, grad_clip=None, k=None,
            train_size=None, val_size=None,
            best_epoch=None, best_train_loss=None, best_val_rmse=None,
            test_size=None, test_rmse=None, test_mean=None, test_median=None,
            test_pct_2a=None, test_pct_5a=None):
    """
    Append or update a row in data/runs.csv keyed by checkpoint filename.
    Call with phase='train' at end of training, phase='test' at end of evaluation.
    """
    row = dict(
        checkpoint=os.path.basename(checkpoint),
        attn_mode=attn_mode, metal_filter=metal_filter,
        num_hypotheses=num_hypotheses, diversity_weight=diversity_weight,
        learn_temperature=learn_temperature, use_plm=use_plm,
        batch_size=batch_size, num_epochs=num_epochs, lr=lr,
        grad_clip=grad_clip, k=k,
        train_size=train_size, val_size=val_size,
        best_epoch=best_epoch, best_train_loss=best_train_loss,
        best_val_rmse=best_val_rmse,
        test_size=test_size, test_rmse=test_rmse,
        test_mean=test_mean, test_median=test_median,
        test_pct_2a=test_pct_2a, test_pct_5a=test_pct_5a,
        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )

    if os.path.exists(RUNS_CSV):
        df   = pd.read_csv(RUNS_CSV)
        mask = df['checkpoint'] == row['checkpoint']
        if mask.any():
            idx = df.index[mask][0]
            for col, val in row.items():
                if val is not None:
                    df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(RUNS_CSV, index=False)
    print(f'[log_run] {phase} → {RUNS_CSV}  ({row["checkpoint"]})')


def run_training(
    train_ids, val_ids,
    chunk_glob, cache_dir,
    checkpoint, ckpt_full,
    attn_mode='softmax',
    single_metal_only=True,
    metal_filter=None,
    num_hypotheses=1,
    diversity_weight=0.05,
    learn_temperature=True,
    temperature=1.0,
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

    train_ds    = AugmentedDataset(chunk_glob, train_ids, cache_dir=cache_dir,
                                   esm_cache_dir=esm_dir, k=k_neighbors)
    val_ds      = MetalBindingDataset(chunk_glob, val_ids, cache_dir=cache_dir,
                                      esm_cache_dir=esm_dir, k=k_neighbors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = MetalPredictionModel(
        attn_mode=attn_mode,
        learn_temperature=learn_temperature,
        temperature=temperature,
        use_plm=use_plm,
        esm_dim=esm_dim,
        num_hypotheses=num_hypotheses,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    start_epoch          = 1
    best_val_loss        = float('inf')
    best_epoch           = 0
    best_train_loss_snap = float('inf')

    if resume and os.path.exists(ckpt_full):
        ckpt = torch.load(ckpt_full, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch          = ckpt['epoch'] + 1
        best_val_loss        = ckpt['best_val_loss']
        best_epoch           = ckpt.get('best_epoch', ckpt['epoch'])
        best_train_loss_snap = ckpt.get('best_train_loss_snap', float('inf'))
        print(f'Resumed from epoch {ckpt["epoch"]} | best val RMSE: {best_val_loss:.2f} Å')
    elif resume and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f'Loaded weights from {checkpoint} (no optimizer state)')
    else:
        # Zero-init final attention layer(s) → offset=0 at epoch 1 (centroid baseline)
        if num_hypotheses > 1:
            for mlp in model.head.attention_mlps:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
        else:
            nn.init.zeros_(model.head.attention_mlp[-1].weight)
            nn.init.zeros_(model.head.attention_mlp[-1].bias)
        print('Fresh start with zero-init attention head(s)')

    is_filtered = single_metal_only or metal_filter is not None

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch    = batch.to(device)
            optimizer.zero_grad()
            esm_feat = batch.esm.to(device) if (use_plm and hasattr(batch, 'esm')) else None
            pred_out = model(batch.x, batch.edge_index, batch.edge_attr,
                             batch.batch, batch.res_type, batch.pos, batch.pos_distal,
                             esm=esm_feat)

            if is_filtered:
                targets = batch.y
            else:
                y_split = torch.split(batch.y, batch.num_metals.tolist())
                targets = torch.stack([
                    metals[torch.randint(len(metals), (1,)).item()]
                    for metals in y_split
                ])

            if num_hypotheses > 1:
                loss = wta_loss(pred_out, targets, diversity_weight)
            else:
                loss = F.mse_loss(pred_out, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_sq_dists = []
        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(device)
                esm_feat = batch.esm.to(device) if (use_plm and hasattr(batch, 'esm')) else None
                pred_out = model(batch.x, batch.edge_index, batch.edge_attr,
                                 batch.batch, batch.res_type, batch.pos, batch.pos_distal,
                                 esm=esm_feat)

                if num_hypotheses > 1:
                    sq    = ((pred_out - batch.y.unsqueeze(1)) ** 2).sum(-1)
                    dists = sq.min(dim=1).values.sqrt()
                elif is_filtered:
                    dists = torch.norm(pred_out - batch.y, dim=-1)
                else:
                    y_split = torch.split(batch.y, batch.num_metals.tolist())
                    dists   = torch.stack([
                        torch.norm(pred_out[i] - metals, dim=-1).min()
                        for i, metals in enumerate(y_split)
                    ])
                val_sq_dists.append(dists.pow(2))

        val_rmse = torch.cat(val_sq_dists).mean().sqrt().item()
        scheduler.step(val_rmse)

        tau_str = ''
        if learn_temperature:
            tau     = model.head.log_temperature.exp().item()
            tau_str = f' | τ={tau:.3f}'

        flag = ''
        if val_rmse < best_val_loss:
            best_val_loss        = val_rmse
            best_epoch           = epoch
            best_train_loss_snap = train_loss
            torch.save(model.state_dict(), checkpoint)
            torch.save({
                'epoch':                epoch,
                'model':                model.state_dict(),
                'optimizer':            optimizer.state_dict(),
                'scheduler':            scheduler.state_dict(),
                'best_val_loss':        best_val_loss,
                'best_epoch':           best_epoch,
                'best_train_loss_snap': best_train_loss_snap,
            }, ckpt_full)
            flag = '  *'

        print(f'Epoch {epoch:3d} | train {train_loss:.4f} | val RMSE {val_rmse:.2f} Å'
              f'{tau_str} | {time.time()-t0:.1f}s{flag}')

    log_run('train', checkpoint,
            attn_mode=attn_mode, metal_filter=metal_filter,
            num_hypotheses=num_hypotheses,
            diversity_weight=diversity_weight if num_hypotheses > 1 else None,
            learn_temperature=learn_temperature, use_plm=use_plm,
            batch_size=batch_size, num_epochs=num_epochs, lr=lr,
            grad_clip=grad_clip, k=k_neighbors,
            train_size=len(train_ids), val_size=len(val_ids),
            best_epoch=best_epoch,
            best_train_loss=round(best_train_loss_snap, 4),
            best_val_rmse=round(best_val_loss, 4))

    return model, best_val_loss
