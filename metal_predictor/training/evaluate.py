import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from metal_predictor.data.dataset import MetalBindingDataset
from metal_predictor.heuristics import donor_cluster_predict
from metal_predictor.models.model import MetalPredictionModel
from metal_predictor.training.train import log_run


def _metrics(errors):
    errors = torch.tensor(errors)
    return dict(
        rmse   =errors.pow(2).mean().sqrt().item(),
        mean   =errors.mean().item(),
        median =errors.median().item(),
        pct_2a =(errors < 2).float().mean().item() * 100,
        pct_5a =(errors < 5).float().mean().item() * 100,
    )


def _report(label, m):
    print(f'-- {label} --')
    print(f'RMSE:         {m["rmse"]:.2f} Å')
    print(f'Mean error:   {m["mean"]:.2f} Å')
    print(f'Median error: {m["median"]:.2f} Å')
    print(f'Error < 2 Å:  {m["pct_2a"]:.1f}%')
    print(f'Error < 5 Å:  {m["pct_5a"]:.1f}%')


def _log_test(name, m, test_size, **extra):
    log_run('test', name, test_size=test_size,
            test_rmse=round(m['rmse'],     4),
            test_mean=round(m['mean'],     4),
            test_median=round(m['median'], 4),
            test_pct_2a=round(m['pct_2a'], 2),
            test_pct_5a=round(m['pct_5a'], 2), **extra)


def run_evaluation(
    test_ids,
    chunk_glob, cache_dir, checkpoint=None,
    attn_mode='softmax',
    single_metal_only=True,
    metal_filter=None,
    num_hypotheses=1,
    num_mpnn_rounds=3,
    use_plm=False,
    esm_cache_dir=None,
    esm_dim=480,
    batch_size=16,
    device=None,
    log=True,
    baseline=False,
    coord_frame='absolute',
    donor_radius=4.0,
):
    """
    Evaluate a checkpoint on test_ids, and/or the model-free baselines.
    Returns a dict with keys: rmse, mean, median, pct_2a, pct_5a — the model's
    metrics, or the donor-cluster heuristic's when checkpoint is None.
    The CA-centroid baseline is always reported and logged as test_centroid_rmse.
    baseline=True also reports the donor-cluster heuristic (single-metal structures only).
    """
    assert checkpoint is not None or baseline, 'need a checkpoint or baseline=True'
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    esm_dir  = esm_cache_dir if use_plm else None
    test_ds  = MetalBindingDataset(chunk_glob, test_ids, cache_dir=cache_dir,
                                   esm_cache_dir=esm_dir, coord_frame=coord_frame)
    loader   = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = None
    if checkpoint is not None:
        sd = torch.load(checkpoint, weights_only=True)
        num_mpnn_rounds = len({k.split('.')[1] for k in sd if k.startswith('layers.')})

        # Infer use_node_features from the attention MLP input dim in the checkpoint.
        # With use_plm=True and node_dim=64: node_features → attn_in_dim=128, else 64.
        attn_w = next((v for k, v in sd.items() if 'attention_mlp' in k and k.endswith('.0.weight')), None)
        if attn_w is not None and use_plm:
            use_node_features = attn_w.shape[1] > 64
        else:
            use_node_features = num_mpnn_rounds > 0

        model = MetalPredictionModel(
            attn_mode=attn_mode, use_plm=use_plm,
            esm_dim=esm_dim, num_hypotheses=num_hypotheses,
            num_mpnn_rounds=num_mpnn_rounds,
            use_node_features=use_node_features,
        ).to(device)
        model.load_state_dict(sd)
        model.eval()

    is_filtered = single_metal_only or metal_filter is not None
    errors, centroid_errors, donor_errors, n_fallback = [], [], [], 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if model is not None:
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
                errors.extend(dists.cpu().tolist())

            c       = global_mean_pool(batch.pos, batch.batch)
            y_split = torch.split(batch.y, batch.num_metals.tolist())
            centroid_errors.extend(torch.stack([
                torch.norm(c[i] - metals, dim=-1).min() for i, metals in enumerate(y_split)
            ]).cpu().tolist())

            if baseline:
                dc, fb = donor_cluster_predict(batch, donor_radius, coord_frame)
                donor_errors.extend(torch.norm(dc - batch.y, dim=-1).cpu().tolist())
                n_fallback += int(fb.sum().item())

    print(f'Test set:     {len(test_ids)} structures  |  filter: {metal_filter}  |  hypotheses: {num_hypotheses}')

    centroid = _metrics(centroid_errors)
    _report('CA centroid baseline', centroid)

    donor = None
    if baseline:
        donor = _metrics(donor_errors)
        _report(f'Donor cluster baseline (r={donor_radius:g} Å, {n_fallback} fallbacks)', donor)
        if log:
            _log_test(f'heuristic_donor_r{donor_radius:g}_{metal_filter or "sm"}', donor, len(test_ids),
                      attn_mode='heuristic', metal_filter=metal_filter, num_hypotheses=1)

    if model is None:
        return donor

    metrics = _metrics(errors)
    print(f'Checkpoint:   {checkpoint}')
    _report('Model', metrics)
    if log:
        _log_test(checkpoint, metrics, len(test_ids), test_centroid_rmse=round(centroid['rmse'], 4))
    return metrics
