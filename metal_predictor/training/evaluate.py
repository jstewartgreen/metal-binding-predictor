import torch
from torch_geometric.loader import DataLoader

from metal_predictor.data.dataset import MetalBindingDataset
from metal_predictor.models.model import MetalPredictionModel
from metal_predictor.training.train import log_run


def run_evaluation(
    test_ids,
    chunk_glob, cache_dir, checkpoint,
    attn_mode='softmax',
    single_metal_only=True,
    metal_filter=None,
    num_hypotheses=1,
    use_plm=False,
    esm_cache_dir=None,
    esm_dim=480,
    batch_size=16,
    device=None,
    log=True,
):
    """
    Load checkpoint and evaluate on test_ids.
    Returns a dict with keys: rmse, mean, median, pct_2a, pct_5a.
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    esm_dir  = esm_cache_dir if use_plm else None
    test_ds  = MetalBindingDataset(chunk_glob, test_ids, cache_dir=cache_dir,
                                   esm_cache_dir=esm_dir)
    loader   = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MetalPredictionModel(
        attn_mode=attn_mode, use_plm=use_plm,
        esm_dim=esm_dim, num_hypotheses=num_hypotheses,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    is_filtered = single_metal_only or metal_filter is not None
    errors = []

    with torch.no_grad():
        for batch in loader:
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
            errors.extend(dists.cpu().tolist())

    errors = torch.tensor(errors)
    metrics = dict(
        rmse   =errors.pow(2).mean().sqrt().item(),
        mean   =errors.mean().item(),
        median =errors.median().item(),
        pct_2a =(errors < 2).float().mean().item() * 100,
        pct_5a =(errors < 5).float().mean().item() * 100,
    )

    print(f'Checkpoint:   {checkpoint}')
    print(f'Test set:     {len(test_ids)} structures  |  filter: {metal_filter}  |  hypotheses: {num_hypotheses}')
    print(f'Test RMSE:    {metrics["rmse"]:.2f} Å')
    print(f'Mean error:   {metrics["mean"]:.2f} Å')
    print(f'Median error: {metrics["median"]:.2f} Å')
    print(f'Error < 2 Å:  {metrics["pct_2a"]:.1f}%')
    print(f'Error < 5 Å:  {metrics["pct_5a"]:.1f}%')

    if log:
        log_run('test', checkpoint,
                test_size=len(test_ids),
                test_rmse=round(metrics['rmse'],    4),
                test_mean=round(metrics['mean'],    4),
                test_median=round(metrics['median'], 4),
                test_pct_2a=round(metrics['pct_2a'], 2),
                test_pct_5a=round(metrics['pct_5a'], 2))

    return metrics
