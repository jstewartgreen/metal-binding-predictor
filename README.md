# Metal-Binding Site Prediction (CS540 Final Project)

A Graph Neural Network for predicting 3D coordinates of metal ion binding sites in enzyme structures from the Protein Data Bank.

## Architecture

- **Data pipeline:** CIF structures → atom14 parquet chunks → MMseqs2 clusters → k-NN graph caches (`data/pt_cache_v3/`)
- **Model:** MPNN (default 3 rounds) + attention head → weighted sum of distal sidechain atom offsets from the CA centroid → metal position
- **Attention modes:** softmax, sparsemax (per-graph), topk
- **Multi-hypothesis:** K parallel attention heads trained with a winner-takes-all loss
- **Optional:** ESM-2 protein language model embeddings injected into the attention head
- **Splits:** cluster-aware 80/10/10 so no sequence family appears in both train and test

Device is chosen automatically: MPS → CUDA → CPU.

## Setup

```bash
pip install torch torch_geometric gemmi biopython pandas numpy scipy matplotlib fastparquet pytest
# Optional (for --use-plm):
pip install esm
# Required for notebooks/parse_cifs.ipynb:
# install mmseqs2 CLI (https://github.com/soedinglab/MMseqs2)
```

All commands below are run from the repo root and expect `data/` to be populated by the notebooks (see [Notebooks](#notebooks)).

## Training

```
python train.py [options]
python train.py --help   # full option list
```

| Flag | Default | Description |
|------|---------|-------------|
| `--attn-mode` | `softmax` | Attention mechanism: `softmax`, `sparsemax`, `topk` |
| `--num-hypotheses` | `1` | Attention heads (>1 enables winner-takes-all multi-hypothesis loss) |
| `--diversity-weight` | `0.05` | Diversity penalty weight for WTA loss |
| `--eps-wta` | `0.0` | ε-WTA: probability of training a random non-winning head (exploration) |
| `--num-mpnn-rounds` | `3` | Message-passing rounds; `0` = no GNN (attention head only) |
| `--metal-filter` | `None` | Restrict training set: `ZN` (zinc only), `bare` (bare metal ions), `None` (all) |
| `--single-metal-only` | `True` | Train on structures with exactly one metal (default on) |
| `--all-metals` | — | Override: include structures with multiple metals |
| `--use-plm` | `False` | Inject ESM-2 embeddings (requires `data/esm2_cache/`) |
| `--esm-cache-dir` | `data/esm2_cache` | Location of precomputed ESM-2 embeddings |
| `--esm-dim` | `480` | ESM-2 embedding width (480 for `esm2_t12_35M_UR50D`) |
| `--temperature` | `1.0` | Initial attention temperature τ |
| `--fixed-temperature` | — | Freeze τ instead of learning it |
| `--k-neighbors` | `16` | k for k-NN graph construction |
| `--epochs` | `25` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--grad-clip` | `1.0` | Gradient clipping norm |
| `--resume` | — | Resume from `_full.pt` checkpoint (includes optimizer state) |
| `--checkpoint` | auto | Override checkpoint path (see naming below) |

**Checkpoint naming** (auto-generated when `--checkpoint` is not set):

```
data/best_model_v9{_zn|_bm|_sm}_{attn_mode}{_plm}{_m<rounds>}{_e<eps>}{_wta<K>}.pt
```

Tags appear only when non-default. Examples:

```bash
python train.py --attn-mode softmax
# → data/best_model_v9_sm_softmax.pt

python train.py --metal-filter ZN --attn-mode sparsemax --num-hypotheses 3 --use-plm
# → data/best_model_v9_zn_sparsemax_plm_wta3.pt

python train.py --metal-filter ZN --attn-mode sparsemax --num-hypotheses 3 --use-plm --num-mpnn-rounds 0
# → data/best_model_v9_zn_sparsemax_plm_m0_wta3.pt
```

Each improvement writes `<name>.pt` (weights) and `<name>_full.pt` (weights + optimizer + scheduler, used by `--resume`). Training metrics are logged to `data/runs.csv`, keyed by checkpoint filename.

## Evaluation

```
python evaluate.py --checkpoint <path> [options]
python evaluate.py --help   # full option list
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--metal-filter` | `None` | Same filter used during training |
| `--attn-mode` | `softmax` | Must match the trained model |
| `--num-hypotheses` | `1` | Must match the trained model |
| `--use-plm` | `False` | Must match the trained model |
| `--baseline` | — | Also report the naive CA-centroid prediction error |
| `--no-log` | — | Skip writing results to `data/runs.csv` |

The number of MPNN rounds is inferred from the checkpoint. Reports RMSE, mean/median error, and accuracy at 2 Å and 5 Å thresholds.

## Closest-Residue Classifier

An alternative training objective that trains the attention weights to peak on the residue nearest the metal, rather than regressing coordinates:

```
python train_classifier.py [options]
python train_classifier.py --help
```

Uses the same data flags as `train.py` (`--metal-filter` defaults to `ZN`, `--attn-mode` to `sparsemax`). Reports per-structure argmax accuracy, saves to `data/best_model_v9{_zn|_bm}_{attn_mode}{_plm}_clf.pt`, and logs to `data/runs_clf.csv`.

## Results

Test set, zinc-only structures (297 structures). Full sweep in `data/runs.csv`.

| Configuration | Test RMSE | < 2 Å | < 5 Å |
|---|---|---|---|
| sparsemax + ESM-2 + 3 hypotheses | 5.00 Å | 66.0% | 86.5% |
| sparsemax + ESM-2 + 3 hypotheses, no MPNN | 6.79 Å | 67.0% | 82.2% |
| sparsemax + 3 hypotheses | 9.30 Å | 23.6% | 51.2% |
| softmax, single head | 12.64 Å | 13.1% | 46.5% |
| CA-centroid baseline | 17.21 Å | — | — |

`sweep*.sh` and `sweep_eval*.sh` reproduce the flag grids that produced these numbers.

## Tests

```
pytest tests/test_training_perf.py -s -v
```

Per-epoch timing benchmarks that guard against epoch-time growth on MPS (Metal recompiles kernels per unique batch shape; `BucketBatchSampler` keeps shapes stable across epochs). These train real models on a small zinc subset and take several minutes.

## Notebooks

Located in `notebooks/`. Run the first two to build `data/`:

1. `pdb_search.ipynb` — Query RCSB PDB API → `data/pdb_ids.txt`
2. `parse_cifs.ipynb` — Parse CIF files, build clusters and caches
3. `model.ipynb` — Legacy interactive training notebook; also holds the one-time cache and ESM-2 embedding builder cells. The `metal_predictor` package is the source of truth.
4. `visualization.ipynb` — 3D visualizations and attention analysis
5. `lab.ipynb` — Sweep result tables and plots from `data/runs.csv`

## Project Structure

```
metal_predictor/              # installable package
├── constants.py              # shared constants (residues, atom14 cols, metal names)
├── data/
│   ├── dataset.py            # MetalBindingDataset, cluster-aware split helpers, metal filters
│   ├── augmentation.py       # SO(3) random rotation augmentation
│   └── sampler.py            # BucketBatchSampler (stable batch shapes for MPS)
├── models/
│   ├── layers.py             # MLP, MetalMPNNLayer
│   ├── attention.py          # MetalPredictionHead, sparsemax, topk
│   └── model.py              # MetalPredictionModel
├── training/
│   ├── train.py              # run_training(), wta_loss(), log_run()
│   ├── train_classifier.py   # run_classifier_training(), clf_loss()
│   └── evaluate.py           # run_evaluation()
└── inference.py              # load_model(), neighborhood_predict()

train.py                      # CLI: coordinate regression training
evaluate.py                   # CLI: test-set evaluation
train_classifier.py           # CLI: closest-residue classifier training
tests/test_training_perf.py   # epoch-time regression benchmarks
sweep*.sh, sweep_eval*.sh     # flag-grid runners
notebooks/                    # data prep, legacy training, visualization
```
