# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **metal-binding site prediction system** for protein structures. A Graph Neural Network (MPNN + attention head) predicts the 3D coordinates of metal ion binding sites in enzyme structures from the Protein Data Bank (PDB). ~43k PDB enzyme structures filtered for metal coordination annotations, X-ray diffraction, and resolution ≤ 2.5 Å.

## Running the Code

Core logic lives in the **`metal_predictor/` Python package**, driven by CLI entry points at the repo root. The notebooks in `notebooks/` are for data prep, exploration, and visualization — they are no longer the training path.

```bash
python train.py [options]              # train the coordinate-regression model
python evaluate.py --checkpoint <path> # evaluate on the test split
python train_classifier.py [options]   # train the closest-residue classifier variant
pytest tests/test_training_perf.py -s  # per-epoch timing regression tests
```

Device selection is automatic and prefers **MPS**, then CUDA, then CPU (this is an Apple Silicon dev box).
`mmseqs` CLI is required for sequence clustering in `notebooks/parse_cifs.ipynb`.

`README.md` holds the user-facing flag tables; keep it in sync when CLI flags change.

## Architecture

### Data Pipeline (`notebooks/parse_cifs.ipynb`)

```
cifs/*.cif → data/gemmi_atom14/chunk_*.parquet → data/sequences.fasta
           → MMseqs clusters (data/clusters/clusters.parquet) → data/pt_cache_v3/*.pt
```

- **atom14 format**: each residue = 14 atoms × 3D coords (42 values); missing atoms are NaN in the parquet, zero-filled for `x`
- **`data/pt_cache_v3/`**: pre-built graph tensors, one `.pt` per structure — the dataset loads these directly and only falls back to parquet parsing on a cache miss
- **`pos_distal`**: per residue, the last finite atom14 entry (≈ distal sidechain atom), with CA fallback. Predictions are built from these, not CA.
- **The cache was built at k = 10** by a `model.ipynb` cell, and `__getitem__` short-circuits on a cache hit, so `--k-neighbors` (CLI default 16) is silently ignored for every cached structure. Do not rebuild the cache to "fix" this mid-comparison; every reported result is k = 10.
- **Node features `x`** are absolute crystal-frame coordinates with missing atoms as exact (0,0,0). `--coord-frame {absolute,residue,centroid}` reframes them at load time (`MetalBindingDataset._reframe`): `residue` subtracts each residue's CA (removes the frame but also neighbor direction, since `x_j − x_i` stops being a displacement), `centroid` subtracts the CA centroid (removes the frame, keeps displacements, matches the head's output origin). Zero slots stay zero in every frame, and no real atom lands within 0.5 Å of the shifted origin, so zero remains a valid missing-atom sentinel (the heuristic relies on this). `pos`/`pos_distal`/`y` are never reframed. `--relative-coords` is a hidden alias for `residue`. Checkpoint shapes are unchanged, so eval cannot infer the frame: pass the flag to match training. No-op for `m0` models. **Measured (ZN, single head, Sept 2026): `residue` is the best frame** (sparsemax 37% < 2 Å vs 17% absolute), and `centroid` is the worst (8–10%, top-k RMSE ≈ naive baseline) despite preserving neighbour displacements. Don't re-propose centroid on the displacement argument; the network doesn't use it. ESM-2 models have only been trained in the absolute frame.
- **Cluster-aware splits**: `cluster_aware_split()` splits 80/10/10 at the MMseqs cluster level to prevent sequence-homology leakage. When a metal filter is active, `single_metal_resplit()` **recomputes** the split over single-metal clusters only (seed 42). All three entry points call that one helper — never re-derive the split inline, or train and eval will score against different test sets.

### Package Layout

```
metal_predictor/
├── constants.py              # RESIDUES, _ATOM14_COLS, BARE_METAL_RESNAMES, DONOR_SLOTS, DISTAL_CACHE_DIR
├── data/
│   ├── dataset.py            # MetalBindingDataset, split helpers, metal filters
│   ├── augmentation.py       # random SO(3) rotation, AugmentedDataset (train only)
│   └── sampler.py            # BucketBatchSampler
├── models/
│   ├── layers.py             # MLP, MetalMPNNLayer (MessagePassing, sum aggr)
│   ├── attention.py          # MetalPredictionHead, graph_sparsemax, graph_topk_attn
│   └── model.py              # MetalPredictionModel
├── training/
│   ├── train.py              # run_training(), wta_loss(), log_run()
│   ├── train_classifier.py   # run_classifier_training(), clf_loss(), log_clf_run()
│   └── evaluate.py           # run_evaluation() — model and/or baselines
├── heuristics.py             # donor_cluster_predict(): model-free HIS/CYS/ASP/GLU donor-atom baseline
└── inference.py              # load_model(), neighborhood_predict()
```

`metal_predictor/parsing/structure.py` holds the raw-file path used by `predict.py` (root CLI): gemmi atom14 extraction copied from `parse_cifs.ipynb`, `structure_to_graph()` which runs `MetalBindingDataset._process` on a one-structure temp parquet (so graph logic is not duplicated; verified bit-identical to the cache), and on-the-fly ESM-2 embeddings (verified identical to `data/esm2_cache`). `metal_predictor/visualization/` and `scripts/` are empty; bulk cache/ESM building is still one-off notebook cells in `notebooks/model.ipynb`.

### Model (`metal_predictor/models/`)

`MetalPredictionModel` forward pass:
1. Node encoding: `Linear(42 → 64)` on atom14 coords **+** residue-type embedding (21 types, incl. `UNK`)
2. `num_mpnn_rounds` MPNN layers (default 3): message MLP over `[x_i, x_j, edge_attr]` → sum aggregate → update MLP.
   **`num_mpnn_rounds=0` is the GNN ablation** — node features are skipped entirely and the head runs on the residue/ESM bypass alone (`use_node_features` follows suit).
3. `MetalPredictionHead`: per-residue logits → `/τ` → softmax / sparsemax / top-k → `centroid(CA) + Σ αᵢ · (pos_distalᵢ − centroid)`

Edge features are 4-dim: CA–CA distance, distal–distal distance, mean cross distance, and sidechain-vector dot product.

Head inputs: node features concatenated with either a residue-type bypass embedding (`res_emb_dim=16`) or, with `--use-plm`, a projected ESM-2 embedding from `data/esm2_cache/`.

**Multi-hypothesis (WTA):** `num_hypotheses > 1` swaps `attention_mlp` for a `attention_mlps` ModuleList of K parallel heads, trained with `wta_loss()` = best-hypothesis MSE + diversity penalty. `--eps-wta` assigns gradient to a random non-winner with probability ε (exploration; helps K > 3). Eval scores the closest hypothesis.

**Loss/metrics:** MSE on predicted vs. actual metal position; reported as RMSE, mean/median error, and % of structures within 2 Å / 5 Å. `evaluate.py` always reports the CA-centroid baseline and logs it as `test_centroid_rmse`; `run_training` prints the val-set centroid RMSE once before epoch 1 and logs `val_centroid_rmse` (epoch-1 val RMSE should be at or below it, since the zero-init head starts at the centroid). `--baseline` additionally reports the donor-cluster heuristic (single-metal only); with no `--checkpoint` it runs only the baselines and logs the heuristic as `heuristic_donor_r<radius>_<filter>`.

**Donor-cluster heuristic** (`heuristics.py`): per graph, the HIS/CYS/ASP/GLU sidechain donor atom with the most other-residue donor atoms within `--donor-radius` (default 4.0 Å, selected on the ZN val split), ties broken by distinct-residue count; prediction = mean of that atom and its neighbours. Never use a "tightest cluster" tie-break: salt bridges are tighter than Zn shells and it costs ~15 points of hit rate. At r = 4.0 it scores 88% of ZN test sites within 2 Å (70% cluster-averaged), above every ML checkpoint, but with worse RMSE from wrong-cluster misses.

**Classifier variant** (`train_classifier.py`): same model, but trains attention weights with BCE against a one-hot target on the residue closest to the metal (`closest_residue_idx`, computed in the dataset). Metric is per-graph argmax accuracy; logs to `data/runs_clf.csv`.

### Training Details

- Training set is wrapped in `AugmentedDataset` (fresh random SO(3) rotation per `__getitem__`); val/test are not.
- `BucketBatchSampler` groups structures by residue count so batch shapes repeat across epochs. This exists specifically because **MPS recompiles a Metal kernel per unique tensor shape** — random shuffling caused per-epoch time to grow. `tests/test_training_perf.py` guards against that regression.
- The final attention layer is zero-initialized on a fresh run, so epoch 1 starts from the centroid baseline.
- Two checkpoints are written per improvement: `<name>.pt` (weights only) and `<name>_full.pt` (+ optimizer/scheduler/epoch, used by `--resume`).
- `ReduceLROnPlateau` on val RMSE, patience 5.

### Checkpoints and Run Logging

Auto-generated name (v9 scheme, from `build_checkpoint_name` in `train.py`):

```
data/best_model_v9{_zn|_bm|_sm}_{attn_mode}{_plm}{_mN}{_eN}{_wtaN}{_rel|_cen}.pt
```

`_sm` = single-metal, `_zn` = zinc only, `_bm` = bare metal ions, `_mN` = non-default MPNN rounds, `_eN` = ε-WTA, `_wtaN` = K hypotheses, `_rel` = residue-frame coords, `_cen` = centroid-frame coords. v4–v8 checkpoints in `data/` are legacy and predate `pos_distal` and the current head.

Every train and eval run appends/updates a row in **`data/runs.csv`** keyed by checkpoint filename (`log_run`). `notebooks/lab.ipynb` reads that CSV to plot sweep results. Best results so far (test set):

| checkpoint | test RMSE | <2 Å | <5 Å |
|---|---|---|---|
| `best_model_v9_zn_sparsemax_plm_wta3.pt` | 5.00 Å | 66.0% | 86.5% |
| `best_model_v9_zn_sparsemax_plm_m0_wta3.pt` | 6.79 Å | 67.0% | 82.2% |

ESM-2 + sparsemax + 3-hypothesis WTA on the ZN subset is the strongest ML configuration, and these are **oracle-of-3, structure-weighted** numbers. Cluster-averaged (each MMseqs family once) the `m0` ablation is the best model on every metric, the residue-frame sparsemax model (`zn_sparsemax_rel`, structure-only) has the highest per-family single-prediction hit rate of any model (34.7%), and the donor-cluster heuristic beats all of them on hit rate. The ZN test set is 43 clusters with 59% in three families; the ZN val split (146 structures) is unrepresentative and early-stopped every single-head PLM run at epoch 1. README's Results section carries both aggregations.

`sweep*.sh` / `sweep_eval*.sh` run flag grids and tee to timestamped `.log` files at the repo root.

### Notebooks (`notebooks/`)

1. `pdb_search.ipynb` — RCSB PDB API query → `data/pdb_ids.txt`
2. `parse_cifs.ipynb` — CIF parsing, atom14 parquet chunks, MMseqs clustering, cache builds
3. `model.ipynb` — **legacy**: still holds inline copies of the dataset/model/training code that predate the package (no `num_mpnn_rounds`, older head). Treat the package as the source of truth; do not sync fixes back into these cells unless asked. Its `[NON-ESSENTIAL]`-tagged cells are the one-time cache/ESM builders and exploratory analyses.
4. `visualization.ipynb` — 3D backbone/attention rendering (py3Dmol) and attention-by-residue plots
5. `lab.ipynb` — sweep result tables and plots from `data/runs.csv`

Notebook cells use paths relative to the repo root (`data/...`), except `lab.ipynb` which uses `../data/`.

### Supporting Directories

- `foundry/` — RosettaCommons toolkit (RFdiffusion3, RosettaFold3, ProteinMPNN) for downstream protein design; not used during training
- `LigandMPNN/` — ligand-aware inverse folding model; not used during training
- Both, plus `data/` and `.venv/`, are gitignored.

## Code Style

- Prefer simple, minimal implementations. Do not add comparisons, extra options, or alternatives unless explicitly asked. When asked for a training loop, give one training loop.
- Before executing any multi-step task, propose a plan and wait for approval.
- When making notebook changes, change only what was asked — do not refactor surrounding cells or add features.
- Use standard pandas dtypes (avoid `StringDtype`) unless explicitly requested.
- Aligned assignment blocks and `# ── section ──` comment rules are the house style in this package; match them.

## Tools & Conventions

- Always use the `NotebookEdit` tool for Jupyter notebook edits. Never use the `Edit` tool on `.ipynb` files.
- Run Python via the project venv: `.venv/bin/python` (dependencies are not on the system interpreter).
- New CLI flags need wiring in three places: the `argparse` block, the `run_*` signature, and `log_*_run()` so the sweep CSV stays complete.
- `runs.csv` columns must never be float64 if they will ever hold a string or bool: pandas 3.0 raises `TypeError` on `.at` writes into a float64 column, and `log_run` updates existing rows with `.at`. Write bools as strings (`coord_frame` is `absolute|residue|centroid`, not True/False).

## Debugging Guidelines

- When a fix doesn't seem to work in a notebook, suggest restarting the Jupyter kernel before assuming the code is still broken — stale kernel state is a common culprit.
- Always suggest a kernel restart after fixing import-level or class-level changes.
- Propose the minimal fix first. Only escalate complexity if the simple fix fails.
- Checkpoint load failures are usually an architecture-flag mismatch: `--attn-mode`, `--num-hypotheses`, `--use-plm`, and `--num-mpnn-rounds` must match training. `run_evaluation()` infers MPNN rounds and `use_node_features` from the state dict keys; the rest is on the caller. `--coord-frame` does not change shapes, so a mismatch loads fine and silently scores garbage; the `_rel`/`_cen` tag in the checkpoint name is the only guard (also needed with `--resume`).
- Per-epoch time growth on MPS points at batch-shape churn — check `BucketBatchSampler` before suspecting the model.

## Key Dependencies

```
torch, torch_geometric    # GNN training
gemmi, biopython          # Protein structure parsing
pandas, numpy, scipy      # Data manipulation and k-NN graphs
mmseqs2                   # Sequence clustering (CLI)
matplotlib, py3Dmol       # Visualization
fastparquet               # Parquet I/O
esm                       # ESM-2 protein language model (optional, for --use-plm)
pytest                    # tests/
```
