# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **metal-binding site prediction system** for protein structures. A Graph Neural Network (3-layer MPNN + attention head) is trained to predict 3D coordinates of metal ion binding sites in enzyme structures from the Protein Data Bank (PDB). ~43k PDB enzyme structures filtered for metal coordination annotations, X-ray diffraction, and resolution ≤ 2.5 Å.

## Running the Code

All core logic lives in Jupyter notebooks — run in order for the full pipeline:

1. `pdb_search.ipynb` — Query RCSB PDB API for metal-containing enzyme structures → `data/pdb_ids.txt`
2. `parse_cifs.ipynb` — Parse CIF files, extract atom14 coordinates, build sequence clusters and k-NN graph caches
3. `model.ipynb` — Train/evaluate the MPNN model, save checkpoints
4. `visualization.ipynb` — Generate 3D visualizations and attention analysis plots

GPU (CUDA) is recommended for model training. `mmseqs` CLI tool is required for sequence clustering in `parse_cifs.ipynb`.

## Architecture

### Data Pipeline (`parse_cifs.ipynb`)

```
cifs/*.cif → gemmi_atom14/chunk_*.parquet → sequences.fasta → MMseqs clusters → pt_cache_v3/*.pt
```

- **atom14 format**: each residue represented as 14 atoms × 3D coordinates (42 values)
- **pt_cache_v3/**: pre-built k-NN graph tensors (one `.pt` per structure), loaded directly by the dataset class
- **Cluster-aware splits**: train/val/test (80/10/10) split at the cluster level to prevent sequence homology leakage

### Model (`model.ipynb`)

**`MetalBindingDataset`** — loads parquet metadata + cached graph tensors, builds k=10–32 nearest-neighbor graphs from CA positions, optionally loads ESM-2 embeddings from `data/esm2_cache/`

**`MetalBindingMPNN`** — architecture:
1. Residue type embedding (20 amino acids → 64-dim)
2. Node encoding: atom14 coords linear projection + residue embedding
3. 3 MPNN layers: message MLP (concat node_i, node_j, edge features) → update MLP
4. Attention head: per-residue logits → softmax/sparsemax/topk → weighted sum of sidechain centroids → predicted metal position

**Key config flags** (top of training cell in `model.ipynb`):
- `SINGLE_METAL_ONLY` — filter to structures with exactly 1 metal
- `ATTN_MODE` — `'softmax'`, `'sparsemax'`, or `'topk'`
- `USE_PLM` — inject ESM-2 embeddings into node features

**Loss:** L2 distance (predicted vs. actual metal position). Metrics: RMSE, MAE, accuracy at 2 Å and 5 Å thresholds.

### Model Checkpoints

Saved to `data/`:
- `best_model_v8_sm_softmax.pt` — best single-metal, softmax attention
- `best_model_v8_sm_softmax_plm.pt` — same with ESM-2 embeddings (best overall)

### Supporting Directories

- `foundry/` — RosettaCommons toolkit (RFdiffusion3, RosettaFold3, ProteinMPNN) for downstream protein design; not used during training
- `LigandMPNN/` — Ligand-aware inverse folding model; not used during training

## Code Style

- Prefer simple, minimal implementations. Do not add comparisons, extra options, or alternatives unless explicitly asked. When asked for a training loop, give one training loop.
- Before executing any multi-step task, propose a plan and wait for approval.
- When making notebook changes, change only what was asked — do not refactor surrounding cells or add features.
- Use standard pandas dtypes (avoid `StringDtype`) unless explicitly requested.

## Tools & Conventions

- Always use the `NotebookEdit` tool for Jupyter notebook edits. Never use the `Edit` tool on `.ipynb` files.

## Debugging Guidelines

- When a fix doesn't seem to work in a notebook, suggest restarting the Jupyter kernel before assuming the code is still broken — stale kernel state is a common culprit.
- Always suggest a kernel restart after fixing import-level or class-level changes.
- Propose the minimal fix first. Only escalate complexity if the simple fix fails.

## Key Dependencies

```
torch, torch_geometric    # GNN training
gemmi, biopython          # Protein structure parsing
pandas, numpy, scipy      # Data manipulation and k-NN graphs
mmseqs2                   # Sequence clustering (CLI)
matplotlib, py3Dmol       # Visualization
fastparquet               # Parquet I/O
esm                       # ESM-2 protein language model (optional, for USE_PLM)
```
