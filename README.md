# MetalMPNN: Metal-Binding Site Prediction from Protein Structure

A graph neural network that predicts the 3D coordinates of a bound metal ion from enzyme protein structures.

## Purpose

Many enzyme families rely on bound metal ion cofactors for structure or reactivity. While computational protein modeling greatly improves the prospects of *de novo* enzyme design, many models such as RFDiffusion and AlphaFold2 generate predicted structures without metal ion contexts. Dedicated metal-ion localization models would thus be valuable to an eventual enzyme generation pipeline. 

Existing tools either classify coordinating residues (individual amino acids) or, like Metal3D[1], predict a metal density on a voxel grid around candidate residues and cluster it into sites. This project instead regresses the ion's coordinates directly from a residue graph via a Message-Passing Neural Network (MPNN), a variation of a Graph Neural Network (GNN). Additional experiments injecting per-residue embeddings from protein language model ESM-2[2] were performed to evaluate performance effects. 

## Summary of findings

1. **ESM-2 embedding injection outperformed structure-only MPNN inference** 

    Adding per-residue ESM-2 embeddings to the attention head halves RMSE and triples the fraction of zinc sites placed within 2 Å, even after ablating message-passing. That gain is concentrated in three large protein families; on the remaining forty families the best ESM-2-only model places 36% of sites within 2 Å, comparable to 35% for the best MPNN-only model.

2. **Residue-relative coordinate frame outperformed absolute and centroid-translated coordinate frames with sparse attention-weighting** 

    With atom positions expressed relative to each residue's own CA, the structure-only sparsemax model places 37% of zinc sites within 2 Å (35% per family) against 7% for the naive centroid baseline, with accuracy spread evenly across families. The same model on raw crystal-frame coordinates reaches 17%, and on coordinates centered at the protein's CA centroid, 9%.

3. **Donor heteroatom heuristic places more coordination sites <2 Å than all models tested** 

    Taking the His/Cys/Asp/Glu donor atom with the most other-residue donor atoms within 4.0 Å, and predicting the centroid of that cluster, places 88% of zinc sites within 2 Å with no learned parameters. Edge-case placements harm the heuristic's average RMSE across test structures.


## Dataset

The dataset is comprised of structures from the RCSB Protein Data Bank matching a single search (`queries/pdb_query.json`): X-ray diffraction crystallographic structures of metal-containing enzymes with resolution <2.5 Å.

**Graph structure.** Processed structures are represented as graphs, with residues as nodes connected to their 10 nearest neighbors by alpha-carbon (CA) distance. Node features are the 42 atom14 coordinates (fixed layout with 14 atom coordinates, empty atoms zero-filled) and the residue type. Edge features are four distances and angles between the two residues: CA–CA distance, distance between their distal side-chain atoms, the mean of the two cross distances, and the cosine between their CA-to-side-chain vectors. Models predict from distal side-chain atoms (`pos_distal`, the last resolved atom14 slot, CA as fallback). 

**Coordinate frames.** Atom positions were encoded by one of three coordinate frames: the experimentally-reported or "absolute" frame, translation to set the CA centroid as the origin ("centroid" frame), and per-residue offsets with regard to the CA ("residue" frame).

**Clustering and splits.** Sequences were clustered with MMseqs2 (`easy-cluster`, 30% identity, 80% coverage), giving 4,573 clusters. Train, validation, and test are split 80/10/10 at the cluster level, so no sequence family appears on both sides of any split. This prevents the model from scoring well by memorizing homologs. Splits are cluster-level but not family-balanced; three families make up 59% of the zinc test set (see Results).

**Subsets used.** Experiments use structures with exactly one metal: 12,726 structures in 1,845 clusters. Two filters on the metal identity define the reported subsets.

| Subset | Metals | Structures | Clusters | Train / val / test |
|---|---|---|---|---|
| Zinc | ZN | 4,728 | 469 | 4,285 / 146 / 297 |
| Bare metal ions | Zn, Mg, Fe, Ca, Cu, Mn, Co, Ni, Mo, Cd, V, W | 9,898 | 1,503 | 8,039 / 873 / 986 |


## Model

The model has three stages: a per-residue encoder, three rounds of message passing over the k-NN graph, and an attention head that turns per-residue scores into one 3D point. All hidden widths are 64.

**Message passing.** Each residue's 42 atom14 coordinates pass through a linear layer (42 → 64 dim) and are summed with a learned 64-dim embedding of its residue type. Next, message passing occurs in three rounds with sum aggregation. For each directed edge the message MLP reads the sender features, receiver features, and the four edge features (64 + 64 + 4 → 64 → 64, ReLU). Messages arriving at a residue are summed and concatenated with its current features, and an update MLP (128 → 64 → 64) produces the next-round features.

**Attention head.** The head scores every residue and converts the scores into a position. Its input is the residue's post-message-passing features concatenated with a small bypass embedding of the residue type (16-dim), so the head can weight residue identity directly. An MLP (80 → 40 → 1, ReLU, dropout 0.3) produces one logit per residue, divided by a learned temperature τ. The logits are normalized over the residues of each structure with one of three functions: softmax, sparsemax, or top-k. Predictions are the attention-weighted sum of residue distal side-chain atoms.

**Training.** Mean squared error between the predicted and deposited metal position. Training samples are augmented via random rotation. Adam at learning rate 1e-3 with plateau-based decay, gradient clipping at 1.0, batch size 16, 25 epochs, batches bucketed by protein length. The checkpoint with the lowest validation RMSE is kept.

**Sequence embeddings (added experiment).** With `--use-plm`, the residue-type bypass is replaced by a linear projection (480 → 64) of per-residue embeddings from a frozen ESM-2 model (`esm2_t12_35M_UR50D`), precomputed once per structure. Everything else is unchanged. This tests how much a protein language model's sequence context adds on top of explicit structure.


## Results

### Evaluation protocol

All numbers are on test structures whose sequence family never appears in training. Every model here makes one committed prediction per structure. Two aggregations are reported side by side. **Structure-weighted** counts every structure once, so large families dominate. **Cluster-averaged** computes each metric within a family first and then averages over families, so every family counts once. Metrics are RMSE, and the fraction of structures whose predicted site lies within 2 Å and within 5 Å of the deposited ion. Rows are labeled by the coordinate frame of the node features (see [Dataset](#dataset)); ESM-2 rows use the absolute frame.

### Zinc

Table 1: Test performance summary across model configurations of interest.
| Configuration | Struct. RMSE (Å) | Struct. < 2 Å | Struct. < 5 Å | Cluster RMSE (Å) | Cluster < 2 Å | Cluster < 5 Å |
|---|---|---|---|---|---|---|
| sparsemax + ESM-2 | **7.25**| 52.9% | 68.0% | **12.11** | 24.1% | 48.6% |
| top-k + ESM-2 | 8.42 | 49.5% | 77.4% | 14.20 | 26.6% | 59.0% |
| softmax + ESM-2 | 7.55 | 33.0% | 67.3% | 12.94 | 22.0% | 50.9% |
| sparsemax, ESM-2 only | 7.65 | 49.5% | 70.4% | 11.97 | 25.7% | 53.1% |
| top-k, ESM-2 only  | 10.75 | 57.2% | 71.7% | 14.86 | 38.8% | 58.7% |
| softmax, ESM-2 only  | 7.39 | 54.5% | 76.1% | 13.42 | 35.0% | 55.8% |
| sparsemax, residue frame | 12.17 | 37.0% | 53.9% | 12.43 | 34.7% | 47.4% |
| top-k, residue frame | 11.44 | 21.2% | 50.2% | 12.63 | 12.6% | 40.6% |
| softmax, residue frame | 15.41 | 16.5% | 47.1% | 13.96 | 24.0% | 40.9% |
| top-k, absolute frame | 12.38 | 29.0% | 39.7% | 13.07 | 25.9% | 42.7% |
| softmax, absolute frame | 12.64 | 13.1% | 46.5% | 13.29 | 13.8% | 37.2% |
| sparsemax, absolute frame | 15.58 | 17.2% | 27.6% | 14.20 | 12.9% | 30.2% |
| top-k, centroid frame | 17.38 | 9.8% | 38.1% | 19.05 | 4.2% | 25.6% |
| softmax, centroid frame | 13.59 | 8.4% | 22.9% | 14.15 | 3.8% | 23.8% |
| sparsemax, centroid frame | 15.02 | 9.4% | 19.9% | 14.54 | 5.3% | 19.6% |
| donor-cluster heuristic (r = 4.0 Å) | 8.51 | **88.2%** | **90.2%** | 13.83 | **69.7%** | **78.1%** |
| CA-centroid baseline | 17.21 | 7.1% | 28.6% | 18.54 | 1.1% | 9.9% |

![ESM-2](/figures/ablation.png)

Fig. 1 Average RMSE (Å) test performance across attention types for MPNN-only, MPNN-ablated (ESM-2 only), and combined models. All models were evaluated using absolute coordinate framing.

In structure-weighted evaluations, both message-passing models and MPNN-ablated (0 rounds message-passing) models with ESM-2 embeddings achieved greater accuracy than the MPNN alone. However, mean per-cluster RMSE yielded similar average accuracy across all model categories, with average test RMSE comparable to the donor-cluster heuristic. This suggests that the benefits of ESM-2 embeddings share some proportionality to the relative abundance of the protein family in the dataset, which are abrogated in cluster-averaged evaluations. All three model categories converge per family, so ESM-2's structure-weighted gain is a large-family effect, and message passing adds nothing on either aggregation.

![coord_frame](/figures/coordframe.png)
Fig. 2 RMSE (Å) performance of models with absolute (*i.e.* experimental), relative (per-residue), and centroid coordinate frames. Left: mean performance over all structures in test set. Right: mean of performance per cluster. 

Highest MPNN-only test accuracy was obtained when per-residue atom positions were encoded relative to the residue's CA with sparsemax or top-k attention, as opposed to the experimentally-reported or centroid-translated coordinate frame. In cluster-averaged testing, these models achieved comparable RMSE to the donor-atom heuristic. 

### Coordination accuracy within protein family groups

Table 2: Coordination predictions <2 Å on three largest and remaining forty test-set clusters

| Configuration | Three largest families (174 structures) | Other 40 families (123 structures) |
|---|---|---|
| sparsemax + ESM-2 | 74.1% | 22.8% |
| top-k + ESM-2 | 71.8% | 17.9% |
| softmax + ESM-2 | 48.9% | 10.6% |
| sparsemax, ESM-2 only (no MPNN) | 74.7% | 13.8% |
| top-k, ESM-2 only (no MPNN) | 71.8% | 36.6% |
| softmax, ESM-2 only (no MPNN) | 74.1% | 26.8% |
| sparsemax, residue frame | 38.5% | 35.0% |
| top-k, residue frame | 29.9% | 8.9% |
| softmax, residue frame | 9.2% | 26.8% |
| top-k, absolute frame | 29.3% | 28.5% |
| softmax, absolute frame | 13.2% | 13.0% |
| sparsemax, absolute frame | 20.7% | 12.2% |
| donor-cluster heuristic (r = 4.0 Å) | 93.1% | 81.3% |
| CA-centroid baseline | 12.1% | 0.0% |

The three centroid-frame models are omitted from this table; they score 12 to 14% on the three largest families and 3 to 6% on the rest. Scored on those three families with the sparsemax + ESM-2 model: MutM DNA glycosylase (C4 zinc finger) 100% within 2 Å; ACE N-domain (HExxH metalloprotease) 71%; GH1 β-glucosidase 48% with RMSE 11.45 Å. Among the smaller clusters, histone deacetylase 10 (20 structures, catalytic zinc) scores 0% within 2 Å but 100% within 5 Å, a consistent offset of about 4 Å on one family rather than noise.


### Discussion

**MPNN architecture alone was insufficient for metal ion coordination prediction**
While MPNN-only implementations were able to improve prediction accuracy over the naive CA-centroid baseline, only models with ESM-2 embeddings reached average accuracy comparable to the donor-cluster heuristic on RMSE. Ablation studies with 0 rounds of message-passing and ESM-2 embeddings performed comparably to the combined MPNN+ESM-2 model in structure-weighted evaluation. However, MPNN-only implementations performed comparably with ESM-2 combination and MPNN-ablated models on cluster-averaged evaluations. 

Assuming the distribution and size of protein families in the test data mirror those in the corpora used to pretrain PLMs like ESM-2, it is plausible that these learned embeddings carry more information for homologs of these high-prevalence familes. Conversely, the MPNN's "structure-only" predictions appear less sensitive to protein family prevalence in the test distribution. Mean attention weights by residue type across the test set were roughly equivalent among all models, favoring cysteine and histidine residues, suggesting that the relevant Zn metallobiochemistry was encoded.

The MPNN's robustness with regards to the test distribution does not translate to the proportion of high accuracy (<2 Å) predictions, however. Structure-weighted, the MPNN-ablated models matched or slightly exceeded the combined models on <2 Å, indicating the message passing added no measurable benefit on top of ESM-2 for any attention type. 

**Residue coordinate frame yielded greater MPNN performance**
Of the three coordinate frames established, the residue-relative framing (*i.e.* in which each residue's atom positions are offsets relative to its CA) showed relatively improved average RMSE with sparsemax and top-k attention compared to the other two coordinate frames, and reduced performance with softmax attention. While further experiments with broader datasets are required to support any significant claims, it is plausible that, given that the residue-relative frame bounds the input domain to the maximal length of a residue's sidechain, the informative fraction of each residue's input range increases compared to other coordinate frames. Sparse attention lets the head act on better per-residue scores, while softmax's diffuse pooling dilutes them regardless of frame, placing softmax's per-frame difference within single-run variation.

**Neither the model nor the heuristic meets the bar for pipeline use**
As shown in Table 1, the greatest proportion of predictions <2 Å were achieved by a simple donor-heteroatom heuristic. By comparison, the strongest published deep learning model in predicting metal ion coordination, Metal3D[1], narrows the structural search to putative donor residues to predict the zinc density via voxelized CNN, reaching a mean absolute deviation of 0.70±0.64 Å at a probability cutoff of 0.9. Future extensions could leverage the donor-atom heuristic as a potential narrowing of the search space. If the heuristic's recall over three to five candidates captures all binding sites, a classifier over these candidates, with either MPNN or lightweight ESM-2/MLP encoding, is a more viable direction than coordinate regression.

## Limitations and next steps

- **Validation split.** Enlarge it or use cluster-level cross-validation so ESM-2 checkpoint selection is not decided by a handful of families. Every ESM-2 checkpoint was selected at epoch 1 on a 146-structure split dominated by one family, and validation error diverged from test.
- **Family imbalance.** Cap structures per cluster during training. One family is 25% of the zinc training set; another is 27% of the bare-metal test set.
- **Adventitious and nucleotide-coordinated metals.** Filter zinc sites with no protein coordination, and either separate nucleotide-coordinated Mg from the bare-metal set or report it on its own.
- **Rank candidates instead of regressing coordinates.** The heuristic proposes sites better than the model does. A model that scores the heuristic's candidate clusters would pair the rule's recall with a learned ranker and could fix the wrong-cluster misses.
- **Coordinate frame and ESM-2 together.** Residue-relative coordinates were the best of the three encodings, but every ESM-2 model was trained on raw crystal coordinates. Combining the two is the obvious next run.
- **Single runs.** Every configuration was trained once. No variance across seeds is reported, so differences of a few points between rows should not be over-read.
- **No packaged inference.** There is no standalone command that takes a PDB file and returns a prediction, and no pretrained weights are distributed. Running the model requires the data pipeline below.

## Reproducing and extending

### Setup

```bash
pip install -r requirements.txt
```

Pinned versions are in `requirements.txt`. Install torch before torch_geometric; PyG wheels are built per torch version and platform, and the [official install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) is the fallback if the pins fail on your system. Only core PyG is used, so the `torch_scatter` / `torch_sparse` extensions are not required. ESM-2 embeddings need `fair-esm` (the `esm` import from facebookresearch, not the EvolutionaryScale package). Sequence clustering needs the [mmseqs2](https://github.com/soedinglab/MMseqs2) CLI.

Device is chosen automatically: MPS, then CUDA, then CPU. All commands below run from the repo root and expect `data/` to be populated by the notebooks.

### Data pipeline

Notebooks in `notebooks/`. Run the first two to build `data/`:

1. `pdb_search.ipynb` — Query the RCSB PDB search API with `queries/pdb_query.json`, paginating the full result set to `data/pdb_ids.txt`
2. `parse_cifs.ipynb` — Parse CIF files to atom14 parquet chunks and run MMseqs2 clustering
3. `model.ipynb` — Legacy interactive training notebook; holds the one-time graph-cache and ESM-2 embedding builder cells. The `metal_predictor` package is the source of truth.
4. `visualization.ipynb` — 3D visualizations and attention analysis (source of `vis1.png`)
5. `lab.ipynb` — Sweep result tables and plots from `data/runs.csv`

### Training

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
| `--k-neighbors` | `16` | k for k-NN graph construction (cached graphs are k = 10 regardless) |
| `--coord-frame` | `absolute` | Frame for atom14 node coordinates: `absolute`, `residue` (relative to each residue's CA), or `centroid` (relative to the CA centroid). `--relative-coords` is an alias for `residue` |
| `--epochs` | `25` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--grad-clip` | `1.0` | Gradient clipping norm |
| `--resume` | — | Resume from `_full.pt` checkpoint (includes optimizer state) |
| `--checkpoint` | auto | Override checkpoint path (see naming below) |

**Checkpoint naming** (auto-generated when `--checkpoint` is not set):

```
data/best_model_v9{_zn|_bm|_sm}_{attn_mode}{_plm}{_m<rounds>}{_e<eps>}{_wta<K>}{_rel|_cen}.pt
```

Tags appear only when non-default. Examples:

```bash
python train.py --attn-mode softmax
# → data/best_model_v9_sm_softmax.pt

python train.py --metal-filter ZN --attn-mode sparsemax --use-plm
# → data/best_model_v9_zn_sparsemax_plm.pt

python train.py --metal-filter ZN --attn-mode topk --coord-frame centroid
# → data/best_model_v9_zn_topk_cen.pt
```

Each improvement writes `<name>.pt` (weights) and `<name>_full.pt` (weights + optimizer + scheduler, used by `--resume`). Training metrics are logged to `data/runs.csv`, keyed by checkpoint filename. `sweep*.sh` and `sweep_eval*.sh` reproduce the flag grids behind the results tables.

### Evaluation

```
python evaluate.py --checkpoint <path> [options]
python evaluate.py --baseline --metal-filter ZN   # heuristic baselines only
python evaluate.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` checkpoint; required unless `--baseline` |
| `--metal-filter` | `None` | Same filter used during training |
| `--attn-mode` | `softmax` | Must match the trained model |
| `--num-hypotheses` | `1` | Must match the trained model |
| `--use-plm` | `False` | Must match the trained model |
| `--coord-frame` | `absolute` | Must match the trained model |
| `--baseline` | — | Also report the donor-cluster heuristic; with no `--checkpoint`, run only the baselines |
| `--donor-radius` | `4.0` | Donor–donor radius in Å for the donor-cluster heuristic |
| `--no-log` | — | Skip writing results to `data/runs.csv` |

The number of MPNN rounds is inferred from the checkpoint. Reports RMSE, mean/median error, and accuracy at 2 Å and 5 Å thresholds. With `--num-hypotheses` > 1 the closest hypothesis is scored. The CA-centroid baseline is always reported alongside the model and logged as `test_centroid_rmse`; training likewise prints the validation-set centroid RMSE once and logs it as `val_centroid_rmse`. The heuristic assumes one metal per structure and is logged under the name `heuristic_donor_r<radius>_<filter>`. Cluster-averaged metrics are not produced by `evaluate.py`; they were computed from per-structure errors grouped by MMseqs2 cluster.

### Closest-residue classifier

An alternative training objective that trains the attention weights to peak on the residue nearest the metal, rather than regressing coordinates:

```
python train_classifier.py [options]
python train_classifier.py --help
```

Uses the same data flags as `train.py` (`--metal-filter` defaults to `ZN`, `--attn-mode` to `sparsemax`). Reports per-structure argmax accuracy, saves to `data/best_model_v9{_zn|_bm}_{attn_mode}{_plm}{_rel|_cen}_clf.pt`, and logs to `data/runs_clf.csv`.

### Tests

```
pytest tests/test_training_perf.py -s -v
```

Per-epoch timing benchmarks that guard against epoch-time growth on MPS (Metal recompiles kernels per unique batch shape; `BucketBatchSampler` keeps shapes stable across epochs). These train real models on a small zinc subset and take several minutes.

### Repo layout

```
metal_predictor/              # installable package
├── constants.py              # shared constants (residues, atom14 cols, metal names, donor slots)
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
├── heuristics.py             # donor_cluster_predict(): model-free donor-atom baseline
└── inference.py              # load_model(), neighborhood_predict()

train.py                      # CLI: coordinate regression training
evaluate.py                   # CLI: test-set evaluation and baselines
train_classifier.py           # CLI: closest-residue classifier training
tests/test_training_perf.py   # epoch-time regression benchmarks
sweep*.sh, sweep_eval*.sh     # flag-grid runners
notebooks/                    # data prep, legacy training, visualization
queries/pdb_query.json        # RCSB search: EC-annotated, metal coordination linkage, X-ray, ≤ 2.5 Å
pdb_search.py                 # runs that query; fetches one page (100 IDs). notebooks/pdb_search.ipynb paginates the full set
vis1.png                      # attention figure used above
```

---
## References

1. Dürr, S. L., Levy, A. & Rothlisberger, U. Metal3D: a general deep learning framework for accurate metal ion location prediction in proteins. *Nat. Commun.* **14**, 2713 (2023). https://doi.org/10.1038/s41467-023-37870-6

2. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023). https://doi.org/10.1126/science.ade2574

3. Dauparas, J. et al. Robust deep learning–based protein sequence design using ProteinMPNN. *Science* **378**, 49–56 (2022). https://doi.org/10.1126/science.add2187

4. Dauparas, J., Lee, G.R., Pecoraro, R. et al. Atomic context-conditioned protein sequence design using LigandMPNN. *Nat Methods* **22**, 717–723 (2025). https://doi.org/10.1038/s41592-025-02626-1

---

This work began as a CS540 course project. Released under the MIT License; see `LICENSE`.
