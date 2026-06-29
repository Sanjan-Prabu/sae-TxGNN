# SAE on TxGNN — interpreting a drug-repurposing graph neural network

A step toward interpreting graph-based drug-repurposing models with **sparse autoencoders
(SAEs)**. We apply an SAE to **TxGNN** (Huang et al., *Nature Medicine* 2024), a pretrained
graph neural network that predicts drug-disease relationships over the **PrimeKG** biomedical
knowledge graph. This is the relational-graph analogue of our earlier molecular-GNN validation.

## What this is

- **Model:** TxGNN's **R-GCN** (relational graph neural network) — message-passing over a
  biomedical knowledge graph, producing a 512-dim **node embedding** for every node. Frozen;
  we only train the SAE. We use the embeddings bundled in the authors' pretrained checkpoint.
- **Method:** train a **TopK SAE** on the node embeddings, then check each learned feature's
  top-activating nodes against the **node type** (drug / disease / gene-protein / pathway /
  biological-process / ...) as an independent answer key, via **purity** and **enrichment**.
- **Cross-validation:** features are validated on a **held-out** set of nodes the SAE never
  trained on.

### Key terms
- **SAE feature** — one learned, unsupervised "detector"; fires on some nodes, off for most.
- **Purity** — fraction of a feature's top-activating nodes sharing one node type.
- **Enrichment ("Nx")** — purity divided by that type's base rate; how many times above chance.
- **Held-out** — validated on nodes excluded from SAE training (generalization).

## Results (held-out nodes)

129,312 nodes (512-dim) across 10 biological types; trained on 80%, validated on 20% held out.

- **Reconstruction:** held-out **FVU ~0.016** (~98% of embedding variance explained).
- **Interpretability:** of ~1,350 active features, **~1,130 are >=0.70 pure**, **~1,080 >=0.80**,
  **~1,020 >=0.90** for a node type.
- **Every node type has a dedicated, ~100%-pure feature**, each strongly enriched over chance:
  exposure ~136x, pathway ~51x, cellular_component ~31x, drug ~17x, molecular_function ~11x,
  anatomy ~9x, disease ~8x, gene/protein ~5x.
- **Finer than node type (qualitative):** individual features organize within a type — e.g. one
  drug feature's top drugs are all **antidiabetics** (Pioglitazone, Repaglinide, Insulin
  glulisine, Empagliflozin...), and disease features split into **neurological** vs
  **infectious/inflammatory** groups (Figure 1). A systematic, deduplicated scan surfaced
  **dozens of distinct, clinically coherent sub-classes** (Figure 4) — confirming the embedding
  space is organized far below the node-type level.

**Takeaway:** TxGNN's embedding space is cleanly organized by biological role, and SAE features
recover it. Node type is validated quantitatively; finer drug-class / disease-category structure
is visible qualitatively and is the natural next validation step (using ontology labels).

## Figures (`figures/`)

- `fig1_top_nodes.png` — representative features with their top-activating nodes named
  (reveals the finer drug-class / disease-group structure).
- `fig2_best_feature_per_type.png` — best feature per node type; the **bright diagonal** shows
  each type has its own dedicated detector, with enrichment in the axis labels.
- `fig3_interpretability_summary.png` — interpretability tiers + enrichment per node type.
- `fig4_discovered_niches.png` — **a catalog of distinct clinical sub-classes the SAE found
  *within* the drug and disease types.** Each card is one feature + example members — drug
  classes (antipsychotics, HIV antiretrovirals, the multiple-myeloma regimen, ...) and disease
  categories (lymphomas/leukemias, botulism, corneal dystrophies, embryonal CNS tumors, ...).

## Run it

Open `txgnn_sae_experiment.ipynb` and run top to bottom. Requirements: `torch`, `numpy`,
`matplotlib`, `gdown`. The notebook downloads TxGNN's pretrained checkpoint (~1.5 GB) from the
authors' Google Drive (contains the trained node embeddings), then trains the SAE and produces
the figures. A GPU helps but isn't required.

## Notes

- The checkpoint (~1.5 GB) and harvested embeddings (~200 MB) are **not committed** — the
  notebook downloads/derives them.
- `results/txgnn_sae_results.json` holds the held-out metrics and per-type detectors.
