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

## Causal steering — is a feature *used*, or just correlated?

The results above show features **correlate** with concepts. Steering tests the causal claim:
inject a feature into the model's reasoning and see whether its predictions move. TxGNN scores a
drug-disease pair with a **DistMult** decoder (`score = sum(drug * relation * disease)`). We find
the **indication** relation (the one that scores known treatment pairs highest), pick the
**antidiabetic feature** (top-activating drugs = Metformin, Pioglitazone, Sitagliptin, ...), then
inject its decoder direction into drug embeddings and re-score every disease; the mirror test
ablates it from real antidiabetic drugs.

- **Injection:** diabetes/metabolic indication scores rise **+0.189** vs **+0.077** for other
  diseases — **2.5x more**.
- **Ablation:** removing the feature from antidiabetic drugs drops diabetes indications
  **-0.184** vs **-0.075** for others — correct sign, again **~2.5x**.

The feature is **causally wired** to diabetes predictions in both directions. Honest caveat: the
single biggest individual risers are broader endocrine conditions (gynecomastia, aromatase excess
syndrome, breast hypertrophy), so one feature is a **diffuse lever** over a metabolic/endocrine
neighborhood, not a clean diabetes-only knob. This is the relational-graph analogue of InterPLM's
steering check; sharpening specificity with feature combinations is the natural next step.

## Linear probe — do the sparse features actually carry the information?

`fig7_probe.png` + `results/probe_results.json`. We train a **linear probe** (a multinomial logistic
regression, i.e. a simple linear classifier) to predict a node's type from (a) the SAE's sparse
feature code and (b) the raw 512-d embedding, scoring on **held-out** nodes.

- Probe on **SAE features: 99.1%** accuracy vs raw embeddings **99.1%** — statistically identical.
- Per-type recall is ≥0.97 for all ten types from the SAE code.

The point: even though each node's SAE code is **sparse** (only k=32 of 4096 features active), a linear
model reads node type out of it just as well as from the full dense embedding. The SAE reorganizes the
information into an interpretable, sparse basis **without discarding it** — the interpretability comes
at no measurable cost to what the representation encodes.

## Embedding map (UMAP)

`fig6_umap.png` projects TxGNN's 512-d node embeddings to 2-D with UMAP. **Left:** colored by node
type — the ten biological types form clean, separated islands, a visual confirmation that the
embedding space is organized by role. **Right:** the same map with the top-activating nodes of a few
SAE features overlaid; each feature lights up a **tight local region** rather than scattering, which
is what "the SAE picks out coherent directions" looks like geometrically.

## Figures (`figures/`)

- `fig1_top_nodes.png` — representative features with their top-activating nodes named
  (reveals the finer drug-class / disease-group structure).
- `fig2_best_feature_per_type.png` — best feature per node type; the **bright diagonal** shows
  each type has its own dedicated detector. Each row also carries the type's **base rate**, which
  makes the perfect (1.0) purity meaningful: 1.0 for gene/protein (21% of nodes) is only ~5x
  enrichment, whereas 1.0 for exposure (0.7%) is ~136x.
- `fig3_interpretability_summary.png` — **precision and recall shown directly** (enrichment alone
  makes rare types look "best"). Left: every feature has near-perfect precision but low recall
  (each type is split across many features). Right: the cleanest detector per type — precision is
  ~1.0 for all types, while single-feature recall scales inversely with base rate (one feature
  catches ~78% of all pathway nodes but ~0% of the much larger gene/protein set).
- `fig4_discovered_niches.png` — **a catalog of distinct clinical sub-classes the SAE found
  *within* the drug and disease types.** Each card is one feature + example members — drug
  classes (antipsychotics, HIV antiretrovirals, the multiple-myeloma regimen, ...) and disease
  categories (lymphomas/leukemias, botulism, corneal dystrophies, embryonal CNS tumors, ...).
- `fig5_steering.png` — **causal steering.** Left: injecting the antidiabetic feature raises
  diabetes/metabolic indication scores 2.5x more than other diseases (box plot, means marked).
  Right: the largest individual risers, showing the feature is a diffuse endocrine/metabolic lever.

## Run it

Open `txgnn_sae_experiment.ipynb` and run top to bottom. Requirements: `torch`, `numpy`,
`matplotlib`, `gdown`. The notebook downloads TxGNN's pretrained checkpoint (~1.5 GB) from the
authors' Google Drive (contains the trained node embeddings), then trains the SAE and produces
the figures. A GPU helps but isn't required.

## Notes

- The checkpoint (~1.5 GB) and harvested embeddings (~200 MB) are **not committed** — the
  notebook downloads/derives them.
- `results/txgnn_sae_results.json` holds the held-out metrics and per-type detectors.
