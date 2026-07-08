The strategy is tiered from simplest to most advanced. Each tier provides a benchmark that the next must beat; a delivered and documented Tier 1 baseline is worth more than a half-working advanced model.

**Tier 1 — Engineering Baseline (start here)**

Applicable when nominal CAD models are available:
*	**Deviation maps:** register each scan to CAD (ICP / Generalized-ICP / FGR for coarse alignment), then compute point-to-surface distances. Result: a per-point deviation value from nominal geometry.
*	**Statistical features per region:** mean / max / percentile deviation, area out of tolerance, distribution moments — computed for defined part zones (e.g., body, thread, piston rod).
*	**Model:** gradient boosting (XGBoost / LightGBM / CatBoost) on these features; logistic regression as a sanity check.

**Advantages:** interpretability (quality engineers can see why a unit was flagged), trains comfortably on 500 samples, and provides the benchmark any deep model must beat. It is also the cheapest and fastest way to answer the fundamental question of whether geometry carries signal at all. Tooling: Open3D, PyVista, trimesh, point-cloud-utils.

**Tier 2 — Unsupervised / Low-Label Anomaly Detection**

Appropriate when defective units are scarce:

*	**Memory-bank methods (PatchCore-style) adapted to 3D:** build a bank of local features from good units; anomaly score = distance to nearest neighbors in the bank. Relevant literature: BTF (“Back to the Feature”, a surprisingly strong FPFH-based baseline), M3DM, Shape-Guided, and teacher–student methods such as 3D-ST. Reference benchmarks: MVTec 3D-AD and Real3D-AD — their leaderboards are worth reviewing, as simple methods often beat deep networks there.
*	**Point cloud autoencoders (FoldingNet, PointNet-based autoencoders):** reconstruct good geometry, use reconstruction error as the anomaly score. Workable but can be temperamental.
*	**2D projection of deviation maps:** unwrap/project the part surface to 2D — the shock absorber is roughly axisymmetric, so a cylindrical unwrapping is natural — and apply mature 2D anomaly detection (PatchCore, PaDiM, EfficientAD via the anomalib library). Often the most pragmatic path: mature tooling, cheap GPU requirements, and easy heatmap visualization for engineers.

**Tier 3 — Deep Learning on 3D Data (Phase 2, if justified)**

For supervised prediction or richer representations:

*	**Point-based networks:** PointNet++ (classic, good starting point), DGCNN, Point Transformer v2/v3, PointMLP. Input: 2,048–8,192 sampled points, ideally with the CAD deviation as an additional per-point feature — this helps enormously, since the network no longer has to learn nominal geometry itself.
*	**Sparse-voxel networks:** MinkowskiEngine, spconv — efficient for large clouds but with higher engineering overhead.
*	**Multi-view projections:** render the part from several angles and apply a standard CNN/ViT. Underrated, often competitive, and much cheaper.
*	**Self-supervised pretraining:** Point-MAE, PointBERT — with only 500 labeled units, pretraining on the Customer's unlabeled scans (if more units are scanned than tested) can genuinely help.
*	**Domain-specific augmentations:** random rotations around the part axis, point jitter at scanner-noise scale, simulated coverage gaps — but not geometric deformations, since those are precisely the signal to be detected.

**Practical guidance:** with 500 units and regression on damping force, the realistic ceiling is probably Tier 1–2 plus a light point-based model. Large 3D transformers should be treated as an experiment, not the base plan.
