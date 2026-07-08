Model Quality Metrics and Evaluation Protocol

*	**Classification:** PR-AUC and recall at a fixed false-positive rate. The operating point must be agreed with the Customer — the cost of an escaped defect vs. unnecessary scrap/re-inspection is a business decision, not the Supplier's to make.
*	**Regression:** MAE / RMSE on damping force plus Spearman correlation; additionally “regression→classification” via specification thresholds.
*	**Anomaly detection:** unit-level AUROC; if localization is delivered, point-level PRO/AUPRO. A protocol is needed for validating flagged units when labels are absent.
*	**Data quality metrics** (the project requires documented metrics — cover both layers): registration error, scan coverage, point count, share of rejected scans.
*	**Evaluation protocol:** split by production time or batch, never randomly. Manufacturing data drifts, and a random split yields inflated, leaky results that collapse in Phase 2 validation.

**Preprocessing Pipeline and Infrastructure**

Format parsing → registration/alignment to CAD or a reference frame → cropping to regions of interest → normalization and downsampling → caching in an ML-friendly format (compressed tensors, Zarr, or WebDataset — do not train directly off raw PLY files at gigabyte scale).
*	**Data validation gates** (point-count sanity checks, registration-error thresholds) so that bad scans are quarantined rather than silently ingested into training.
*	**Data versioning** (DVC or lakeFS) from day one.

**Infrastructure and reproducibility**

*	**Experiment tracking** (MLflow or Weights & Biases).
*	**Containerized environments with lockfiles and seeded runs** — the project explicitly promises reproducibility, so it should be built in rather than retrofitted.
*	**Streaming data loaders with pre-computed caches** to satisfy the gigabyte-scale requirement.
