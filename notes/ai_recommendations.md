Core recommendations:
1.	**Adopt a hybrid strategy**: geometric anomaly detection against nominal geometry as the robust baseline, with supervised prediction of damping outcomes layered on top if — and only if — the geometry-to-label signal is confirmed to exist.
2.	**Prefer regression on full damping force–velocity curves over binary pass/fail classification.** With ~500 labeled units and a low natural failure rate, regression is feasible while supervised classification is not. The final deliverable is still a defect detector: predicted values are compared against specification thresholds.
3.	**Validate the fundamental assumption early**: damping force is largely determined by internal components (valve stacks, oil fill, seals) that a 3D surface scan may not capture. Whether scanned geometry carries signal about test outcomes is the single biggest project risk and should be pressure-tested at kick-off and resolved cheaply with the feature-based baseline.
4.	**Commit to learning curves rather than fixed sample-size promises** when asked how much labeled data advanced models require. Train on 25/50/75/100% of available data and let the metric curve answer empirically whether more data helps.
5.	**Split data by production time or batch, never randomly.** Manufacturing data drifts; random splits produce flattering, leaky results that collapse in Phase 2 validation.

“Identifying deviations from production standards” can mean three quite different machine learning problems. Resolving which one(s) the project targets is the first order of business, because they differ in data requirements, methods, and deliverables.
1. **Supervised prediction of quality outcomes:**	Predict damping force (regression) or pass/fail (classification) from 3D geometry. This is what the project labels suggest, but it carries a hidden risk (below).
2. **Geometric anomaly detection:**	Compare each scan against nominal CAD geometry or a “golden” distribution of good units and flag deviating units, regardless of test outcome. Requires little or no labeled failure data.
3. **Localized defect detection:**	Identify where on the part the deviation occurs (dents, weld issues, dimensional drift). Requires per-region ground truth or at least reliable CAD registration; supervised variants are likely out of reach for this project.

**The hidden risk in supervised prediction**

**Damping force is largely determined by internal components — valve stacks, oil fill and viscosity, seals — that a 3D surface scan may not capture at all.** If the scanned geometry is not causally linked to the damping test outcome, no model will find signal, and this would otherwise only be discovered months into the project. This is the most important assumption to pressure-test at kick-off: which components are scanned, and does the quality engineering team believe their geometry actually drives test results? Historical evidence should be requested — e.g., documented cases where a dimensional issue caused a damping failure.

**Recommended strategy: hybrid**

Plan for anomaly detection against nominal geometry as the robust baseline — it always produces something useful for quality control — with supervised prediction layered on top if the geometry-to-label signal exists. The two act as complementary sensors: the regressor captures deviations that manifest in damping force; the geometric anomaly detector captures deviations that do not (cosmetic or purely dimensional defects, or issues that manifest as leaks or noise rather than damping).
