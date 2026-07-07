**Class balance is the bottleneck, not total volume**

500 units sounds substantial until class balance is considered. At a typical production failure rate of ~2%, the dataset would contain only ~10 failed units — far too few for supervised classification. Two things change this picture:
*	**Continuous damping curves instead of binary pass/fail.** Regression on force values is far more data-efficient (see Section 6). Full force–velocity curves from the test rig should be requested rather than only the verdict.
*	**Deliberately produced out-of-spec units.** 30–50 intentionally deviated units are worth more than 500 additional good ones.

**Expect the unglamorous problems to dominate early**

Budget real time for an exploratory data analysis (EDA) phase before committing to a modeling approach. Typical issues in this class of project:
*	Scans that fail to register to CAD; inconsistent fixturing between scans.
*	Missing regions caused by reflective chrome surfaces and occlusions.
*	ID mismatches between scan files and test rig records.
*	Multiple product variants mixed together in the dataset.

**How Much Labeled Data Do Tier 3 Models Need?**

The honest answer is “it depends on the task and the richness of the label signal”, but defensible orders of magnitude can be stated. These are heuristics from practice and literature, not hard laws; the real answer comes from a learning curve on the actual data.

* **Binary pass/fail classification (trained from scratch on point clouds):**	Absolute minimum for anything to work: ~1,000–2,000 units including at least 100–200 defective units — the minority class is the bottleneck, not the total. Comfortable: 5,000–10,000 units with several hundred defects. With 500 units and ~2% failure rate (~10 defects), deep classification from scratch is practically doomed: the model will learn noise, and validation will have enormous variance (a single defective unit in the test set moves recall by double-digit percentage points).
* **Regression on damping force (continuous values):**	Much more favorable, because every unit carries information, not only the defective ones: sensible results are sometimes achievable from ~300–500 units, good results from 1,000–2,000. This is the main argument for insisting on full test-rig curves rather than pass/fail verdicts.
* **Segmentation / per-point defect localization:**	The most data- and annotation-hungry: hundreds of units with annotated defect regions, which the Customer almost certainly does not have and cannot produce cheaply. Realistically out of reach in the supervised variant for this project; localization comes “for free” from deviation maps and anomaly detection methods.

**Levers that shift these numbers downward**
1.	**CAD deviation as an input feature.** If the network receives per-point deviation from nominal instead of raw coordinates, it does not have to learn nominal geometry — the task becomes an order of magnitude easier. This can pull requirements from thousands down to hundreds of units.
2.	**Self-supervised pretraining (Point-MAE, PointBERT) on unlabeled scans.** If the Customer scans more units than it tests on the rig (common), pretraining on 5,000–10,000 unlabeled scans plus fine-tuning on 500 labeled ones is the best-case scenario for Tier 3.
3.	**2D projection + pretrained backbone. Cylindrical unwrapping of the deviation map + a CNN/ViT pretrained on ImageNet** — transfer learning in 2D is far more mature than in 3D and fine-tunes sensibly on 300–500 images. Quietly the cheapest route to “deep learning” within this data budget.
4.	**Augmentations (axis rotations, scanner-scale jitter, simulated coverage gaps)** effectively multiply the dataset several-fold, but cannot substitute for missing defective units.
5.	**Deliberately out-of-spec units:** 30–50 such units are worth more than 500 additional good ones.

**Recommended framing when the question is asked**


“With 500 units and a natural failure rate, advanced 3D models only make sense as regression on damping curves or with pretraining on additional unlabeled scans. Supervised pass/fail classification from scratch would require a minimum of 100–200 documented defects — i.e., 5,000–10,000 tested units at a typical failure rate. The base plan is therefore the feature baseline and anomaly detection, which work at current volume, with Tier 3 as a data-conditional experiment. After the first data tranche we will present a learning curve showing whether more data genuinely helps.”

**Commit to a learning curve rather than a single number:** train on 25/50/75/100% of available data and plot the metric. This is the professional answer to “how much is needed” — it shows empirically whether the metric is still rising with data, gives the Customer a hard basis for deciding whether to collect more, and avoids committing to a number the team could later be held to.

**Standing caveat:** no amount of data helps if the geometry carries no signal about damping. “How many examples” is only meaningful after a positive answer to “is there information in the scans at all” — which is exactly what the Tier 1 baseline resolves most cheaply.
