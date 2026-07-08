**Kick-off Meeting Questions**

**Data and scanning**

1.	What device produces the scans (structured light such as GOM/Zeiss, laser, CMM)? Exact format, resolution, and typical point count per scan? Does a proprietary format require an SDK?
2.	Are nominal CAD models available for each component and variant, and can they be provided? Are scans pre-registered to CAD, or is registration the Supplier's responsibility?
3.	How many components per unit are scanned, and which ones? Is the unit scanned assembled or as individual parts?
4.	How many product variants exist — one model or per-variant models?
5.	What known scan-quality issues exist (occlusions, reflective surfaces, missing patches), and what is the rejection/rescan policy?
6.	What is the exact ID scheme linking scan file ↔ unit ↔ test result, and who guarantees its integrity? Request 5–10 real sample units end-to-end before the pipeline design is finalized.
7.	Total data volume, transfer mechanism (S3, SFTP, on-prem share), and any restrictions on moving data outside the Customer's infrastructure?

**Labels and the quality process**

1.	Full damping force–velocity curves, or only pass/fail? What are the thresholds, and do they vary by variant or end-customer specification?
2.	Historical failure rate — how many failed units will actually be in the 500?
3.	What is the repeatability of the test rig itself (gauge R&R)? If label noise is large relative to the effect being detected, it caps achievable accuracy.
4.	What fraction of damping failures are believed to be geometry-related vs. caused by internals (oil, valves, seals) invisible to the scan? Are there root-cause analysis records?
5.	Can out-of-spec or deliberately deviated units be produced or provided to enrich the failure class?

**Problem definition and success criteria**

1.	What decision does the model output drive — sorting units out, triggering re-inspection, process feedback to machining? This determines the acceptable false-positive rate.
2.	What does “suitable for technical validation” mean in Phase 2, concretely? Agree measurable acceptance criteria (e.g., detect X% of failures at ≤Y% false alarms on a held-out time period) or explicitly document that Phase 2 is exploratory.
3.	What is the current baseline — existing tolerance checks, SPC, manual inspection — and what does it catch? The model needs to beat something specific.
4.	Inference latency and throughput expectations: inline within the production takt time, or offline batch?

**Infrastructure and operations**

1.	Where does training run — Customer cloud, Supplier environment, on-prem? What GPU resources are available or budgeted?
2.	Is there an existing MLOps/monitoring stack to integrate with, or is this greenfield?

**Process**

1.	Data delivery schedule — everything up front or in tranches? (Tranches by production date are actually preferable — they allow drift testing.)
2.	Who is the subject-matter contact among the quality engineers, and how much of their time is available? Their involvement is needed continuously for interpreting deviations.
3.	What happens if the scanning process or fixturing changes mid-project? Agree a change-notification protocol.

**Key Risks to State Openly at Kick-off**

These should be put on the table early so they become shared risks rather than the Supplier's later surprises:
1.	**Weak geometry signal:** geometry may carry weak signal for damping outcomes (internal components dominate).
2.	**Class imbalance:** 500 units with few failures may only support anomaly detection, not supervised classification.
3.	**Label noise from the test rig** (repeatability caps achievable model accuracy).
4.	**Distribution shift** between the historical dataset and future production.

The project already contains a risk-acceptance mechanism (“accept the associated risks”) for a smaller dataset; the same risk-acceptance logic should be documented for class imbalance, not only dataset size.
