# AssetGraph Cycle 07 protocol

Frozen before scoring.

## Gate A — Selective Memory v4
- Development bank: the 22 DUT missions already exposed in v1-v3.
- Final lockbox: intersection_10, intersection_12, intersection_13, intersection_14, intersection_16, intersection_17. These six clips were not previously scored by AssetGraph.
- Only first 35% of each mission is visible to the query representation.
- Per-target memory is promoted only when development leave-one-mission-out CV improves >=10% and does not degrade >5%. Otherwise the target abstains to baseline.
- Final PASS: at least one target improves >=10% on the six-clip lockbox and no reported target degrades >5%.

## Gate B — Overhead Training Factory
- Dataset candidate: UAV-OBB 2026, CC BY 4.0, fixed published train/valid/test split.
- Fine-tune a small OBB detector only on train, select checkpoint/threshold only on validation, score the published test once.
- Training data license, checkpoint hash, source URL, hyperparameters and runtime are evidence.
- Initial technical PASS: binary vehicle F1 >=0.60 on test and count MAPE <=15%. Promotion to PRODUCT_CANDIDATE separately requires license/provenance/runtime gates.

## Gate C — SeaDronesSee native MOT
- Acquire a real train/validation MOT micro-sequence with native source mapping and persistent IDs if publicly accessible.
- No identity continuity may be inferred from OD frame adjacency alone.
- If native GT cannot be acquired without the 29.6GB archive, produce an acquisition manifest and do not claim MOT PASS.

No acceptance threshold may be changed after a lockbox result is observed.
