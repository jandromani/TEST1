# AssetGraph Core

Persistent Asset Intelligence runtime promoted from the AssetGraph evidence lab.

## Product boundary

AssetGraph Core is **not** a detector and is **not** a benchmark harness. It is the stable layer that turns sensor observations into persistent asset hypotheses, events, evidence-backed decision objects, and institutional memory.

```text
Sensor / Provider
  -> Adapter
  -> Observation
  -> Perception Runtime
  -> Asset Resolution
  -> Temporal State
  -> Event / Claim
  -> DecisionObject
  -> Evidence Ledger / Analyst Review
```

## Promotion rule

Experimental Cycle code stays in the lab. A capability may enter this tree only when it has an explicit evidence state and scope. FAIL evidence is retained and may block promotion; it is never silently removed.

## Current promoted capabilities

- Domain contracts for Observation, AssetHypothesis, Event, Claim and DecisionObject.
- Pluggable perception runtime interface.
- Runtime registry with explicit license/product status.
- Claim governance and evidence references.
- Evidence hashing / replay manifest primitives.
- Dataset/model/capability registries.

## Current evidence status

- PETS real RGB -> identity -> world: component E2E PASS.
- SeaDronesSee person perception: PASS on frozen stress scope.
- Identity Memory v1: measured lockbox uplift.
- UAV-OBB learning: learned capability proven internally.
- TRANSSET H00 zero-shot external generalization: FAIL and consumed diagnostic.
- Cycle 16 small/rare recovery: improved internal capability but promotion FAIL because small recall 0.266 < 0.30.
- Apache RTMDet-R native runtime probe: PASS.
- Apache RTMDet-R raw PyTorch -> ONNX Runtime parity: PASS on all nine tensors.
- Apache RTMDet-R final detection parity through one shared decoder and rotated NMS: PASS, 201/201 matched with minimum rotated IoU 0.999989.
- Cycle 18A lawful multidomain intake: PASS. UAV-OBB, HIT-UAV and SeaDronesSee are admitted to verified acquisition; no dataset is yet admitted to the new training queue.
- AU-AIR commercial intake: QUARANTINED because embedded noncommercial license entries conflict with the public CC-BY claim.

The Apache path is now **transport validated**, not accuracy promoted. The official DOTA checkpoint remains evaluation-only; product weights, multidomain accuracy, dynamic shapes and tiled inference are still open gates.

## Next product gates

1. Cycle 18B: acquire only the approved queue; freeze archive/member hashes and complete leakage, annotation, privacy and taxonomy audits.
2. Cycle 18C: train product-candidate Apache RTMDet-R weights and pass internal multidomain gates.
3. Open H01 once only after that internal promotion pass.
4. Replace the MMRotate-backed shared postprocess with a lightweight independently tested production implementation.
5. Persistent identity service, temporal/event engine and measured Intelligence Memory.
6. API/workers, operational UI, replay/provenance unification and durable evidence release.
7. Enterprise IAM, tenancy, observability and security gates.
