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

## Next product gates

1. Apache RTMDet-R -> ONNX parity.
2. Multi-domain lawful training without touching external H01.
3. Persistent identity service and temporal state engine.
4. Event engine + DecisionObject compiler.
5. API/CLI + workers + storage.
6. Docker distribution and demo missions.
7. Enterprise IAM/tenancy/observability.
