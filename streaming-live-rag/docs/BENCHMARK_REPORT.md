# Benchmarking & Evaluation Report

*This file will be updated with actual numbers upon completion of Phase 7 (Ablations & Optimization).*

## Metrics Reference

| Metric | Target | Measured Result |
| :--- | :--- | :--- |
| **G1 Reproducibility** | Pass/Fail, unattended | TBD |
| **G2 Early retrieval rate** | ≥ 80% of eligible cases | TBD |
| **G2 False-trigger rate (no-retrieval cases)** | As low as achievable, reported explicitly | TBD |
| **G3 Multi-intent identification** | ≥ 70% of compound cases | TBD |
| **Over-fragmentation rate (single-question controls)** | Team-set bar, reported explicitly | TBD |
| **G4 Citation support** | ≥ 85%, zero fabricated IDs | TBD |
| **G5 Session continuity** | 100% of refinement/suppression cases behave correctly | TBD |
| **G6 Telemetry field coverage** | 100% | TBD |
| **Retrieval recall@k (hybrid vs. dense-only)** | Hybrid ≥ dense-only, or a documented reason why not | TBD |
| **P95 retrieval+rerank latency**| ~60–70% of total per-turn budget | TBD |
| **Time-to-first-token** | As low as your LLM provider allows; tracked every phase, optimized Phase 8 | TBD |
| **Cost per turn (token estimate)**| Tracked from Phase 6 telemetry onward, minimized Phase 8 | TBD |

## Ablation Results

### Ablation #1: Hybrid vs. Dense-Only Retrieval
*(Draft numbers to be added in Phase 2, finalized in Phase 7)*

| Approach | Recall@K | Hit-Rate | Latency |
| :--- | :--- | :--- | :--- |
| Dense-Only | TBD | TBD | TBD |
| Hybrid + Rerank | TBD | TBD | TBD |

### Ablation #2: Rule-Based vs. Model-Based Controller
*(To be completed in Phase 7)*

| Approach | G2 (Early Retrieval) | False-Trigger Rate |
| :--- | :--- | :--- |
| Heuristics-Only | TBD | TBD |
| Heuristics + LLM Classifier| TBD | TBD |

## Documented Edge Cases
*(To be completed in Phase 7)*
1. TBD
2. TBD
3. TBD
