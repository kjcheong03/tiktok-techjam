# Baseline comparison - 200 public sessions

## Configuration

- Keyword: organizer SQLite FTS5/BM25 retriever.
- Dense: `sentence-transformers/all-MiniLM-L6-v2`, exact cosine search on CPU.
- State: deterministic slot accumulation with provenance and intent-override invalidation.
- Hybrid: top-200 keyword and dense rankings fused with RRF (`k=60`), without tuning.
- All variants use the unchanged organizer evaluator with at most 10 turns and 10 recommendations.
- Evaluation seconds are warm-cache diagnostics and are not a controlled speed benchmark.

Reproduce with: `uv run python -m scripts.run_baselines`

## Overall results

| Variant | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore | Eval seconds |
|---|---:|---:|---:|---:|---:|---:|
| official_keyword | 0.125000 | 0.068034 | 9.810000 | 0.119000 | 0.106710 | 10.681 |
| keyword_current_turn | 0.520000 | 0.358060 | 6.990000 | 0.401000 | 0.447618 | 3.229 |
| keyword_state | 0.595000 | 0.377867 | 6.375000 | 0.462500 | 0.503360 | 3.723 |
| dense_current_turn | 0.295000 | 0.164768 | 8.690000 | 0.231000 | 0.243130 | 1.655 |
| dense_state | 0.340000 | 0.146607 | 8.165000 | 0.283500 | 0.270682 | 1.651 |
| hybrid_current_turn | 0.480000 | 0.243167 | 7.115000 | 0.388500 | 0.390650 | 0.271 |
| hybrid_state | 0.515000 | 0.256236 | 6.795000 | 0.420500 | 0.418471 | 0.529 |

## Scenario Technical Inputs

### official_keyword

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.000000 | 0.000000 | 11.000000 |
| browsing | 80 | 0.025000 | 0.004514 | 10.750000 |
| buying | 80 | 0.237500 | 0.126508 | 8.625000 |
| intent_override | 30 | 0.133333 | 0.104167 | 10.066667 |

### keyword_current_turn

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.500000 | 0.269444 | 8.400000 |
| browsing | 80 | 0.550000 | 0.394945 | 6.900000 |
| buying | 80 | 0.550000 | 0.358294 | 6.187500 |
| intent_override | 30 | 0.366667 | 0.288611 | 8.900000 |

### keyword_state

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.500000 | 0.325000 | 7.800000 |
| browsing | 80 | 0.712500 | 0.480069 | 5.462500 |
| buying | 80 | 0.512500 | 0.294747 | 6.737500 |
| intent_override | 30 | 0.533333 | 0.344603 | 7.366667 |

### dense_current_turn

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.100000 | 0.014286 | 10.500000 |
| browsing | 80 | 0.375000 | 0.228487 | 8.162500 |
| buying | 80 | 0.287500 | 0.150397 | 8.525000 |
| intent_override | 30 | 0.166667 | 0.083333 | 9.933333 |

### dense_state

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.100000 | 0.011111 | 10.400000 |
| browsing | 80 | 0.500000 | 0.219474 | 7.000000 |
| buying | 80 | 0.237500 | 0.083155 | 8.675000 |
| intent_override | 30 | 0.266667 | 0.166667 | 9.166667 |

### hybrid_current_turn

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.400000 | 0.082500 | 8.400000 |
| browsing | 80 | 0.562500 | 0.286012 | 6.562500 |
| buying | 80 | 0.500000 | 0.249092 | 6.537500 |
| intent_override | 30 | 0.233333 | 0.166667 | 9.700000 |

### hybrid_state

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| boundary | 10 | 0.500000 | 0.256944 | 7.100000 |
| browsing | 80 | 0.662500 | 0.345332 | 5.737500 |
| buying | 80 | 0.400000 | 0.181126 | 7.312500 |
| intent_override | 30 | 0.433333 | 0.218704 | 8.133333 |

