# Compact Cross-Encoder Challenger V1

Date: 2026-08-26
Branch: `exp/cross-encoder`
Parent champion: `189f0c6338e2d2ec1a795dce543e881ff2037f2a`
Split: frozen 150-session `nested_v1`
Protected holdout: sealed and not accessed

## Decision

Park the tested zero-shot cross-encoder standalone. Do not add its model asset or
runtime path to integration. The best fixed candidate improved the label-free
field+quality scorer slightly but did not beat the actual fold-specific linear
champion, and nested candidate selection was significantly worse than that champion.

## Validity corrections before evaluation

An independent pre-run audit stopped the first launch before a cross-encoder
candidate completed. The final experiment then:

- used stored fold-specific linear OOF sessions as the primary paired benchmark;
- excluded those OOF control predictions from fold-wise candidate selection to
  prevent cross-fold leakage;
- predeclared all four Top-20 weights and one fixed Top-50 candidate;
- separated uncached neural timing from shared-score-cache policy replay;
- froze `catalog_fields_v2`, which puts title, category, price, store, features, and
  details before the bounded description tail;
- pinned the model and exact revision and ran with local files only.

The passage audit covered all 50,000 products. Median document length was 153
tokens, p95 was 216, and 3.846% exceeded the conservative 220-document-token
allowance. Important short fields occur first, so model truncation affects the tail.

## Results

| Policy | Hit@10 | MRR | MTTC | Technical score | Delta vs linear OOF |
|---|---:|---:|---:|---:|---:|
| Two-feature linear champion OOF | 0.933333 | 0.631720 | 2.926667 | 0.817649 | — |
| Fixed field + quality | 0.913333 | 0.630860 | 3.266667 | 0.800591 | -0.017058 |
| Cross-encoder Top-20, weight 0.10 | 0.913333 | 0.638397 | 3.266667 | 0.802852 | -0.014797 |
| Cross-encoder Top-20, weight 0.25 | 0.913333 | 0.633735 | 3.213333 | 0.802520 | -0.015129 |
| Cross-encoder Top-20, weight 0.50 | — | — | — | 0.801564 | -0.016085 |
| Cross-encoder Top-20, weight 1.00 | — | — | — | 0.780905 | -0.036744 |
| Cross-encoder Top-50, weight 0.25 | — | — | — | 0.798714 | -0.018935 |

Fold-wise selection chose a cross-encoder in two of five folds, but the stitched
policy scored `0.790915` versus champion OOF `0.817649`. Its paired delta was
`-0.026734`, bootstrap 95% interval `[-0.047852, -0.007315]`, and randomization
`p=0.008198`. The worst selected outer fold scored `0.726732`, so this is a clear
stability failure rather than a complexity tie.

## Runtime and packaging

The exact pinned asset occupied 91,821,247 unique bytes. Initialization including
50,000 bounded passages took 2.441 seconds and peak process memory was 2,346.625 MB.
The uncached Top-20 reference had p95 turn latency 166.195 ms; Top-50 had 226.238
ms. These remain within the checkpoint budgets, with zero observed runtime failures,
but runtime feasibility cannot compensate for the negative relevance evidence.

The model was selected from the official Sentence Transformers pretrained
cross-encoder documentation and its official Hugging Face model card:

- <https://www.sbert.net/docs/cross_encoder/pretrained_models.html>
- <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2>

Detailed sessions, folds, paired tests, scenario metrics, hashes, asset inventory,
and timing references are stored in
`artifacts/reports/challenger_cross_encoder_v1.json`.
