# Data and Model Attribution

## Competition data

This competition package is derived from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), published by McAuley Lab at UCSD.

- Selected category: `Clothing_Shoes_and_Jewelry`
- Product join key: `parent_asin`
- Competition modality: text and structured product metadata only

The competition package does not contain images, videos, account credentials, private
organizer labels, or the private holdout sessions.

Participants must follow the source dataset's applicable terms and use the data only
for the competition, research, and other permitted purposes. The competition organizer
does not claim ownership of the underlying Amazon review or product content.

## Third-party model assets

GhostLab does not commit or redistribute third-party model weights. Its setup and
optional-asset scripts download models from their publishers at the pinned revisions
recorded below and verify local assets before use. Each model remains subject to its
publisher's licence and usage terms.

### Published champion and complete optimizer

| Model | Publisher | GhostLab role | Pinned revision | Licence |
|---|---|---|---|---|
| [`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2) | intfloat | Dense semantic retrieval and the E5 release index | `ffb93f3bd4047442299a41ebb6fa998a38507c52` | [MIT](https://opensource.org/license/mit) |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Sentence Transformers | MiniLM dense-retrieval alternative and release index | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | Sentence Transformers | Bounded cross-encoder fallback and optimizer alternative | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | Hugging Face | Selected bounded semantic ranker for the GhostLab Champion | `31b70e2e869a7173562077fd711b654946d38674` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

### Historical development candidates

These models were evaluated during local-LLM selection but are not used by the
published champion and are not required by the clean setup path.

| Model | Publisher | Evaluation role | Pinned revision | Licence |
|---|---|---|---|---|
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Qwen / Alibaba Cloud | Local semantic-ranker candidate | `7ae557604adf67be50417f59c2c2f167def9a775` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Qwen / Alibaba Cloud | Local semantic-ranker candidate | `c1899de289a04d12100db370d81485cdf75e47ca` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) | Google DeepMind | Gated local semantic-ranker candidate | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

Gemma access requires accepting Google's terms through the model publisher. GhostLab
does not distribute Gemma weights or a Gemma-derived model.

## Verification records

Machine-readable model names, destinations, pinned revisions, licences, and available
file hashes are stored under `configs/assets/`. The dense-index release manifest is
`configs/assets/dense_indexes_50k_v1.json`.
