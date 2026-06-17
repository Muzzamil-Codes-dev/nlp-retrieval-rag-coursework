# Information Retrieval & RAG on Natural Questions

This project builds an end-to-end information retrieval and Retrieval-Augmented Generation (RAG) system for the Natural Questions dataset. It starts from a sparse TF-IDF baseline, progressively improves retrieval with dense embeddings, hybrid fusion, query expansion, and cross-encoder re-ranking, then plugs the best retrieval setup into a full RAG pipeline with error analysis and a held-out test-set submission.

## Project overview

| Stage | What it does |
|---|---|
| **Q1 — TF-IDF baseline** | Sparse retrieval with unigrams + bigrams, stop-word removal, L2-normalised vectors |
| **Q2 — Dense retrieval** | Bi-encoder retrieval with `all-MiniLM-L6-v2`, compared against the larger `all-mpnet-base-v2` |
| **Q3 — Advanced retrieval** | Four improvements, each tuned on the dev set: Reciprocal Rank Fusion, weighted score fusion (α), cross-encoder re-ranking, and LLM-based query expansion with Flan-T5 |
| **Q4 — RAG pipeline** | FAISS vector database (`IndexFlatIP`) + best Q3 retrieval/re-ranking setup + Flan-T5-small answer generation, evaluated with Exact Match and Token F1 |
| **Q5 — Error analysis** | Qualitative analysis of failure cases and final evaluation, ending with the held-out test-set submission |

## Key techniques

- **Sparse retrieval:** TF-IDF with cosine similarity
- **Dense retrieval:** Sentence-Transformers bi-encoders with normalised embeddings
- **Hybrid retrieval:** Reciprocal Rank Fusion and normalised weighted score fusion of sparse + dense rankings
- **Re-ranking:** Cross-encoder applied to the top-N candidates for a precision boost at manageable cost
- **Query expansion:** Flan-T5-small generates query elaborations, blended with the original query via a tuned β weight
- **Vector database:** FAISS index built, saved to disk, and reloaded to avoid re-encoding the corpus
- **Generation:** Flan-T5-small answers questions from the top retrieved passages

## Repository contents

| File | Description |
|---|---|
| `Rasully_Muzzamil_210927089.ipynb` | Main notebook — run top to bottom |
| `Muzzamil Rasully ECS665U 210927089 Report.pdf` | Written report |
| `LLM Assisted Retrieval Improvements for NLP Coursework - DSeek.pdf` | Supporting document on LLM-assisted retrieval improvements |
| `submission.csv` | Held-out test-set submission |
| `q5_error_analysis_selected_cases.csv` | Selected cases used in the Q5 error analysis |
| `rag_k_sweep.png` | RAG performance sweep over the number of retrieved passages (k) |
| `similarity_heatmap.png` | Query–passage similarity heatmap |
| `faiss_minilm_*.index` | Cached FAISS indexes (validation and test corpora) |
| `passage_embeddings_*.npy` | Cached passage embeddings for the bi-encoders |
| `query_elaborations.npy` | Cached Flan-T5 query elaborations |

The cached embeddings, FAISS indexes, and query elaborations let later notebook runs skip the expensive encoding/generation steps.

## Running the notebook

The notebook is designed to run top to bottom (originally in Google Colab). Main dependencies:

```
sentence-transformers
faiss-cpu
transformers
scikit-learn
numpy
pandas
```

Cached artifacts in this repo are picked up automatically, so a re-run does not need to re-encode the corpus or regenerate query elaborations.
