# page66_72 RAG explanation

This note follows slides 66-72 of [AI_Full_Curriculum_EN.pptx.pptx](/Users/aarunyan/Documents/project/ss6_guidebook_new/AI_Full_Curriculum_EN.pptx.pptx): RAG pipeline, sparse vs dense retrieval, hybrid retrieval, chunking strategy, and advanced patterns.

Runnable code: [page66_72_rag_demo.py](/Users/aarunyan/Documents/project/ss6_guidebook_new/page66_72/page66_72_rag_demo.py)

## 4.1 Start with an analogy

Think of RAG like a lawyer preparing an argument:

- The LLM is the lawyer.
- Your document store is the evidence archive.
- Retrieval is the paralegal finding the most relevant pages.
- Augmentation is putting those pages on the lawyer's desk before speaking.

Without RAG, the lawyer speaks from memory. With RAG, the lawyer cites current evidence.

Second analogy for hybrid retrieval:

- Sparse retrieval is searching by exact words in a library catalog.
- Dense retrieval is asking a smart librarian who understands meaning and synonyms.
- Hybrid retrieval is using both: catalog + librarian + tie-breaker (RRF).

## 4.2 Draw a diagram

```text
               OFFLINE (done once)
    +---------------------------------------+
    | docs -> chunk -> sparse/dense index   |
    +---------------------------------------+
                    |
                    v
               ONLINE (per query)
    user query
        |
        v
  +-------------+     +-------------+
  | sparse rank |     | dense rank  |
  +-------------+     +-------------+
         \                 /
          \               /
           v             v
          +---------------+
          | hybrid fusion |
          +---------------+
                  |
                  v
         top-k context chunks
                  |
                  v
        prompt = query + context
                  |
                  v
            grounded answer
```

## 4.3 Walk through the code

1. `DOCUMENTS` defines a small knowledge base with metadata (`source`, `title`) to mimic private/internal data.
2. `build_chunks()` performs sentence-based chunking for indexing.
3. `build_sparse_index()` + `sparse_score()` implement a mini TF-IDF retriever (slide 69 sparse side).
4. `dense_vector()` + `cosine_similarity()` implement a mini semantic retriever with synonym-like vectors (`car` ~ `vehicle`) (slide 69 dense side).
5. In `main()`, the same query is scored by both retrievers and ranked independently.
6. `reciprocal_rank_fusion()` merges both rank lists into a hybrid score (slide 70).
7. Top-k hybrid chunks are selected as context and passed to `build_grounded_answer()` (augment + generate from slides 67-68).
8. The script saves four visuals:
   - `page66_72_rag_pipeline.png`
   - `page66_72_sparse_vs_dense.png`
   - `page66_72_hybrid_rrf.png`
   - `page66_72_chunking_strategies.png`
9. The terminal transcript is saved to `page66_72_run_output.txt`.

## 4.4 Highlight a gotcha

Common misconception: "RAG guarantees truth."

Not exactly. RAG reduces hallucination risk, but still fails if:

- retrieval misses the right chunk,
- chunks are too small/too large,
- stale or low-quality documents are indexed,
- top-k context is too noisy.

Another easy mistake: evaluating only the final answer text.  
You should also evaluate retrieval quality itself (relevance/recall), which is why slide 72 highlights RAG evaluation patterns.

## How to run

```bash
python3 page66_72/page66_72_rag_demo.py
```
