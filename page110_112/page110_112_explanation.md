# Page 110-112: Search Strategies (Semantic, Keyword, Hybrid, Agentic)

## 1) Start with an analogy
Think of finding information like finding a book in a huge library:

- **Keyword search** is like asking the librarian: "Give me books with this exact word in the title."
- **Semantic search** is like saying: "I don't remember the exact word, but I remember the meaning."
- **Hybrid + RRF** is like asking **two librarians** (exact-word + meaning) and combining both recommendation lists fairly.
- **Agentic search** is like a **research assistant** who does this loop: search -> read -> refine question -> search again.

Another quick analogy for Agentic search:
- Regular retrieval is one taxi ride.
- Agentic retrieval is a multi-stop trip where the driver adjusts route after each checkpoint.

## 2) Draw a diagram (ASCII)

```text
+----------------------+      +----------------------+      +------------------------+
| User Question        | ---> | Retriever Layer      | ---> | Candidate Documents    |
+----------------------+      +----------------------+      +------------------------+
                                    |                                  |
                                    |                                  v
                                    |                         +------------------+
                                    |                         | Rank / Fuse      |
                                    |                         | (RRF for hybrid) |
                                    |                         +------------------+
                                    |                                  |
                                    v                                  v
                         +----------------------+         +----------------------+
                         | Agent (optional)     | <-----> | Observe + Re-query   |
                         | decides next query   |         | iterative refinement |
                         +----------------------+         +----------------------+
                                    |
                                    v
                           +-------------------+
                           | Final answer path |
                           +-------------------+
```

## 3) Walk through the code

1. **Build toy corpus and queries**
- `build_corpus()` creates 6 tiny documents.
- `build_queries()` creates 4 user questions mixing paraphrases and technical terms.

2. **Define two retrieval styles**
- `semantic_scores()` uses vector similarity (`cosine_similarity`) to capture meaning.
- `keyword_scores()` uses a BM25-like exact-term matching score with IDF-style weighting.

3. **Rank and fuse**
- `rank_desc()` sorts docs by score.
- `rrf_fuse()` combines ranking lists using Reciprocal Rank Fusion (RRF), matching slide 111's "Hybrid + RRF" idea.

4. **Add iterative (agentic) behavior**
- `agentic_search()` runs one fused search pass.
- It then "observes" top doc terms, expands the query, and searches again.
- That simulates the slide 112 loop: search -> observe -> search again.

5. **Evaluate and visualize**
- `evaluate_top1_accuracy()` compares top-1 retrieval accuracy across strategies.
- `make_accuracy_chart()` saves bar chart: `page110_112_strategy_accuracy.png`.
- `make_flow_chart()` saves tradeoff scatter: `page110_112_quality_vs_complexity.png`.
- `write_ascii_flow()` saves diagram text: `page110_112_ascii_flow.txt`.

## 4) Highlight a gotcha
A common misconception is: **"Semantic always beats keyword."**

Not always.
- If the query contains exact technical tokens (`BM25`, `GPU`, proper nouns), keyword matching can be stronger.
- If the query is paraphrased (different words, same meaning), semantic can win.
- In practice, **Hybrid** is often safest as default (exactly what slide 111 recommends), and **Agentic** is strongest for multi-step/complex tasks when latency and system complexity are acceptable.

## 5) Files produced
- `page110_112_search_strategies_demo.py`
- `page110_112_run_output.txt`
- `page110_112_strategy_accuracy.png`
- `page110_112_quality_vs_complexity.png`
- `page110_112_ascii_flow.txt`
