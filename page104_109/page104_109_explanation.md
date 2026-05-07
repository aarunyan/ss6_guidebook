# Query Optimization (Slides 104-109)

## 1) Start with an analogy
Think of search like asking a librarian for help.

- If you say only “python error fix,” it’s like saying “I need help” without context.
- **Query rewriting** is the librarian asking follow-up questions and rewriting your request clearly.
- **HyDE** is like the librarian drafting a “sample answer paragraph” first, then using that richer paragraph to find better books.
- **Step-back prompting** is zooming out from a specific event to a broader category (from “2008 crisis” to “causes of financial crises”).
- **Sub-question decomposition** is splitting one big shopping list into smaller aisle-specific lists.
- **Query routing** is sending you to the right section: statistics desk (SQL), concept shelves (Vector DB), or live news terminal (Web).
- **Multi-query + RRF** is asking multiple librarians in parallel, then combining their ranked suggestions fairly.

## 2) ASCII diagram
```text
User Query (often vague)
        |
        v
+-------------------------+
| Query Optimization Layer |
+-------------------------+
| 1) Rewrite query         |
| 2) Generate HyDE text    |
| 3) Step-back generalize  |
| 4) Decompose subqueries  |
| 5) Multi-query variants  |
+-------------------------+
        |
        v
+-------------------------+
| Semantic Router         |
| - SQL DB                |
| - Vector DB             |
| - Web Search            |
+-------------------------+
        |
        v
Parallel retrieval from selected source(s)
        |
        v
RRF merge + deduplicate
        |
        v
Final ranked results
```

## 3) Walk through the code
File: `page104_109/page104_109_query_optimization_demo.py`

1. It defines a small in-memory document set (`DOCUMENTS`) as a toy corpus.
2. It implements a basic lexical retriever (`normalize`, `lexical_score`, `retrieve`) so results are measurable.
3. It adds optimization functions:
   - `rewrite_query(...)`
   - `hyde_paragraph(...)`
   - `step_back_query(...)`
   - `decompose_subqueries(...)`
4. It adds a semantic router (`route_query`) with three destinations:
   - `sql_db`
   - `vector_db`
   - `web`
5. It implements **Reciprocal Rank Fusion** (`reciprocal_rank_fusion`) to combine multiple ranked lists.
6. In `main()` it runs a full experiment:
   - Baseline retrieval for vague query
   - Rewritten retrieval
   - HyDE retrieval
   - Multi-query retrieval + RRF fusion
   - Step-back and decomposition examples
   - Routing examples
7. It saves three visualizations:
   - `page104_109/page104_109_quality_comparison.png`
   - `page104_109/page104_109_routing_distribution.png`
   - `page104_109/page104_109_rrf_fused_scores.png`

## 4) Highlight a gotcha
A common misconception is: “If I generate more query variants, results are always better.”

Not always. Low-quality variants can add noise and bury relevant docs. Multi-query helps most when:
- variants are **diverse but relevant**,
- routing sends each query to the **right source**, and
- fusion (RRF) is used to reduce domination by any single noisy list.

Another gotcha: routing mistakes are expensive. If a real-time question is routed to static docs only, quality drops even with great rewriting.

## 5) How to run
```bash
python3 page104_109/page104_109_query_optimization_demo.py
```

Then check:
- `page104_109/page104_109_run_output.txt`
- generated `.png` files in `page104_109/`
