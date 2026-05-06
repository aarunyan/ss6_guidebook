# page73_76 Query Engine explanation

This note follows slides 73-76 of [AI_Full_Curriculum_EN.pptx.pptx](/Users/aarunyan/Documents/project/ss6_guidebook_new/AI_Full_Curriculum_EN.pptx.pptx): Query Engine, Text2SQL pipeline, a sample `text2sql_pipeline`, and when to choose Text2Pandas instead.

Runnable code: [page73_76_query_engine_demo.py](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_query_engine_demo.py)

Reference slide text:
- [page73_76_slide-073_text.txt](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_slide-073_text.txt)
- [page73_76_slide-074_text.txt](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_slide-074_text.txt)
- [page73_76_slide-075_text.txt](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_slide-075_text.txt)
- [page73_76_slide-076_text.txt](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_slide-076_text.txt)

## 4.1 Start with an analogy

Think of a Query Engine like a restaurant waiter who speaks both human and kitchen.

- You say, "I want something light and spicy."
- The waiter translates that into kitchen instructions.
- The kitchen prepares the dish.
- The waiter brings it back and explains what you got.

That is the slide 74 pipeline in plain language:

- `Schema Linking` = the waiter checks what ingredients the kitchen actually has.
- `SQL Generation` = the waiter writes the kitchen ticket.
- `Execute + Validate` = the kitchen cooks it and checks nothing is unsafe or impossible.
- `Explain Results` = the waiter presents the dish in a way you understand.

Second analogy for slide 76:

- `Text2SQL` is like ordering from a warehouse system. It is great when the data lives in a serious production database and you need logs, joins, and scale.
- `Text2Pandas` is like opening ingredients on your own kitchen counter. It is great when the data is already in memory and you want to slice, transform, and chart it quickly.

## 4.2 Draw a diagram

```text
User asks a question
        |
        v
+-------------------+
| 1. Schema Linking |
| Which table?      |
| Which columns?    |
+-------------------+
        |
        v
+-------------------+
| 2. SQL Generation |
| Build SELECT SQL  |
| Keep dialect safe |
+-------------------+
        |
        v
+----------------------+
| 3. Execute + Validate|
| Run query            |
| Limit rows           |
| Reject dangerous SQL |
+----------------------+
        |
        v
+--------------------+
| 4. Explain Results |
| Table -> insight   |
| Insight -> user    |
+--------------------+

Alternative branch:

User asks a question
        |
        v
+--------------------+
| Text2Pandas path   |
| DataFrame in memory|
| groupby / filter   |
| chart / transform  |
+--------------------+
```

## 4.3 Walk through the code

1. `build_database()` creates a tiny SQLite database called `race_records`. This mirrors the table shown on slide 75.
2. `schema_linking()` does the "what parts of the schema matter?" step. In a real system, an LLM would infer this from the user query and database metadata. In this teaching example, we use keyword hints so the pipeline stays easy to follow.
3. `generate_sql()` converts a natural-language request into a safe `SELECT` query. This matches slide 74's SQL generation stage and slide 75's simple `text2sql_pipeline`.
4. `validate_sql()` blocks dangerous commands like `DROP` and `DELETE`. That directly reflects the slide note saying the system should stay read-only.
5. `execute_sql()` runs the query with a row limit. This is the "execute + validate" stage from slide 74.
6. `explain_results()` turns the returned rows into a conversational answer. This is the last stage of the Query Engine: results become plain-English insight.
7. `run_pandas_analysis()` shows the other branch from slide 76. Instead of generating SQL, it works on a `pandas` DataFrame already loaded in memory and computes average speed rating by track.
8. The script also saves four visuals:
   - [page73_76_query_engine_pipeline.png](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_query_engine_pipeline.png)
   - [page73_76_text2sql_result.png](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_text2sql_result.png)
   - [page73_76_text2pandas_result.png](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_text2pandas_result.png)
   - [page73_76_mode_comparison.png](/Users/aarunyan/Documents/project/ss6_guidebook_new/page73_76/page73_76_mode_comparison.png)

## 4.4 Highlight a gotcha

Common misconception: "If the model can write SQL, the problem is solved."

Not really. The fragile part is often not the SQL syntax. It is the mapping from human words to the correct schema.

Examples:

- If the user says "best horse," do they mean lowest `rank`, highest `speed_rating`, or most wins?
- If the schema context is incomplete, the model may write valid SQL against the wrong table.
- If you skip validation, a generated query might be unsafe or too expensive.

Another easy mistake is choosing the wrong tool:

- Use `Text2SQL` when data already lives in a relational database and you need joins, traceability, or scale.
- Use `Text2Pandas` when the data is already loaded in memory and you want flexible analysis or quick charting.

## How to run

```bash
python3 page73_76/page73_76_query_engine_demo.py
```
