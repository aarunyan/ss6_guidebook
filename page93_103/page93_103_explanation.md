# Conversation and Context Compression (Slides 93-103)

## 1) Start with an analogy
Think of this like planning a long road trip with friends:

- **Conversation compression** is your group chat recap. Instead of reading all 2,000 messages, someone posts: "Plan is Bangkok -> Chiang Mai, leave Friday 6 AM, 2 hotel options."
- **Context compression** is your travel folder cleanup. You keep only the pages about your route and budget, and ignore unrelated pages (old restaurant screenshots, random memes).

Another analogy:

- **Conversation compression** = meeting minutes from previous meetings.
- **Context compression** = highlighting only the 3 relevant paragraphs in a 50-page report.

## 2) Draw a diagram (ASCII)
```text
User asks question
      |
      v
+-------------------------------+
| Conversation side             |
| - Old turns -> summarized     |
| - Recent turns -> kept full   |
+-------------------------------+
      |
      +---------> Prompt assembly --------+
                                           |
+-------------------------------+          v
| Context side                  |      +--------+
| - Retrieved docs              |----->|  LLM   |
| - Keep relevant chunks only   |      +--------+
+-------------------------------+          |
                                           v
                                      Better answer
                               with lower token usage
```

## 3) Walk through the code
File: `page93_103_conversation_context_compression_demo.py`

1. **Simulate a multi-turn chat** with realistic token sizes per turn (`simulate_conversation`).
2. **Measure growth without compression** (`total_context_without_compression`): every new turn carries all previous turns.
3. **Apply conversation compression** (`total_context_with_conversation_compression`):
   - keep last 3 turns verbatim
   - compress older turns into a shorter summary
4. **Apply context compression** (`context_compression_tokens`): reduce retrieved document tokens by configurable ratios (100%, 80%, 60%, etc.).
5. **Model quality tradeoff** (`quality_score`): moderate compression helps remove noise, but too much compression harms answer quality.
6. **Generate visualizations**:
   - `page93_103_token_growth_comparison.png`
   - `page93_103_context_compression_curve.png`
   - `page93_103_quality_vs_compression.png`
7. **Save artifacts and run summary** in `page93_103_run_output.txt`.

## 4) Highlight a gotcha
A common mistake is assuming:

- "More compression is always better."

It is not. Over-compression can delete key facts, names, dates, or constraints. In practice, compression should be **adaptive**:

- keep high-salience items intact,
- compress low-value repetition,
- and preserve very recent turns in full detail.

## 5) Generated files
- `page93_103_token_growth_comparison.png`
- `page93_103_context_compression_curve.png`
- `page93_103_quality_vs_compression.png`
- `page93_103_ascii_flow.txt`
- `page93_103_run_output.txt`
