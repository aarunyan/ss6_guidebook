# Page 113-116: ReACT Agent and Deep Research

## 1) Start with an analogy
Think of a **ReACT agent** like a chef in a busy kitchen:
- The chef **thinks**: "What should I cook next?"
- Then **acts**: grabs ingredients or uses a tool.
- Then **observes**: tastes the result and adjusts.
- This repeats until the dish is right.

Think of **Deep Research** like a journalist preparing an investigative story:
- They collect evidence from many sources.
- They score source quality (credibility, relevance, recency).
- They compare conflicting signals.
- They publish a balanced synthesis, not just the first source they found.

Second analogy for the difference:
- ReACT is a **driver navigating turn-by-turn** in real time.
- Deep Research is a **trip planner** comparing flights, weather, traffic, and cost before deciding.

## 2) Draw a diagram (ASCII)
```text
ReACT Loop (execution-time loop)

[Question]
    |
    v
[Thought] -> [Action (tool call)] -> [Observation]
    ^                                  |
    |__________________________________|
             repeat until done


Deep Research Pipeline (evidence synthesis)

[Question]
    |
    v
[Collect Sources] -> [Score Signals] -> [Rank Evidence] -> [Synthesize Answer]
                        (credibility,
                         relevance,
                         recency)
```

## 3) Walk through the code
1. `react_agent_demo(question)` builds a 3-iteration ReACT trace.
- Each iteration stores `thought`, `action`, `observation`.
- This shows the core loop behavior directly.

2. `deep_research_demo()` returns a small evidence set.
- Each source includes three quality signals: credibility, relevance, recency.
- These are represented by an `Evidence` dataclass for clarity.

3. `compute_weighted_scores(...)` combines the three signals.
- Weighted formula: `0.45*credibility + 0.40*relevance + 0.15*recency`.
- This simulates a transparent, auditable ranking step.

4. Visualization functions make the behavior visible.
- `plot_react_timeline(...)`: timeline of Thought/Action/Observation across iterations.
- `plot_deep_research_scores(...)`: ranked confidence by source.
- `plot_signal_breakdown(...)`: grouped bars for each signal by source.

5. `save_run_output(...)` writes a readable run log.
- Includes the full ReACT trace and final ranked scores.

6. `main()` orchestrates everything.
- Runs demos, computes scores, saves all images + text output.
- Prints generated filenames so execution is easy to verify.

## 4) Highlight a gotcha
Common misconception: **ReACT and Deep Research are the same thing.**

They overlap, but they are not identical:
- ReACT is primarily a **control loop pattern** (how an agent reasons and acts step-by-step).
- Deep Research is primarily an **evidence workflow pattern** (how an agent gathers, scores, and synthesizes many sources).

Another gotcha:
- People often trust one high-credibility source too early.
- In deep research, a source can be credible but still low relevance for this exact question.

## 5) Files produced
- `page113-116_react_deep_research_demo.py`
- `page113-116_explanation.md`
- `page113-116_run_output.txt`
- `page113-116_react_timeline.png`
- `page113-116_deep_research_scores.png`
- `page113-116_deep_research_signal_breakdown.png`
- `page113-116_slide-113.png`
- `page113-116_slide-114.png`
- `page113-116_slide-115.png`
- `page113-116_slide-116.png`
