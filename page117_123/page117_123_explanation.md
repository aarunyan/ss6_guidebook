# Page 117-123: Guardrails and LLM-as-a-Judge

## Start with an analogy
Think of this system like an airport:

- **Input guardrail** is the security check at the entrance. Dangerous items should never get inside.
- **LLM generation** is the traveler moving through the airport and producing a travel plan (the model output).
- **Output guardrail** is a second security scan before boarding. Even if someone got through the first gate, this check can still stop risky behavior.
- **LLM-as-a-judge** is the gate supervisor scoring if the final plan is high-quality and safe enough to approve.

Second analogy (for why both guardrails and judge are needed):
- Guardrails are like **hard traffic lights** (red means stop, no negotiation).
- LLM-as-a-judge is like a **driving examiner** (scores quality and decision-making, not just rule matching).

## Draw a diagram (ASCII)
```text
User Prompt
   |
   v
+------------------+
| Input Guardrail  |---- blocked ----> Reject (policy)
+------------------+
   |
   v
+------------------+
|   LLM Generate   |
+------------------+
   |
   v
+------------------+
| Output Guardrail |---- blocked ----> Reject (unsafe output)
+------------------+
   |
   v
+------------------+
|  LLM-as-a-Judge  |---- low score --> Reject (quality/safety)
+------------------+
   |
   v
Approved Response
```

## Walk through the code
1. `build_requests()` creates sample prompts with ground-truth labels (`safe` / `unsafe`) for simulation.
2. `input_guardrail()` uses keyword/pattern rules (like `phishing`, `malware`, `bomb`) to block clearly unsafe prompts early.
3. `simulate_generation()` creates synthetic model behavior:
   - Safe prompts: usually high quality, low risk.
   - Unsafe prompts: noisier quality, higher risk.
4. `output_guardrail()` blocks risky generated outputs using a risk threshold.
5. `llm_as_judge_score()` applies a rubric score:
   - `0.7 * quality + 0.3 * (1 - safety_risk)`
   - Higher score means better quality and lower risk.
6. `process_requests()` combines all stages and stores decisions/reasons per request.
7. Plot functions create the visuals:
   - `page117_123_guardrails_funnel.png`: how many requests survive each stage.
   - `page117_123_judge_scatter.png`: relation between quality and judge score.
   - `page117_123_decision_matrix.png`: final approve/reject versus true label.
8. `save_run_log()` writes a table to `page117_123_run_output.txt` for transparent inspection.

## Highlight a gotcha
A common misconception is: **"If I have LLM-as-a-judge, I don’t need guardrails."**

That is risky.
- Judge models can be inconsistent and can fail on adversarial prompts.
- Guardrails provide deterministic policy constraints.
- Best practice is **layered control**: input guardrail + output guardrail + judge.

Another gotcha: tuning thresholds (`output_guardrail` threshold, judge approval threshold) too aggressively can over-block good responses; too loose can allow risky outputs. Thresholds should be validated with evaluation data, not guessed.

## How to run
```bash
python3 page117_123/page117_123_guardrails_judge_demo.py
```

## Generated artifacts
- `page117_123_guardrails_judge_demo.py`
- `page117_123_guardrails_funnel.png`
- `page117_123_judge_scatter.png`
- `page117_123_decision_matrix.png`
- `page117_123_run_output.txt`
- `page117_123_explanation.md`
- `page117_123_slide-117.png` ... `page117_123_slide-123.png`
