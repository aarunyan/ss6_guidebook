# Page 81-86: Agentic Tool Call and Skill System

This explanation is based on slides **81-86** from `AI_Full_Curriculum_EN.pptx.pptx`.

## 4.1 Start with an analogy

Think of an LLM agent like a **smart office manager**:
- The manager (LLM) can think and decide, but cannot directly check weather, open calendars, or search the internet.
- So it asks specialists (tools) to do those tasks, then uses the results to continue planning.

Now think of **skills** as the company’s **standard operating procedures (SOPs)**:
- Instead of each manager inventing steps every time, the company stores reusable, tested playbooks.
- Those playbooks are versioned (`v1.1`, `v2.0`) so teams can improve them safely.

Second analogy for skill composition:
- Skill A is like a **lab test**, Skill B is like a **doctor’s prescription**.
- The test output becomes the input to the prescription decision.

## 4.2 Draw a diagram (ASCII)

```text
User Prompt
    |
    v
+-------------------+
|   LLM / Agent     |
+-------------------+
   | decide next step
   v
+-------------------+
| Tool Call Loop    |
| call -> result    |
| call -> result    |
+-------------------+
   |
   v
+-------------------+
| Skill Registry    |
| reusable/versioned|
+-------------------+
   |
   v
Final Action/Answer
```

## 4.3 Walk through the code

1. `Tool` and `Skill` dataclasses define structure.
- `Tool` includes `name`, `description`, `input_schema`, and function (`fn`).
- `Skill` includes `name`, `version`, `description`, and executable function (`run`).

2. Tool implementations simulate external systems.
- `tool_web_search(...)` represents external web API calls and can fail.
- `tool_weather(...)` and `tool_calendar(...)` simulate typical utility tools.

3. The `Agent` class demonstrates the **tool call loop**.
- `call_tool(...)` validates required input from schema.
- It logs `[ToolCall]` and `[ToolResult]` so the loop is visible.

4. Sequential vs parallel execution.
- `run_sequential_tools()` calls weather and calendar one by one.
- `run_parallel_tools()` uses `ThreadPoolExecutor` to call both together.
- This illustrates the latency benefit from slide 82 (parallel tool calls).

5. Error handling and fallback.
- `run_with_fallback()` intentionally triggers a tool error (`unstable` query).
- It catches `ToolExecutionError`, logs the issue, retries with a safer fallback query.

6. Skill system and composition.
- `SKILLS` acts like a mini skill registry with explicit versions.
- `run_skill_pipeline(...)` runs:
  - `query_customer_db@1.1` (Skill A)
  - then feeds output into `send_slack_alert@2.0` (Skill B)

7. Output artifacts.
- `page81_86_run_output.json`: machine-readable results.
- `page81_86_run_output.txt`: console log from execution.
- `page81_86_agentic_flow_ascii.txt`: reusable ASCII flow.
- `page81_86_parallel_latency.png`: visual sequential vs parallel comparison.

## 4.4 Highlight a gotcha

Common misconception: **“If I wrote good tool code, the model will always pick it correctly.”**

In practice, slide 83 is exactly right: the model mostly relies on the **tool description + schema** to decide usage.
- Weak description => wrong tool choice or no tool call.
- Missing required fields in schema => runtime errors.
- Vague skill versioning => brittle production behavior.

So the hidden engineering work is not only writing functions; it is designing:
- excellent tool descriptions,
- strict input schemas,
- clear skill version contracts.

## Run command

```bash
cd page81_86
python3 page81_86_agentic_tool_skill_demo.py
```
