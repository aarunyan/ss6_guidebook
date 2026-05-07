# Page 124-134: Model Context Protocol (MCP)

## 4.1 Start with an analogy
Think of MCP like a **universal power strip + adapter standard** for AI.

- Before MCP: each laptop brand needs its own charger for each country socket.
- After MCP: one universal standard lets many devices plug into many sockets safely.

Another analogy for the A2A comparison:
- **MCP** is like giving one assistant a toolbox and a library card.
- **A2A** is like letting your assistant coordinate with other specialist assistants.

So MCP is mostly about **LLM ↔ tools/data context**, while A2A is about **agent ↔ agent teamwork**.

## 4.2 Draw a diagram (ASCII)
```text
User Request
   |
   v
+--------------------+
|   MCP Host (LLM)   |
+--------------------+
   | discovers capabilities
   v
+-----------------------------------------+
|              MCP Server                 |
|  - Tools (callable functions)           |
|  - Resources (read-only data/files)     |
|  - Prompts (reusable templates)         |
+-----------------------------------------+
   |          |                 |
   | call     | read            | render
   v          v                 v
 Tool Fn   Resource Data    Prompt Template
   |
   v
Result -> back to Host -> final answer to user

Security gate:
Host must provide scope (e.g., tool:build_itinerary)
No scope -> PermissionDenied
```

## 4.3 Walk through the code
1. `integration_effort_without_mcp()` and `integration_effort_with_mcp()` model slide 125's key idea:
   - Without MCP: integrations grow as `hosts * services`.
   - With MCP: integration grows as `hosts + services`.

2. `Tool`, `Resource`, and `PromptTemplate` classes model slide 133's 3 MCP primitives:
   - Tools: executable functions.
   - Resources: read-only context.
   - Prompts: reusable templates.

3. `MCPServer` is a tiny server abstraction:
   - `list_capabilities()` acts like discovery.
   - `read_resource()` simulates resource reading.
   - `call_tool()` simulates tool execution with scope checks.

4. `make_travel_server()` builds a travel-domain MCP server inspired by slides 129-132:
   - A weather resource (`api://weather/maui-next-week`).
   - A tool (`build_itinerary`) that computes budget allocation.
   - A prompt template (`travel_summary`) for structured responses.

5. `run_mcp_host_flow()` simulates host behavior:
   - Discover primitives.
   - Read weather resource.
   - Call the tool with permission scope.
   - Call the same tool without scope to show denial.
   - Render a reusable prompt.

6. Visualization output:
   - `page124_134_integration_effort.png`: before/after integration scaling.
   - `page124_134_primitives_bar.png`: the 3 primitives used in the run.
   - `page124_134_run_output.txt`: JSON-like execution trace for learning.

## 4.4 Highlight a gotcha
A common misconception: **"MCP is just function calling."**

Not quite. MCP includes function calling, but also standardizes:
- resource access (context injection)
- reusable prompts
- transport and permission patterns

Another gotcha: people assume "connected" means "fully trusted."
MCP still needs strict scopes and user consent, especially for remote servers (OAuth), exactly like slide 134 emphasizes.

## Run
```bash
python3 page124_134/page124_134_mcp_demo.py
```

## Generated files
- `page124_134_mcp_demo.py`
- `page124_134_explanation.md`
- `page124_134_integration_effort.png`
- `page124_134_primitives_bar.png`
- `page124_134_run_output.txt`
- `page124_134_slide-124.png` ... `page124_134_slide-134.png`
