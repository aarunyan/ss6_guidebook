"""page81_86_agentic_tool_skill_demo.py
Demonstrates Agentic Tool Call and Skill System concepts from slides 81-86.
"""

from __future__ import annotations

import concurrent.futures
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List


OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class Tool:
    """Represents an agent-callable tool with a JSON-style schema."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Skill:
    """Represents a reusable, versioned production skill."""

    name: str
    version: str
    description: str
    run: Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolExecutionError(Exception):
    """Raised when a tool fails and the agent needs fallback behavior."""


# ---------------------------
# Tool implementations
# ---------------------------

def tool_web_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate external search with occasional transient failure."""
    query = payload["query"]
    time.sleep(0.5)
    if "unstable" in query.lower():
        raise ToolExecutionError("web_search API timeout")
    return {
        "results": [
            f"Result 1 for '{query}'",
            f"Result 2 for '{query}'",
            "Parallel calls reduce total waiting time.",
        ]
    }


def tool_weather(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate weather API output."""
    city = payload["city"]
    time.sleep(0.7)
    return {"city": city, "forecast": "32C, chance of rain 20%"}


def tool_calendar(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate calendar availability lookup."""
    person = payload["person"]
    time.sleep(0.6)
    return {"person": person, "available_slots": ["10:00", "14:30"]}


TOOLS: Dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description=(
            "Search the web for current information. Use when facts are time-sensitive "
            "or need verification from external data."
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=tool_web_search,
    ),
    "weather_api": Tool(
        name="weather_api",
        description="Get weather forecast for planning decisions.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        fn=tool_weather,
    ),
    "calendar_api": Tool(
        name="calendar_api",
        description="Check a person's free time slots from calendar.",
        input_schema={
            "type": "object",
            "properties": {"person": {"type": "string"}},
            "required": ["person"],
        },
        fn=tool_calendar,
    ),
}


# ---------------------------
# Skill registry implementations
# ---------------------------

def skill_query_customer_db_v1(payload: Dict[str, Any]) -> Dict[str, Any]:
    customer_id = payload["customer_id"]
    return {"customer_id": customer_id, "tier": "Gold", "risk_score": 0.22}


def skill_send_slack_alert_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "channel": payload.get("channel", "#ops-alerts"),
        "status": "sent",
        "message": payload["message"],
    }


SKILLS: Dict[str, Skill] = {
    "query_customer_db@1.1": Skill(
        name="query_customer_db",
        version="1.1",
        description="Lookup customer profile and risk metadata.",
        run=skill_query_customer_db_v1,
    ),
    "send_slack_alert@2.0": Skill(
        name="send_slack_alert",
        version="2.0",
        description="Post alert to Slack with routing metadata.",
        run=skill_send_slack_alert_v2,
    ),
}


class Agent:
    """Simple agent showing tool-call loops and skill composition."""

    def __init__(self, tools: Dict[str, Tool], skills: Dict[str, Skill]) -> None:
        self.tools = tools
        self.skills = skills

    def call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with schema checking and error wrapping."""
        tool = self.tools[tool_name]
        required_fields = tool.input_schema.get("required", [])
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValueError(f"Missing required fields for {tool_name}: {missing}")

        print(f"[ToolCall] {tool_name} <- {payload}")
        result = tool.fn(payload)
        print(f"[ToolResult] {tool_name} -> {result}")
        return result

    def run_parallel_tools(self) -> Dict[str, Any]:
        """Demonstrate latency improvement by running tools concurrently."""
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                "weather": pool.submit(self.call_tool, "weather_api", {"city": "Bangkok"}),
                "calendar": pool.submit(self.call_tool, "calendar_api", {"person": "Alex"}),
            }
            results = {name: fut.result() for name, fut in futures.items()}
        elapsed = time.time() - start
        results["parallel_elapsed_sec"] = round(elapsed, 3)
        return results

    def run_sequential_tools(self) -> Dict[str, Any]:
        """Baseline to compare against parallel execution."""
        start = time.time()
        weather = self.call_tool("weather_api", {"city": "Bangkok"})
        calendar = self.call_tool("calendar_api", {"person": "Alex"})
        elapsed = time.time() - start
        return {
            "weather": weather,
            "calendar": calendar,
            "sequential_elapsed_sec": round(elapsed, 3),
        }

    def run_with_fallback(self) -> Dict[str, Any]:
        """Show the loop: tool failure -> retry/fallback -> continue reasoning."""
        try:
            return self.call_tool("web_search", {"query": "unstable market update"})
        except ToolExecutionError as err:
            print(f"[ToolError] {err} | applying fallback strategy")
            # Fallback strategy: retry with narrower query less likely to fail.
            return self.call_tool("web_search", {"query": "market update summary"})

    def run_skill_pipeline(self, customer_id: str) -> Dict[str, Any]:
        """Compose Skill A output into Skill B input."""
        profile = self.skills["query_customer_db@1.1"].run({"customer_id": customer_id})
        message = (
            f"Customer {profile['customer_id']} tier={profile['tier']} "
            f"risk={profile['risk_score']} needs follow-up"
        )
        alert = self.skills["send_slack_alert@2.0"].run(
            {"channel": "#customer-success", "message": message}
        )
        return {"profile": profile, "alert": alert}


def save_ascii_flow() -> None:
    """Save an ASCII diagram for the markdown explanation."""
    ascii_diagram = r"""
User Prompt
    |
    v
+-------------------+
|   LLM / Agent     |
+-------------------+
   | plan next step
   |------------------------------.
   v                              |
+-------------------+             |
| Tool Decision     |             |
+-------------------+             |
   | call tool(s)                 |
   v                              |
+-------------------+             |
| External Tools    |             |
| web/weather/etc   |             |
+-------------------+             |
   | return result                |
   '------------------------------'
   |
   v
+-------------------+
| Skill Registry    |
| versioned skills  |
+-------------------+
   |
   v
Final Answer / Action
""".strip("\n")
    (OUTPUT_DIR / "page81_86_agentic_flow_ascii.txt").write_text(ascii_diagram, encoding="utf-8")


def save_visualization(sequential_sec: float, parallel_sec: float) -> None:
    """Create latency comparison visualization.

    Saves PNG when matplotlib is available; otherwise saves text visualization.
    """
    try:
        import matplotlib.pyplot as plt

        labels = ["Sequential", "Parallel"]
        values = [sequential_sec, parallel_sec]
        colors = ["#d95f02", "#1b9e77"]

        plt.figure(figsize=(7, 4))
        bars = plt.bar(labels, values, color=colors)
        plt.title("Tool Calls: Sequential vs Parallel Latency")
        plt.ylabel("Seconds")
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}s", ha="center")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "page81_86_parallel_latency.png", dpi=150)
        plt.close()
    except Exception as err:
        fallback = (
            "Matplotlib unavailable; text fallback visualization\n"
            f"Sequential: {sequential_sec:.2f}s\n"
            f"Parallel  : {parallel_sec:.2f}s\n"
            f"Speedup   : {sequential_sec / max(parallel_sec, 1e-6):.2f}x\n"
            f"Error     : {err}\n"
        )
        (OUTPUT_DIR / "page81_86_parallel_latency_fallback.txt").write_text(
            fallback, encoding="utf-8"
        )


def main() -> None:
    # Set deterministic seed for reproducible demo behavior.
    random.seed(7)

    print("=== Agentic Tool Call + Skill System Demo ===")
    agent = Agent(TOOLS, SKILLS)

    print("\n1) Sequential tool calls")
    sequential = agent.run_sequential_tools()

    print("\n2) Parallel tool calls")
    parallel = agent.run_parallel_tools()

    print("\n3) Error handling with fallback")
    fallback_search = agent.run_with_fallback()

    print("\n4) Skill composition with versioned registry")
    skill_pipeline = agent.run_skill_pipeline(customer_id="CUST-2048")

    summary = {
        "sequential": sequential,
        "parallel": parallel,
        "fallback_search": fallback_search,
        "skill_pipeline": skill_pipeline,
    }

    # Save machine-readable run output for review.
    (OUTPUT_DIR / "page81_86_run_output.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    save_ascii_flow()
    save_visualization(
        sequential_sec=sequential["sequential_elapsed_sec"],
        parallel_sec=parallel["parallel_elapsed_sec"],
    )

    print("\nDemo artifacts saved:")
    print("- page81_86_run_output.json")
    print("- page81_86_agentic_flow_ascii.txt")
    print("- page81_86_parallel_latency.png (or fallback txt)")


if __name__ == "__main__":
    main()
