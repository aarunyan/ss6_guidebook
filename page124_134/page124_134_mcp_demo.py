"""
page124_134_mcp_demo.py

Educational simulation for slides 124-134 about MCP (Model Context Protocol).
It demonstrates:
1) Before/After MCP integration effort
2) MCP Host <-> Server request flow
3) MCP primitives: Tools, Resources, Prompts
4) Security/OAuth style permission checks

Run:
    python3 page124_134/page124_134_mcp_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any
import random
import json

import matplotlib.pyplot as plt


# ------------------------------
# Section 1: "Before vs After MCP" simulation
# ------------------------------

def integration_effort_without_mcp(hosts: int, services: int) -> int:
    """Without MCP, every host needs a custom integration for every service."""
    return hosts * services


def integration_effort_with_mcp(hosts: int, services: int) -> int:
    """With MCP, each host implements MCP once and each service publishes one MCP server."""
    return hosts + services


# ------------------------------
# Section 2: Minimal MCP-like server primitives
# ------------------------------

@dataclass
class Tool:
    """A callable operation that may have side effects (matches MCP 'tools' idea)."""

    name: str
    description: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Resource:
    """A read-only data source (matches MCP 'resources' idea)."""

    uri: str
    data: Dict[str, Any]


@dataclass
class PromptTemplate:
    """Reusable prompt skeleton (matches MCP 'prompts' idea)."""

    name: str
    template: str

    def render(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)


@dataclass
class MCPServer:
    """
    Tiny educational MCP-like server.
    - Exposes tools/resources/prompts
    - Enforces permission scopes for tool calls
    """

    name: str
    tools: Dict[str, Tool] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    prompts: Dict[str, PromptTemplate] = field(default_factory=dict)

    def list_capabilities(self) -> Dict[str, List[str]]:
        return {
            "tools": sorted(self.tools.keys()),
            "resources": sorted(self.resources.keys()),
            "prompts": sorted(self.prompts.keys()),
        }

    def read_resource(self, uri: str) -> Dict[str, Any]:
        if uri not in self.resources:
            raise KeyError(f"Resource not found: {uri}")
        return self.resources[uri].data

    def call_tool(self, tool_name: str, args: Dict[str, Any], scopes: List[str]) -> Dict[str, Any]:
        # Simple permission model: each tool requires a scope named "tool:<tool_name>"
        required_scope = f"tool:{tool_name}"
        if required_scope not in scopes:
            return {
                "ok": False,
                "error": "PermissionDenied",
                "required_scope": required_scope,
            }

        if tool_name not in self.tools:
            return {"ok": False, "error": "ToolNotFound"}

        result = self.tools[tool_name].fn(args)
        return {"ok": True, "result": result}


# ------------------------------
# Section 3: Example domain (travel planning from the slides)
# ------------------------------


def make_travel_server() -> MCPServer:
    random.seed(42)

    # Read-only resource: mini weather snapshot
    weather_resource = Resource(
        uri="api://weather/maui-next-week",
        data={
            "location": "Maui",
            "forecast": [
                {"day": "Mon", "condition": "Sunny", "temp_f": 82},
                {"day": "Tue", "condition": "Sunny", "temp_f": 83},
                {"day": "Wed", "condition": "Shower", "temp_f": 79},
                {"day": "Thu", "condition": "Sunny", "temp_f": 82},
                {"day": "Fri", "condition": "Partly Cloudy", "temp_f": 81},
            ],
        },
    )

    # Tool with side effect style behavior: creates itinerary object
    def build_itinerary(args: Dict[str, Any]) -> Dict[str, Any]:
        budget = int(args.get("budget_usd", 3000))
        travelers = int(args.get("travelers", 1))

        # Educational budget allocation logic
        flights = int(1150 * travelers)
        hotel = int(900 * travelers)
        activities = int(275 * travelers)
        meals = int(0.2 * budget)

        spent = flights + hotel + activities + meals
        remaining = budget - spent

        return {
            "flights_usd": flights,
            "hotel_usd": hotel,
            "activities_usd": activities,
            "meals_usd": meals,
            "spent_usd": spent,
            "remaining_usd": remaining,
            "status": "within_budget" if remaining >= 0 else "over_budget",
        }

    prompt = PromptTemplate(
        name="travel_summary",
        template=(
            "Create a concise travel summary for {location}. "
            "Mention weather highlights and budget status ({status})."
        ),
    )

    server = MCPServer(name="travel-mcp-server")
    server.resources[weather_resource.uri] = weather_resource
    server.tools["build_itinerary"] = Tool(
        name="build_itinerary",
        description="Constructs a 5-day itinerary budget split.",
        fn=build_itinerary,
    )
    server.prompts[prompt.name] = prompt
    return server


# ------------------------------
# Section 4: Host simulation
# ------------------------------


def run_mcp_host_flow() -> Dict[str, Any]:
    server = make_travel_server()

    # Host discovers server capabilities (Tools, Resources, Prompts)
    capabilities = server.list_capabilities()

    # Host reads weather resource (read-only context injection)
    weather = server.read_resource("api://weather/maui-next-week")

    # Simulate OAuth-like scope grant from user consent
    granted_scopes = ["tool:build_itinerary"]

    # Host asks server to execute tool
    itinerary_response = server.call_tool(
        tool_name="build_itinerary",
        args={"budget_usd": 3000, "travelers": 1},
        scopes=granted_scopes,
    )

    # Also show a failed permission case to highlight security model
    denied_response = server.call_tool(
        tool_name="build_itinerary",
        args={"budget_usd": 3000, "travelers": 1},
        scopes=[],
    )

    # Use prompt template to generate a reusable instruction
    status = itinerary_response["result"]["status"] if itinerary_response["ok"] else "unknown"
    prompt_text = server.prompts["travel_summary"].render(location="Maui", status=status)

    return {
        "capabilities": capabilities,
        "weather": weather,
        "itinerary_response": itinerary_response,
        "denied_response": denied_response,
        "prompt_text": prompt_text,
    }


# ------------------------------
# Section 5: Visualizations
# ------------------------------


def plot_integration_savings(host_values: List[int], services: int, output_path: str) -> None:
    without_vals = [integration_effort_without_mcp(h, services) for h in host_values]
    with_vals = [integration_effort_with_mcp(h, services) for h in host_values]

    plt.figure(figsize=(9, 5))
    plt.plot(host_values, without_vals, marker="o", label="Without MCP (hosts x services)")
    plt.plot(host_values, with_vals, marker="o", label="With MCP (hosts + services)")
    plt.title("Integration Effort: Before vs After MCP")
    plt.xlabel("Number of Hosts (LLM apps)")
    plt.ylabel("Integration Units")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()



def plot_primitives_bar(output_path: str) -> None:
    # Simple chart to visualize the 3 MCP primitives and relative usage frequency in this demo run
    labels = ["Tools", "Resources", "Prompts"]
    values = [1, 1, 1]
    colors = ["#27C1D6", "#5BA3F5", "#F7A44C"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("MCP Primitives Used in This Demo")
    plt.ylabel("Count")
    plt.ylim(0, 1.4)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.03, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


# ------------------------------
# Main execution
# ------------------------------


def main() -> None:
    # 1) Run conceptual host/server flow
    flow = run_mcp_host_flow()

    # 2) Generate visual outputs
    plot_integration_savings(
        host_values=list(range(1, 11)),
        services=6,
        output_path="page124_134/page124_134_integration_effort.png",
    )
    plot_primitives_bar("page124_134/page124_134_primitives_bar.png")

    # 3) Save structured run output for easy inspection
    with open("page124_134/page124_134_run_output.txt", "w", encoding="utf-8") as f:
        f.write("=== MCP DEMO RUN OUTPUT ===\n\n")
        f.write("Capabilities discovered:\n")
        f.write(json.dumps(flow["capabilities"], indent=2))
        f.write("\n\nWeather resource:\n")
        f.write(json.dumps(flow["weather"], indent=2))
        f.write("\n\nAuthorized tool call response:\n")
        f.write(json.dumps(flow["itinerary_response"], indent=2))
        f.write("\n\nUnauthorized tool call response:\n")
        f.write(json.dumps(flow["denied_response"], indent=2))
        f.write("\n\nRendered prompt template:\n")
        f.write(flow["prompt_text"] + "\n")

    print("Generated:")
    print("- page124_134/page124_134_integration_effort.png")
    print("- page124_134/page124_134_primitives_bar.png")
    print("- page124_134/page124_134_run_output.txt")


if __name__ == "__main__":
    main()
