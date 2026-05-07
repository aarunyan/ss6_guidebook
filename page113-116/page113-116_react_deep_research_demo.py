"""
page113-116_react_deep_research_demo.py

Demonstrates two concepts:
1) ReACT Agent: Reason -> Act -> Observe loop for tool-using agents
2) Deep Research: multi-source evidence collection, scoring, and synthesis

All output files are prefixed with page113-116_ as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
PREFIX = "page113-116_"


@dataclass
class Evidence:
    source: str
    claim: str
    credibility: float
    relevance: float
    recency: float


def react_agent_demo(question: str) -> List[Dict[str, str]]:
    """Run a tiny, deterministic ReACT loop over mock tools."""
    steps: List[Dict[str, str]] = []

    # Step 1: reason about what to do first.
    thought_1 = "I should gather broad context before answering."
    action_1 = "search_web('ReACT agent components')"
    observation_1 = "Found recurring pattern: Thought -> Action -> Observation."
    steps.append({"thought": thought_1, "action": action_1, "observation": observation_1})

    # Step 2: reason again based on observation.
    thought_2 = "Need a concrete example with tools and loops."
    action_2 = "read_docs('agent_tool_use_examples')"
    observation_2 = "Examples show iterative tool calls plus intermediate reasoning traces."
    steps.append({"thought": thought_2, "action": action_2, "observation": observation_2})

    # Step 3: finalize with synthesis action.
    thought_3 = "I now have enough to draft a concise explanation."
    action_3 = f"compose_answer('{question[:30]}...')"
    observation_3 = "Draft emphasizes loop behavior and error-correction via observations."
    steps.append({"thought": thought_3, "action": action_3, "observation": observation_3})

    return steps


def deep_research_demo() -> List[Evidence]:
    """Create mock evidence entries and score them like a deep research workflow."""
    return [
        Evidence("Paper A", "ReACT improves tool-use reliability", 0.95, 0.88, 0.65),
        Evidence("Blog B", "Deep research should cross-check sources", 0.70, 0.82, 0.92),
        Evidence("Docs C", "Agent traces support auditability", 0.90, 0.76, 0.85),
        Evidence("Forum D", "Long chains can drift without verification", 0.58, 0.67, 0.90),
        Evidence("Benchmark E", "Multi-hop retrieval boosts complex QA", 0.92, 0.93, 0.80),
    ]


def compute_weighted_scores(evidence_list: List[Evidence]) -> Dict[str, float]:
    """Combine credibility, relevance, and recency into one ranking score."""
    scores: Dict[str, float] = {}
    for ev in evidence_list:
        # Weighted sum chosen for readability (not a universal formula).
        score = 0.45 * ev.credibility + 0.40 * ev.relevance + 0.15 * ev.recency
        scores[ev.source] = score
    return scores


def plot_react_timeline(steps: List[Dict[str, str]]) -> Path:
    """Visualize the ReACT loop as a timeline with three phases per step."""
    fig, ax = plt.subplots(figsize=(12, 4.8))

    x = np.arange(len(steps))
    thought_y = np.full_like(x, 3)
    action_y = np.full_like(x, 2)
    observe_y = np.full_like(x, 1)

    ax.plot(x, thought_y, marker="o", linewidth=2.5, label="Thought", color="#1d4ed8")
    ax.plot(x, action_y, marker="s", linewidth=2.5, label="Action", color="#ea580c")
    ax.plot(x, observe_y, marker="^", linewidth=2.5, label="Observation", color="#059669")

    for i, step in enumerate(steps):
        ax.text(i, 3.15, f"T{i+1}: reason", ha="center", fontsize=9)
        ax.text(i, 2.15, f"A{i+1}: tool", ha="center", fontsize=9)
        ax.text(i, 1.15, f"O{i+1}: result", ha="center", fontsize=9)

    ax.set_title("ReACT Loop Timeline: Thought -> Action -> Observation")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Observation", "Action", "Thought"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Iteration {i+1}" for i in x])
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper right")

    out = OUTPUT_DIR / f"{PREFIX}react_timeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_deep_research_scores(evidence_list: List[Evidence], scores: Dict[str, float]) -> Path:
    """Bar chart showing ranked evidence confidence after weighted scoring."""
    ordered = sorted(evidence_list, key=lambda e: scores[e.source], reverse=True)
    labels = [e.source for e in ordered]
    values = [scores[e.source] for e in ordered]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(labels, values, color=["#0f766e", "#2563eb", "#7c3aed", "#d97706", "#dc2626"])

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weighted Confidence")
    ax.set_title("Deep Research Evidence Ranking")
    ax.grid(axis="y", alpha=0.25)

    out = OUTPUT_DIR / f"{PREFIX}deep_research_scores.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_signal_breakdown(evidence_list: List[Evidence]) -> Path:
    """Grouped comparison of credibility/relevance/recency signals per source."""
    labels = [e.source for e in evidence_list]
    credibility = [e.credibility for e in evidence_list]
    relevance = [e.relevance for e in evidence_list]
    recency = [e.recency for e in evidence_list]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(x - w, credibility, w, label="Credibility", color="#0284c7")
    ax.bar(x, relevance, w, label="Relevance", color="#65a30d")
    ax.bar(x + w, recency, w, label="Recency", color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Deep Research Signal Breakdown by Source")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    out = OUTPUT_DIR / f"{PREFIX}deep_research_signal_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def save_run_output(question: str, steps: List[Dict[str, str]], scores: Dict[str, float]) -> Path:
    """Persist a readable text log of what happened during the run."""
    out = OUTPUT_DIR / f"{PREFIX}run_output.txt"

    lines = [
        "ReACT + Deep Research Demo Run",
        "=" * 34,
        f"Question: {question}",
        "",
        "ReACT Trace:",
    ]

    for i, step in enumerate(steps, start=1):
        lines.append(f"Iteration {i}")
        lines.append(f"  Thought     : {step['thought']}")
        lines.append(f"  Action      : {step['action']}")
        lines.append(f"  Observation : {step['observation']}")

    lines.append("")
    lines.append("Deep Research Weighted Scores:")
    for source, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {source}: {score:.3f}")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    question = "What is the difference between a ReACT agent and deep research workflow?"

    steps = react_agent_demo(question)
    evidence_list = deep_research_demo()
    scores = compute_weighted_scores(evidence_list)

    timeline_path = plot_react_timeline(steps)
    ranking_path = plot_deep_research_scores(evidence_list, scores)
    breakdown_path = plot_signal_breakdown(evidence_list)
    log_path = save_run_output(question, steps, scores)

    print("Saved files:")
    print(f"- {timeline_path.name}")
    print(f"- {ranking_path.name}")
    print(f"- {breakdown_path.name}")
    print(f"- {log_path.name}")


if __name__ == "__main__":
    main()
