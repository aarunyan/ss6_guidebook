"""
page87_92_agentic_memory_demo.py

A compact simulation of agentic-system memory inspired by slides 87-92:
- Memory types: short-term vs long-term
- Memory operations: store, retrieve, update, forget, consolidate
- Edge case: write conflict in multi-agent setting (optimistic lock)

Outputs:
- page87_92_memory_types_timeline.png
- page87_92_memory_operations_flow.png
- page87_92_consolidation_before_after.png
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class MemoryItem:
    """Structured memory record for long-term storage."""

    key: str
    value: str
    category: str  # episodic | semantic | procedural | profile
    version: int
    timestamp: datetime


class AgentMemorySystem:
    """Simple agent memory system with short-term and long-term memory."""

    def __init__(self, short_term_capacity: int = 4) -> None:
        self.short_term_capacity = short_term_capacity
        self.short_term: List[str] = []
        self.long_term: Dict[str, MemoryItem] = {}

    def add_to_short_term(self, event: str) -> None:
        """Add event to short-term memory, evicting oldest when full."""
        self.short_term.append(event)
        if len(self.short_term) > self.short_term_capacity:
            self.short_term.pop(0)

    def store(self, key: str, value: str, category: str) -> None:
        """Store a new long-term memory item."""
        now = datetime.now()
        self.long_term[key] = MemoryItem(
            key=key,
            value=value,
            category=category,
            version=1,
            timestamp=now,
        )
        print(f"STORE  : {key} -> {value} ({category})")

    def retrieve(self, key: str) -> Optional[MemoryItem]:
        """Retrieve long-term memory by key."""
        item = self.long_term.get(key)
        if item:
            print(f"RETRIEVE: {key} -> {item.value} (v{item.version})")
        else:
            print(f"RETRIEVE: {key} -> NOT FOUND")
        return item

    def update(self, key: str, new_value: str) -> bool:
        """Update existing long-term memory item."""
        if key not in self.long_term:
            print(f"UPDATE : {key} -> FAILED (missing key)")
            return False

        item = self.long_term[key]
        item.value = new_value
        item.version += 1
        item.timestamp = datetime.now()
        print(f"UPDATE : {key} -> {new_value} (v{item.version})")
        return True

    def forget(self, key: str) -> bool:
        """Intentional deletion from long-term memory."""
        if key in self.long_term:
            del self.long_term[key]
            print(f"FORGET : {key} deleted")
            return True
        print(f"FORGET : {key} -> nothing to delete")
        return False

    def consolidate_events_to_semantic(self, source_keys: List[str], target_key: str) -> None:
        """Consolidate several episodic events into one semantic summary."""
        collected = [self.long_term[k].value for k in source_keys if k in self.long_term]
        if not collected:
            print("CONSOLIDATE: no source events found")
            return

        summary = " | ".join(collected)
        if target_key in self.long_term:
            self.update(target_key, summary)
        else:
            self.store(target_key, summary, category="semantic")
        print(f"CONSOLIDATE: {source_keys} -> {target_key}")

    def safe_update_with_version(self, key: str, proposed_value: str, expected_version: int) -> bool:
        """Simulate optimistic locking to handle multi-agent write conflicts."""
        current = self.long_term.get(key)
        if current is None:
            print(f"SAFE_UPDATE: {key} missing")
            return False

        if current.version != expected_version:
            print(
                f"SAFE_UPDATE: conflict on {key} "
                f"(expected v{expected_version}, found v{current.version})"
            )
            return False

        current.value = proposed_value
        current.version += 1
        current.timestamp = datetime.now()
        print(f"SAFE_UPDATE: {key} updated to '{proposed_value}' (v{current.version})")
        return True


def create_visual_memory_timeline(path: str) -> None:
    """Visualize short-term vs long-term persistence over time."""
    x = list(range(1, 11))
    short_term_signal = [1 if i >= 7 else 0 for i in x]  # survives only recent turns
    long_term_signal = [1 for _ in x]  # persists across timeline

    plt.figure(figsize=(10, 4))
    plt.step(x, short_term_signal, where="mid", label="Short-term (session-limited)", linewidth=2)
    plt.step(x, long_term_signal, where="mid", label="Long-term (cross-session)", linewidth=2)
    plt.ylim(-0.1, 1.2)
    plt.yticks([0, 1], ["Absent", "Available"])
    plt.xlabel("Conversation Turn")
    plt.ylabel("Memory Availability")
    plt.title("Agentic Memory: Short-term vs Long-term")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def create_visual_operations_flow(path: str) -> None:
    """Visualize main memory operations as a simple flow."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    labels = ["Store", "Retrieve", "Update", "Forget", "Consolidate"]
    xs = [0.07, 0.28, 0.48, 0.67, 0.85]

    for x, label in zip(xs, labels):
        rect = plt.Rectangle((x - 0.075, 0.42), 0.15, 0.2, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.52, label, ha="center", va="center", fontsize=11)

    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - 0.09, 0.52),
            xytext=(xs[i] + 0.09, 0.52),
            arrowprops=dict(arrowstyle="->", lw=1.8),
        )

    ax.text(0.5, 0.82, "Memory Operations Pipeline", ha="center", fontsize=13, fontweight="bold")
    ax.text(
        0.5,
        0.2,
        "Typical loop: new input -> write/read memory -> refine memory quality",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def create_visual_consolidation(path: str, episodic_count: int, semantic_count: int) -> None:
    """Visualize consolidation (many events -> fewer summarized facts)."""
    labels = ["Episodic Items", "Semantic Summaries"]
    values = [episodic_count, semantic_count]
    colors = ["#5B8FF9", "#5AD8A6"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.05, str(val), ha="center")
    plt.ylim(0, max(values) + 1)
    plt.ylabel("Count")
    plt.title("Consolidation: Episodic -> Semantic Memory")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_demo() -> None:
    system = AgentMemorySystem(short_term_capacity=4)

    print("=== 1) Short-term memory behavior ===")
    turns = [
        "User says their preferred nickname is Arthur.",
        "User asks for a summary of legal documents.",
        "User corrects nickname to Kui Kui.",
        "User asks the deadline again.",
        "User asks to forget the old nickname.",
    ]
    for t in turns:
        system.add_to_short_term(t)
        print(f"SHORT_TERM_APPEND: {t}")
    print("SHORT_TERM_SNAPSHOT:")
    for i, item in enumerate(system.short_term, start=1):
        print(f"  {i}. {item}")

    print("\n=== 2) Long-term memory operations ===")
    system.store("user_alias", "Arthur", "profile")
    system.retrieve("user_alias")
    system.update("user_alias", "Kui Kui")
    system.retrieve("user_alias")

    system.store("event_1", "User works in legal ops.", "episodic")
    system.store("event_2", "User often asks for contract summaries.", "episodic")
    system.store("event_3", "User prefers bullet-point answers.", "episodic")

    system.consolidate_events_to_semantic(
        source_keys=["event_1", "event_2", "event_3"],
        target_key="user_work_style",
    )
    system.retrieve("user_work_style")

    system.forget("event_2")
    system.retrieve("event_2")

    print("\n=== 3) Edge case: multi-agent write conflict ===")
    # Both agents read the same version number at the start.
    base_version = system.long_term["user_alias"].version
    print(f"AGENT_A and AGENT_B both read user_alias at version {base_version}")

    agent_a_ok = system.safe_update_with_version("user_alias", "Kui", expected_version=base_version)
    agent_b_ok = system.safe_update_with_version(
        "user_alias", "Mr. Kui", expected_version=base_version
    )

    if agent_a_ok and not agent_b_ok:
        print("Conflict handled: Agent B must re-read then retry.")

    print("\n=== 4) Create visualizations ===")
    create_visual_memory_timeline("page87_92_memory_types_timeline.png")
    print("Saved: page87_92_memory_types_timeline.png")

    create_visual_operations_flow("page87_92_memory_operations_flow.png")
    print("Saved: page87_92_memory_operations_flow.png")

    episodic_count = sum(1 for v in system.long_term.values() if v.category == "episodic")
    semantic_count = sum(1 for v in system.long_term.values() if v.category == "semantic")
    create_visual_consolidation(
        "page87_92_consolidation_before_after.png",
        episodic_count=episodic_count + 2,  # approximate "before" count for educational contrast
        semantic_count=semantic_count,
    )
    print("Saved: page87_92_consolidation_before_after.png")

    print("\nDemo finished.")


if __name__ == "__main__":
    run_demo()
