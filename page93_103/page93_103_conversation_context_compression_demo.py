"""
Page 93-103 demo: Conversation Compression vs Context Compression.

This script builds a small simulation to show:
1) Why conversation history grows across turns.
2) How conversation compression (history summarization) reduces token load.
3) How context compression (document filtering) reduces retrieved context tokens.
4) The quality tradeoff when compression is too aggressive.

Outputs (all prefixed with page93_103_):
- page93_103_token_growth_comparison.png
- page93_103_context_compression_curve.png
- page93_103_quality_vs_compression.png
- page93_103_ascii_flow.txt
- page93_103_run_output.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random

import matplotlib.pyplot as plt


# Keep outputs reproducible so learners get the same chart each run.
random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class Turn:
    """A single conversation turn with user input and assistant output token counts."""

    user_tokens: int
    assistant_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.user_tokens + self.assistant_tokens


def simulate_conversation(turns: int = 12) -> list[Turn]:
    """Create a synthetic conversation where turn lengths vary naturally."""
    history: list[Turn] = []
    for i in range(turns):
        # Mild growth pattern: later turns often include more details/questions.
        base = 120 + i * 12
        user = random.randint(base - 30, base + 25)
        assistant = random.randint(base + 10, base + 70)
        history.append(Turn(user_tokens=user, assistant_tokens=assistant))
    return history


def total_context_without_compression(history: list[Turn]) -> list[int]:
    """At each turn, carry full prior history into the next prompt."""
    running = 0
    sizes = []
    for t in history:
        running += t.total_tokens
        sizes.append(running)
    return sizes


def total_context_with_conversation_compression(
    history: list[Turn],
    keep_recent_turns: int = 3,
    summary_ratio: float = 0.25,
) -> list[int]:
    """
    Compress older turns into a summary while keeping recent turns verbatim.

    summary_ratio=0.25 means old history is compacted to 25% of original tokens.
    """
    sizes = []
    for i in range(len(history)):
        recent = history[max(0, i - keep_recent_turns + 1) : i + 1]
        old = history[: max(0, i - keep_recent_turns + 1)]
        recent_tokens = sum(t.total_tokens for t in recent)
        old_tokens = sum(t.total_tokens for t in old)
        compressed_old = int(old_tokens * summary_ratio)
        sizes.append(recent_tokens + compressed_old)
    return sizes


def context_compression_tokens(
    retrieved_docs_tokens: int,
    compression_levels: list[float],
) -> list[int]:
    """Apply document-context compression ratios to retrieved docs."""
    return [int(retrieved_docs_tokens * lvl) for lvl in compression_levels]


def quality_score(compression_ratio: float) -> float:
    """
    Toy quality model:
    - Moderate compression helps remove noise.
    - Excessive compression starts dropping critical facts.
    """
    noise_reduction_bonus = 15 * math.exp(-((compression_ratio - 0.45) ** 2) / 0.03)
    info_loss_penalty = 35 * max(0.0, 0.30 - compression_ratio)
    baseline = 68
    score = baseline + noise_reduction_bonus - info_loss_penalty
    return max(0.0, min(100.0, score))


def save_ascii_flow(path: Path) -> None:
    """Save an ASCII diagram that mirrors slide concepts."""
    ascii_diagram = r"""
Conversation Compression (chat history)         Context Compression (retrieved docs)

Turn N Prompt                                   Retrieval Prompt
+----------------------------------+            +----------------------------------+
| Summary of old turns (compressed)|            | User question                   |
| Recent turns (full detail)       |            | + top-K docs                    |
+----------------+-----------------+            +----------------+-----------------+
                 |                                              |
                 v                                              v
           LLM generates answer                     Compressor keeps only relevant spans
                 |                                              |
                 +----------------------+-----------------------+
                                        v
                                Shorter effective context
""".strip("\n")
    path.write_text(ascii_diagram + "\n", encoding="utf-8")


def main() -> None:
    history = simulate_conversation(turns=12)

    no_comp = total_context_without_compression(history)
    conv_comp = total_context_with_conversation_compression(
        history,
        keep_recent_turns=3,
        summary_ratio=0.28,
    )

    turns = list(range(1, len(history) + 1))

    # Plot 1: token growth in conversation flow.
    plt.figure(figsize=(9, 5))
    plt.plot(turns, no_comp, marker="o", label="No compression (full history)")
    plt.plot(turns, conv_comp, marker="s", label="Conversation compression")
    plt.title("Conversation Token Growth Across Turns")
    plt.xlabel("Turn")
    plt.ylabel("Prompt Tokens")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p1 = OUTPUT_DIR / "page93_103_token_growth_comparison.png"
    plt.savefig(p1, dpi=150)
    plt.close()

    # Plot 2: context compression on retrieved documents.
    original_retrieval_tokens = 4200
    compression_levels = [1.0, 0.8, 0.6, 0.45, 0.3, 0.2]
    compressed_sizes = context_compression_tokens(
        retrieved_docs_tokens=original_retrieval_tokens,
        compression_levels=compression_levels,
    )

    plt.figure(figsize=(9, 5))
    labels = [f"{int(level * 100)}%" for level in compression_levels]
    plt.bar(labels, compressed_sizes)
    plt.title("Context Compression on Retrieved Docs")
    plt.xlabel("Remaining Context After Compression")
    plt.ylabel("Tokens")
    plt.tight_layout()
    p2 = OUTPUT_DIR / "page93_103_context_compression_curve.png"
    plt.savefig(p2, dpi=150)
    plt.close()

    # Plot 3: quality vs compression ratio (tradeoff curve).
    ratios = [x / 100 for x in range(15, 101, 5)]
    quality = [quality_score(r) for r in ratios]

    plt.figure(figsize=(9, 5))
    plt.plot([r * 100 for r in ratios], quality, marker=".")
    plt.axvline(45, linestyle="--", linewidth=1, label="Often good balance zone")
    plt.title("Quality Tradeoff vs Compression Aggressiveness")
    plt.xlabel("Remaining Context (%)")
    plt.ylabel("Simulated Answer Quality")
    plt.ylim(30, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p3 = OUTPUT_DIR / "page93_103_quality_vs_compression.png"
    plt.savefig(p3, dpi=150)
    plt.close()

    save_ascii_flow(OUTPUT_DIR / "page93_103_ascii_flow.txt")

    run_report = []
    run_report.append("=== Page 93-103 Demo: Conversation & Context Compression ===")
    run_report.append(f"Turns simulated: {len(history)}")
    run_report.append(f"Final prompt tokens without compression: {no_comp[-1]}")
    run_report.append(f"Final prompt tokens with conversation compression: {conv_comp[-1]}")

    reduction_pct = (1 - conv_comp[-1] / no_comp[-1]) * 100
    run_report.append(f"Conversation compression reduction: {reduction_pct:.1f}%")

    run_report.append(f"Original retrieved context tokens: {original_retrieval_tokens}")
    for lvl, size in zip(compression_levels, compressed_sizes):
        run_report.append(f" - keep {int(lvl * 100):>3}% -> {size:>4} tokens")

    best_ratio = max(ratios, key=quality_score)
    run_report.append(f"Peak simulated quality near: {best_ratio * 100:.0f}% remaining context")

    (OUTPUT_DIR / "page93_103_run_output.txt").write_text("\n".join(run_report) + "\n", encoding="utf-8")

    print("\n".join(run_report))
    print("Saved:")
    for path in [p1, p2, p3, OUTPUT_DIR / "page93_103_ascii_flow.txt", OUTPUT_DIR / "page93_103_run_output.txt"]:
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
