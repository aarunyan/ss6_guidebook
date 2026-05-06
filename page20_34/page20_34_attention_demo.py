from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

OUTPUT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = Path("/private/tmp/page20_34_mplconfig")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


OUTPUT_PREFIX = OUTPUT_DIR / "page20_34"
RUN_LOG = OUTPUT_DIR / "page20_34_run_output.txt"

# Slide 23 uses this sentence to explain how a pronoun can focus on the right word.
TOKENS = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]

# Each token gets a tiny feature vector.
# Dimensions:
# 0 = subject / entity
# 1 = movement / action
# 2 = location
# 3 = pronoun-ness
# 4 = tired / condition
EMBEDDINGS = np.array(
    [
        [0.05, 0.00, 0.00, 0.00, 0.00],  # The
        [1.00, 0.10, 0.00, 0.00, 0.35],  # animal
        [0.00, -0.70, 0.00, 0.00, 0.00],  # didn't
        [0.15, 0.95, 0.00, 0.00, 0.10],  # cross
        [0.05, 0.00, 0.00, 0.00, 0.00],  # the
        [0.00, 0.15, 1.00, 0.00, 0.00],  # street
        [0.00, 0.05, 0.00, 0.00, 0.10],  # because
        [0.70, 0.00, 0.00, 0.35, 0.20],  # it
        [0.00, 0.00, 0.00, 0.00, 0.20],  # was
        [0.00, 0.00, 0.00, 0.00, 0.35],  # too
        [0.25, 0.00, 0.00, 0.00, 1.00],  # tired
    ],
    dtype=float,
)


def log(message: str, lines: List[str]) -> None:
    """Print a message now and preserve it for the saved transcript."""
    print(message)
    lines.append(message)


def softmax(values: np.ndarray) -> np.ndarray:
    """Stable softmax for turning raw scores into attention weights."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def ascii_bar(value: float, width: int = 28) -> str:
    """Turn a probability into a simple text bar."""
    clipped = max(0.0, min(1.0, value))
    count = int(round(clipped * width))
    return "#" * count + "." * (width - count)


def build_qkv(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Create query, key, and value vectors with hand-crafted matrices."""
    w_query = np.array(
        [
            [1.10, 0.10, 0.00, 0.70],
            [0.15, 1.00, 0.10, 0.10],
            [0.00, 0.10, 1.10, 0.00],
            [0.90, 0.00, 0.00, 1.20],
            [0.75, 0.05, 0.00, 0.85],
        ]
    )
    w_key = np.array(
        [
            [1.30, 0.15, 0.00, 0.35],
            [0.10, 1.00, 0.10, 0.05],
            [0.00, 0.10, 1.20, 0.00],
            [0.55, 0.00, 0.00, 1.00],
            [1.10, 0.00, 0.00, 0.25],
        ]
    )
    w_value = np.array(
        [
            [1.00, 0.00, 0.00, 0.25],
            [0.00, 1.00, 0.00, 0.10],
            [0.00, 0.00, 1.00, 0.10],
            [0.40, 0.00, 0.00, 0.80],
            [0.80, 0.00, 0.00, 0.65],
        ]
    )

    return {
        "Q": embeddings @ w_query,
        "K": embeddings @ w_key,
        "V": embeddings @ w_value,
    }


def scaled_dot_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    causal_mask: bool = False,
) -> Dict[str, np.ndarray]:
    """Run scaled dot-product attention and optionally hide future tokens."""
    scale = math.sqrt(keys.shape[1])
    scores = (queries @ keys.T) / scale

    if causal_mask:
        mask = np.triu(np.ones_like(scores), k=1)
        scores = np.where(mask == 1, -1e9, scores)

    weights = np.vstack([softmax(row) for row in scores])
    mixed = weights @ values
    return {"scores": scores, "weights": weights, "mixed": mixed}


def print_attention_walkthrough(
    tokens: List[str],
    scores: np.ndarray,
    weights: np.ndarray,
    query_index: int,
    lines: List[str],
) -> None:
    """Print a live explanation for one token's attention calculation."""
    query_token = tokens[query_index]
    log(f"Query token: {query_token}", lines)
    log("It compares itself with every other token and then turns the scores into probabilities.", lines)
    log("", lines)

    for key_index, token in enumerate(tokens):
        score = float(scores[query_index, key_index])
        weight = float(weights[query_index, key_index])
        bar = ascii_bar(weight)
        log(
            f"looks at {token:<8} raw_score={score:+.3f} attention={weight:.3f} [{bar}]",
            lines,
        )


def save_heatmap(tokens: List[str], weights: np.ndarray, destination: Path) -> None:
    """Save the full self-attention matrix."""
    fig, ax = plt.subplots(figsize=(10.5, 8))
    image = ax.imshow(weights, cmap="YlOrRd", vmin=0.0, vmax=max(0.35, float(weights.max())))
    ax.set_title("Self-attention heatmap: each row asks where to focus")
    ax.set_xlabel("Key token being attended to")
    ax.set_ylabel("Query token doing the looking")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)

    for row_index in range(weights.shape[0]):
        for col_index in range(weights.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{weights[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(image, ax=ax, shrink=0.85, label="Attention weight")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_pronoun_focus(tokens: List[str], weights: np.ndarray, query_index: int, destination: Path) -> None:
    """Show one row of attention more clearly."""
    row = weights[query_index]
    colors = ["#d9d9d9"] * len(tokens)
    colors[1] = "#2a9d8f"  # animal
    colors[query_index] = "#e76f51"  # it

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(tokens, row, color=colors)
    ax.set_title('Where does "it" focus?')
    ax.set_ylabel("Attention weight")
    ax.set_ylim(0, max(0.45, float(np.max(row) + 0.05)))
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(row):
        ax.text(idx, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_context_vectors(mixed: np.ndarray, destination: Path) -> None:
    """Plot how attention mixes information into new context vectors."""
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    image = ax.imshow(mixed.T, cmap="Blues", aspect="auto")
    ax.set_title("Context vectors after attention mixing")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mixed feature dimension")
    ax.set_xticks(range(len(TOKENS)))
    ax.set_xticklabels(TOKENS, rotation=45, ha="right")
    ax.set_yticks(range(mixed.shape[1]))
    ax.set_yticklabels([f"dim {idx}" for idx in range(mixed.shape[1])])
    fig.colorbar(image, ax=ax, shrink=0.85, label="Mixed value")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_transformer_variants(destination: Path) -> None:
    """Draw the three architecture patterns from slides 29-31."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    variants = [
        ("Encoder-only", ["reads both sides", "understands context", "BERT-like"]),
        ("Decoder-only", ["masked attention", "left context only", "GPT-like"]),
        ("Encoder-Decoder", ["input understood first", "output generated next", "T5/BART-like"]),
    ]

    for ax, (title, bullet_lines) in zip(axes, variants):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((1.0, 6.4), 8.0, 2.0, facecolor="#d8f3dc", edgecolor="#2d6a4f", lw=2))
        ax.text(5.0, 7.4, title, ha="center", va="center", fontsize=13, weight="bold")

        if title == "Encoder-only":
            ax.annotate("", xy=(8.4, 5.0), xytext=(1.6, 5.0), arrowprops={"arrowstyle": "<->", "lw": 2})
        elif title == "Decoder-only":
            ax.annotate("", xy=(8.4, 5.0), xytext=(1.6, 5.0), arrowprops={"arrowstyle": "->", "lw": 2})
            ax.text(5.0, 4.2, "future hidden", ha="center", fontsize=10, color="#a11d33")
        else:
            ax.add_patch(plt.Rectangle((1.0, 6.4), 3.2, 2.0, facecolor="#d8f3dc", edgecolor="#2d6a4f", lw=2))
            ax.add_patch(plt.Rectangle((5.8, 2.2), 3.2, 2.0, facecolor="#fde4cf", edgecolor="#bc6c25", lw=2))
            ax.text(2.6, 7.4, "Encoder", ha="center", va="center", fontsize=12, weight="bold")
            ax.text(7.4, 3.2, "Decoder", ha="center", va="center", fontsize=12, weight="bold")
            ax.annotate("", xy=(6.0, 4.5), xytext=(4.1, 6.4), arrowprops={"arrowstyle": "->", "lw": 2})

        for offset, line in enumerate(bullet_lines):
            ax.text(5.0, 1.4 - offset * 0.55, line, ha="center", va="center", fontsize=10)

    fig.suptitle("Transformer architecture families", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_mask_comparison(tokens: List[str], weights: np.ndarray, masked_weights: np.ndarray, destination: Path) -> None:
    """Contrast bidirectional attention with causal decoder attention."""
    query_index = TOKENS.index("it")
    indices = np.arange(len(tokens))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(indices - width / 2, weights[query_index], width=width, label="Bidirectional", color="#457b9d")
    ax.bar(indices + width / 2, masked_weights[query_index], width=width, label="Masked causal", color="#e63946")
    ax.set_title('Same token, different rules: encoder-style vs decoder-style attention')
    ax.set_ylabel("Attention weight")
    ax.set_xticks(indices)
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_attention_animation(tokens: List[str], weights: np.ndarray, query_index: int, destination: Path) -> None:
    """Animate one query token scanning across all possible keys."""
    row = weights[query_index]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    def draw(frame: int) -> None:
        ax.clear()
        colors = ["#bde0fe"] * len(tokens)
        colors[frame] = "#ffb703"
        colors[query_index] = "#e76f51"
        ax.bar(tokens, row, color=colors)
        ax.set_ylim(0, max(0.45, float(np.max(row) + 0.05)))
        ax.set_title(f'Query "it" scanning token {frame + 1}/{len(tokens)}: {tokens[frame]}')
        ax.set_ylabel("Attention weight")
        ax.grid(axis="y", alpha=0.25)

        for idx, value in enumerate(row):
            ax.text(idx, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    animation = FuncAnimation(fig, draw, frames=len(tokens), interval=750, repeat=True)
    animation.save(destination, writer=PillowWriter(fps=1))
    plt.close(fig)


def main() -> None:
    lines: List[str] = []

    log("Attention demo aligned to slides 20-34", lines)
    log("=" * 38, lines)
    log("", lines)
    log('Sentence: "The animal didn\'t cross the street because it was too tired"', lines)
    log("Goal: show why the token 'it' focuses on the right context instead of relying on one bottleneck vector.", lines)

    qkv = build_qkv(EMBEDDINGS)
    attention = scaled_dot_attention(qkv["Q"], qkv["K"], qkv["V"], causal_mask=False)
    masked_attention = scaled_dot_attention(qkv["Q"], qkv["K"], qkv["V"], causal_mask=True)

    query_index = TOKENS.index("it")

    log("", lines)
    print_attention_walkthrough(TOKENS, attention["scores"], attention["weights"], query_index, lines)

    log("", lines)
    log("Top 3 places where 'it' looks:", lines)
    ranked = np.argsort(attention["weights"][query_index])[::-1][:3]
    for idx in ranked:
        weight = float(attention["weights"][query_index, idx])
        log(f"  {TOKENS[idx]:<8} -> {weight:.3f}", lines)

    log("", lines)
    log("Decoder-style masked attention cannot look ahead to future tokens.", lines)
    log(
        f'Bidirectional attention from "it" to "tired": {attention["weights"][query_index, TOKENS.index("tired")]:.3f}',
        lines,
    )
    log(
        f'Masked attention from "it" to "tired": {masked_attention["weights"][query_index, TOKENS.index("tired")]:.3f}',
        lines,
    )

    save_heatmap(TOKENS, attention["weights"], OUTPUT_PREFIX.with_name("page20_34_attention_heatmap.png"))
    save_pronoun_focus(
        TOKENS,
        attention["weights"],
        query_index,
        OUTPUT_PREFIX.with_name("page20_34_it_focus.png"),
    )
    save_context_vectors(attention["mixed"], OUTPUT_PREFIX.with_name("page20_34_context_vectors.png"))
    save_transformer_variants(OUTPUT_PREFIX.with_name("page20_34_transformer_variants.png"))
    save_mask_comparison(
        TOKENS,
        attention["weights"],
        masked_attention["weights"],
        OUTPUT_PREFIX.with_name("page20_34_mask_comparison.png"),
    )
    save_attention_animation(
        TOKENS,
        attention["weights"],
        query_index,
        OUTPUT_PREFIX.with_name("page20_34_attention_scan.gif"),
    )

    log("", lines)
    log("Saved files:", lines)
    log("  page20_34_attention_heatmap.png", lines)
    log("  page20_34_it_focus.png", lines)
    log("  page20_34_context_vectors.png", lines)
    log("  page20_34_transformer_variants.png", lines)
    log("  page20_34_mask_comparison.png", lines)
    log("  page20_34_attention_scan.gif", lines)
    log("  page20_34_run_output.txt", lines)
    RUN_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
