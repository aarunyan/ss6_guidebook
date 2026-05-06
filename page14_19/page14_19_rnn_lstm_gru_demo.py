from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

OUTPUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "page14_19_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "page14_19_mplconfig"))

import matplotlib

# Use a file-based backend so the script runs cleanly in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


RUN_LOG = OUTPUT_DIR / "page14_19_run_output.txt"
OUTPUT_PREFIX = OUTPUT_DIR / "page14_19"

# A short sentence with a few strong keywords and a few filler words.
TOKENS = ["the", "cat", "sat", "on", "the", "mat", "while", "purring"]
SIGNALS = np.array([0.10, 1.00, 0.55, 0.15, 0.10, 0.85, 0.20, 0.65], dtype=float)

# A tiny seq2seq example inspired by slides 16-18.
SOURCE_TOKENS = ["I", "saw", "a", "cute", "cat"]
TARGET_TOKENS = ["I", "saw", "a", "cat"]
SOURCE_EMBEDDINGS = np.array(
    [
        [0.10, 0.25],
        [0.35, 0.60],
        [0.05, 0.10],
        [0.82, 0.88],
        [0.95, 0.20],
    ]
)


def log(message: str, lines: List[str]) -> None:
    """Print a message and keep a copy for the saved run transcript."""
    print(message)
    lines.append(message)


def sigmoid(value: float) -> float:
    """Classic logistic gate used by LSTM and GRU."""
    return 1.0 / (1.0 + math.exp(-value))


def ascii_bar(value: float, width: int = 24, filled: str = "#", empty: str = ".") -> str:
    """Convert a value in [-1, 1] or [0, 1] into a text bar for live terminal output."""
    clipped = max(-1.0, min(1.0, value))
    normalized = max(0.0, clipped)
    count = int(round(normalized * width))
    return filled * count + empty * (width - count)


def simulate_rnn(tokens: List[str], signals: np.ndarray) -> List[Dict[str, float]]:
    """A simple vanilla RNN state update."""
    hidden = 0.0
    trace: List[Dict[str, float]] = []

    for step, (token, signal) in enumerate(zip(tokens, signals), start=1):
        hidden = math.tanh(0.58 * hidden + 0.70 * signal - 0.08)
        trace.append(
            {
                "step": step,
                "token": token,
                "signal": float(signal),
                "state": hidden,
            }
        )

    return trace


def simulate_lstm(tokens: List[str], signals: np.ndarray) -> List[Dict[str, float]]:
    """A small hand-crafted LSTM to show forget, input, and output gates."""
    cell = 0.0
    hidden = 0.0
    trace: List[Dict[str, float]] = []

    for step, (token, signal) in enumerate(zip(tokens, signals), start=1):
        forget_gate = sigmoid(1.2 - 1.0 * signal)
        input_gate = sigmoid(-0.5 + 2.4 * signal)
        candidate = math.tanh(1.6 * signal - 0.2)
        cell = forget_gate * cell + input_gate * candidate
        output_gate = sigmoid(0.4 + 1.4 * signal)
        hidden = output_gate * math.tanh(cell)

        trace.append(
            {
                "step": step,
                "token": token,
                "signal": float(signal),
                "forget": forget_gate,
                "input": input_gate,
                "candidate": candidate,
                "cell": cell,
                "output": output_gate,
                "state": hidden,
            }
        )

    return trace


def simulate_gru(tokens: List[str], signals: np.ndarray) -> List[Dict[str, float]]:
    """A small hand-crafted GRU to show update and reset gates."""
    hidden = 0.0
    trace: List[Dict[str, float]] = []

    for step, (token, signal) in enumerate(zip(tokens, signals), start=1):
        update_gate = sigmoid(-0.2 + 2.0 * signal)
        reset_gate = sigmoid(1.2 - 1.8 * signal)
        candidate = math.tanh(reset_gate * hidden + 1.4 * signal - 0.1)
        hidden = (1.0 - update_gate) * hidden + update_gate * candidate

        trace.append(
            {
                "step": step,
                "token": token,
                "signal": float(signal),
                "update": update_gate,
                "reset": reset_gate,
                "candidate": candidate,
                "state": hidden,
            }
        )

    return trace


def seq2seq_encode(source_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Compress several token vectors into one final context vector."""
    states = []
    hidden = np.zeros(source_embeddings.shape[1], dtype=float)
    for embedding in source_embeddings:
        hidden = np.tanh(0.62 * hidden + embedding)
        states.append(hidden.copy())

    return {
        "states": np.vstack(states),
        "context": hidden,
        "mean_context": np.mean(source_embeddings, axis=0),
    }


def print_runtime_table(
    model_name: str,
    trace: List[Dict[str, float]],
    state_key: str,
    extra_keys: List[str],
    lines: List[str],
) -> None:
    """Print a readable runtime trace while the script runs."""
    log("", lines)
    log(model_name, lines)
    log("-" * len(model_name), lines)
    for row in trace:
        extras = " ".join(f"{key}={row[key]:.2f}" for key in extra_keys)
        bar = ascii_bar(row[state_key], width=26)
        log(
            f"t={row['step']:>2} token={row['token']:<8} "
            f"signal={row['signal']:.2f} state={row[state_key]:+.3f} [{bar}] {extras}".rstrip(),
            lines,
        )


def save_memory_trace_plot(
    tokens: List[str],
    signals: np.ndarray,
    rnn_trace: List[Dict[str, float]],
    lstm_trace: List[Dict[str, float]],
    gru_trace: List[Dict[str, float]],
    destination: Path,
) -> None:
    """Plot how each model's state evolves over the same token sequence."""
    steps = np.arange(1, len(tokens) + 1)
    rnn_state = [row["state"] for row in rnn_trace]
    lstm_state = [row["state"] for row in lstm_trace]
    gru_state = [row["state"] for row in gru_trace]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(steps, signals, marker="o", linewidth=2.2, color="#ffb703", label="Token importance")
    ax1.set_title("Toy sentence signal: which words matter most?")
    ax1.set_ylabel("Importance")
    ax1.set_ylim(0, 1.1)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(steps, rnn_state, marker="o", linewidth=2.2, color="#d62828", label="RNN state")
    ax2.plot(steps, lstm_state, marker="o", linewidth=2.2, color="#2a9d8f", label="LSTM hidden state")
    ax2.plot(steps, gru_state, marker="o", linewidth=2.2, color="#4361ee", label="GRU hidden state")
    ax2.set_title("Memory carried forward by each model")
    ax2.set_ylabel("State value")
    ax2.set_xlabel("Token step")
    ax2.set_xticks(steps)
    ax2.set_xticklabels(tokens)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_gate_heatmaps(
    lstm_trace: List[Dict[str, float]],
    gru_trace: List[Dict[str, float]],
    destination: Path,
) -> None:
    """Show the gate values as heatmaps so the control logic is easy to see."""
    lstm_matrix = np.array(
        [[row["forget"], row["input"], row["output"]] for row in lstm_trace],
        dtype=float,
    )
    gru_matrix = np.array(
        [[row["reset"], row["update"]] for row in gru_trace],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cmap = "viridis"

    lstm_image = axes[0].imshow(lstm_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("LSTM gates")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["forget", "input", "output"])
    axes[0].set_yticks(range(len(lstm_trace)))
    axes[0].set_yticklabels([row["token"] for row in lstm_trace])

    gru_image = axes[1].imshow(gru_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("GRU gates")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["reset", "update"])
    axes[1].set_yticks(range(len(gru_trace)))
    axes[1].set_yticklabels([row["token"] for row in gru_trace])

    for ax, matrix in zip(axes, [lstm_matrix, gru_matrix]):
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

    fig.colorbar(lstm_image, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(gru_image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_bottleneck_diagram(
    source_tokens: List[str],
    target_tokens: List[str],
    encoding: Dict[str, np.ndarray],
    destination: Path,
) -> None:
    """Draw a simple encoder -> fixed vector -> decoder diagram."""
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")

    source_x = np.linspace(0.08, 0.40, len(source_tokens))
    target_x = np.linspace(0.62, 0.92, len(target_tokens))
    y = 0.5

    for x, token in zip(source_x, source_tokens):
        rect = plt.Rectangle((x - 0.035, y - 0.07), 0.07, 0.14, facecolor="#d9f0d3", edgecolor="black")
        ax.add_patch(rect)
        ax.text(x, y, token, ha="center", va="center", fontsize=12)

    ax.text(0.24, 0.83, "Encoder reads source tokens one-by-one", ha="center", fontsize=13, color="#2a9d8f")
    ax.annotate("", xy=(0.47, 0.5), xytext=(0.44, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2a9d8f"))

    context = encoding["context"]
    context_label = f"[{context[0]:.2f}, {context[1]:.2f}]"
    bottleneck = plt.Rectangle((0.47, 0.35), 0.08, 0.30, facecolor="#fff3bf", edgecolor="#d62828", linewidth=2.5)
    ax.add_patch(bottleneck)
    ax.text(0.51, 0.53, "Fixed\nvector", ha="center", va="center", fontsize=13, color="#d62828")
    ax.text(0.51, 0.38, context_label, ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(0.60, 0.5), xytext=(0.55, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#d62828"))
    ax.text(0.51, 0.79, "Bottleneck: all source meaning must fit here", ha="center", fontsize=13, color="#d62828")

    for x, token in zip(target_x, target_tokens):
        rect = plt.Rectangle((x - 0.035, y - 0.07), 0.07, 0.14, facecolor="#fde2e4", edgecolor="black")
        ax.add_patch(rect)
        ax.text(x, y, token, ha="center", va="center", fontsize=12)

    ax.text(0.77, 0.83, "Decoder generates target tokens sequentially", ha="center", fontsize=13, color="#6a4c93")
    ax.set_title("Seq2Seq bottleneck from slides 16-18", fontsize=16, pad=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_context_comparison(
    source_tokens: List[str],
    source_embeddings: np.ndarray,
    encoding: Dict[str, np.ndarray],
    destination: Path,
) -> None:
    """Compare all token embeddings with the one final context vector."""
    states = encoding["states"]
    context = encoding["context"]
    mean_context = encoding["mean_context"]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.scatter(source_embeddings[:, 0], source_embeddings[:, 1], s=120, color="#2a9d8f", label="Source token embeddings")
    ax.plot(states[:, 0], states[:, 1], color="#4361ee", linewidth=2, marker="o", label="Encoder states")
    ax.scatter(context[0], context[1], s=220, color="#d62828", marker="X", label="Final context")
    ax.scatter(mean_context[0], mean_context[1], s=160, color="#ffb703", marker="D", label="Mean context")

    for token, (x, y) in zip(source_tokens, source_embeddings):
        ax.text(x + 0.01, y + 0.015, token, fontsize=11)

    ax.set_title("Many source tokens, one compressed summary vector")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_sequential_animation(
    tokens: List[str],
    rnn_trace: List[Dict[str, float]],
    lstm_trace: List[Dict[str, float]],
    gru_trace: List[Dict[str, float]],
    destination: Path,
) -> None:
    """Animate one token arriving at a time to reinforce the sequential nature."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    def update(frame_index: int) -> None:
        ax.clear()
        ax.set_xlim(0, len(tokens) + 1)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")

        current_step = frame_index + 1
        ax.set_title(f"Sequential processing step {current_step}/{len(tokens)}", fontsize=16)

        for index, token in enumerate(tokens, start=1):
            facecolor = "#8ecae6" if index <= current_step else "#d9d9d9"
            rect = plt.Rectangle((index - 0.38, 0.73), 0.76, 0.14, facecolor=facecolor, edgecolor="black")
            ax.add_patch(rect)
            ax.text(index, 0.80, token, ha="center", va="center", fontsize=12)

        ax.text(0.6, 0.52, "RNN", color="#d62828", fontsize=13, fontweight="bold")
        ax.text(0.6, 0.34, "LSTM", color="#2a9d8f", fontsize=13, fontweight="bold")
        ax.text(0.6, 0.16, "GRU", color="#4361ee", fontsize=13, fontweight="bold")

        current_rnn = rnn_trace[frame_index]["state"]
        current_lstm = lstm_trace[frame_index]["state"]
        current_gru = gru_trace[frame_index]["state"]
        for y, value, color in [
            (0.50, current_rnn, "#d62828"),
            (0.32, current_lstm, "#2a9d8f"),
            (0.14, current_gru, "#4361ee"),
        ]:
            width = max(0.02, 0.75 * value)
            rect = plt.Rectangle((1.2, y - 0.045), width, 0.09, facecolor=color, alpha=0.85)
            ax.add_patch(rect)
            ax.text(1.25 + width, y, f"{value:.2f}", va="center", fontsize=12)

        ax.text(
            len(tokens) / 2 + 0.5,
            0.93,
            "Only one new token is processed per step",
            ha="center",
            fontsize=13,
            color="#444444",
        )

    animation = FuncAnimation(fig, update, frames=len(tokens), interval=900, repeat=True)
    animation.save(destination, writer=PillowWriter(fps=1))
    plt.close(fig)


def summarize_bottleneck(lines: List[str], encoding: Dict[str, np.ndarray]) -> None:
    """Print a short numeric summary of the seq2seq context vector."""
    context = encoding["context"]
    mean_context = encoding["mean_context"]
    log("", lines)
    log("Seq2Seq bottleneck snapshot", lines)
    log("--------------------------", lines)
    log(f"Source tokens : {' '.join(SOURCE_TOKENS)}", lines)
    log(f"Target tokens : {' '.join(TARGET_TOKENS)}", lines)
    log(f"Final context : [{context[0]:.3f}, {context[1]:.3f}]", lines)
    log(f"Mean context  : [{mean_context[0]:.3f}, {mean_context[1]:.3f}]", lines)
    log("Observation   : many source words end up squeezed into one final vector.", lines)


def main() -> None:
    lines: List[str] = []

    log("page14_19 RNN / LSTM / GRU demo", lines)
    log("================================", lines)
    log("Sentence under inspection: " + " ".join(TOKENS), lines)
    log("The live bars below show how much state each model carries forward.", lines)

    rnn_trace = simulate_rnn(TOKENS, SIGNALS)
    lstm_trace = simulate_lstm(TOKENS, SIGNALS)
    gru_trace = simulate_gru(TOKENS, SIGNALS)

    print_runtime_table("Vanilla RNN", rnn_trace, "state", [], lines)
    print_runtime_table("LSTM", lstm_trace, "state", ["forget", "input", "output"], lines)
    print_runtime_table("GRU", gru_trace, "state", ["reset", "update"], lines)

    encoding = seq2seq_encode(SOURCE_EMBEDDINGS)
    summarize_bottleneck(lines, encoding)

    save_memory_trace_plot(
        TOKENS,
        SIGNALS,
        rnn_trace,
        lstm_trace,
        gru_trace,
        OUTPUT_PREFIX.with_name("page14_19_memory_trace.png"),
    )
    save_gate_heatmaps(
        lstm_trace,
        gru_trace,
        OUTPUT_PREFIX.with_name("page14_19_gate_heatmap.png"),
    )
    save_bottleneck_diagram(
        SOURCE_TOKENS,
        TARGET_TOKENS,
        encoding,
        OUTPUT_PREFIX.with_name("page14_19_bottleneck.png"),
    )
    save_context_comparison(
        SOURCE_TOKENS,
        SOURCE_EMBEDDINGS,
        encoding,
        OUTPUT_PREFIX.with_name("page14_19_context_map.png"),
    )
    save_sequential_animation(
        TOKENS,
        rnn_trace,
        lstm_trace,
        gru_trace,
        OUTPUT_PREFIX.with_name("page14_19_sequential_flow.gif"),
    )

    log("", lines)
    log("Saved files:", lines)
    for filename in [
        "page14_19_memory_trace.png",
        "page14_19_gate_heatmap.png",
        "page14_19_bottleneck.png",
        "page14_19_context_map.png",
        "page14_19_sequential_flow.gif",
        "page14_19_run_output.txt",
    ]:
        log(f"  - {filename}", lines)

    RUN_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
