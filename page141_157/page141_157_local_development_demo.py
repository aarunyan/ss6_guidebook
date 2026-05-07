"""Local Development demo based on slides 141-157.

This script simulates:
1) Quantization trade-offs (memory, speed, quality)
2) Local inference backend selection
3) vLLM-style paged KV cache efficiency

It saves charts and a run summary in the same folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


@dataclass
class QuantOption:
    """Represents one quantization format and its rough characteristics."""

    name: str
    bits: int
    quality_drop: float  # Lower is better; value is an estimated drop from FP16 baseline.
    throughput_gain: float  # Relative speedup over FP16 baseline.


def estimate_memory_gb(param_count_billion: float, bits: int) -> float:
    """Estimate model weight memory (GB) from parameter count and bit-width.

    Formula:
    params * bits / 8 => bytes
    bytes / 1e9 => GB
    """
    params = param_count_billion * 1_000_000_000
    return (params * bits / 8) / 1_000_000_000


def simulate_quantization_table() -> list[dict]:
    """Build a small comparison table similar to the slides."""
    options = [
        QuantOption("FP16", 16, 0.0, 1.0),
        QuantOption("GGUF Q4_K_M", 4, 1.2, 2.0),
        QuantOption("GPTQ 4-bit", 4, 2.2, 2.3),
        QuantOption("AWQ 4-bit", 4, 0.8, 2.1),
    ]

    model_size_b = 70.0  # Inspired by LLaMA-scale example in the slides.
    rows: list[dict] = []
    for opt in options:
        rows.append(
            {
                "format": opt.name,
                "bits": opt.bits,
                "memory_gb": round(estimate_memory_gb(model_size_b, opt.bits), 1),
                "quality_drop": opt.quality_drop,
                "throughput_gain": opt.throughput_gain,
            }
        )
    return rows


def plot_quantization_tradeoff(rows: list[dict]) -> Path:
    """Create a scatter plot for quality vs memory trade-off."""
    names = [r["format"] for r in rows]
    memory = np.array([r["memory_gb"] for r in rows])
    drop = np.array([r["quality_drop"] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(memory, drop, s=180)

    for i, name in enumerate(names):
        ax.annotate(name, (memory[i], drop[i]), textcoords="offset points", xytext=(8, 6))

    ax.set_title("Quantization Trade-off: Memory vs Quality Drop")
    ax.set_xlabel("Estimated Weight Memory (GB)")
    ax.set_ylabel("Quality Drop vs FP16 (lower is better)")
    ax.grid(alpha=0.3)

    out = OUT_DIR / "page141_157_quantization_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_backend_choice() -> Path:
    """Visualize local inference options from the slides."""
    tools = ["Ollama", "llama.cpp", "vLLM", "TGI", "LM Studio"]
    speed = np.array([3, 4, 5, 4, 3])
    ease = np.array([5, 3, 3, 3, 5])

    x = np.arange(len(tools))
    w = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, speed, width=w, label="Speed", color="#f5a524")
    ax.bar(x + w / 2, ease, width=w, label="Ease", color="#3fb950")

    ax.set_xticks(x)
    ax.set_xticklabels(tools)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Local Inference Options: Speed vs Ease")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    out = OUT_DIR / "page141_157_local_inference_options.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def simulate_paged_attention_efficiency(seed: int = 7) -> dict:
    """Compare naive KV reservation vs paged allocation.

    Naive: reserve max tokens for every request.
    Paged: allocate only pages actually used.
    """
    rng = np.random.default_rng(seed)
    request_count = 800
    max_seq_len = 4096
    page_size = 128

    # Simulate realistic traffic: many short requests, some long ones.
    lengths = rng.integers(low=64, high=max_seq_len + 1, size=request_count)

    naive_reserved_tokens = request_count * max_seq_len
    actual_tokens_used = int(lengths.sum())

    pages_used = np.ceil(lengths / page_size).astype(int)
    paged_reserved_tokens = int((pages_used * page_size).sum())

    naive_waste = naive_reserved_tokens - actual_tokens_used
    paged_waste = paged_reserved_tokens - actual_tokens_used

    utilization_naive = actual_tokens_used / naive_reserved_tokens
    utilization_paged = actual_tokens_used / paged_reserved_tokens

    return {
        "request_count": request_count,
        "max_seq_len": max_seq_len,
        "page_size": page_size,
        "actual_tokens_used": actual_tokens_used,
        "naive_reserved_tokens": naive_reserved_tokens,
        "paged_reserved_tokens": paged_reserved_tokens,
        "naive_waste_tokens": naive_waste,
        "paged_waste_tokens": paged_waste,
        "utilization_naive": round(utilization_naive, 4),
        "utilization_paged": round(utilization_paged, 4),
        "efficiency_gain_x": round(utilization_paged / utilization_naive, 2),
    }


def plot_paged_attention(eff: dict) -> Path:
    """Plot reserved token volume for naive vs paged strategies."""
    labels = ["Naive reserve", "Paged reserve", "Actual used"]
    values = [
        eff["naive_reserved_tokens"],
        eff["paged_reserved_tokens"],
        eff["actual_tokens_used"],
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=["#d73a49", "#f5a524", "#2da44e"])
    ax.set_title("KV Cache Allocation: Naive vs Paged")
    ax.set_ylabel("Tokens")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")

    out = OUT_DIR / "page141_157_paged_attention_efficiency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def pick_tool(scenario: str) -> str:
    """Simple rule-based recommendation from the slide guidance."""
    s = scenario.strip().lower()
    if "easy" in s or "beginner" in s or "chat" in s:
        return "Ollama"
    if "desktop" in s or "gui" in s:
        return "LM Studio"
    if "embedded" in s or "c++" in s:
        return "llama.cpp"
    if "api" in s or "production" in s or "throughput" in s:
        return "vLLM"
    if "enterprise" in s or "kubernetes" in s:
        return "TGI"
    return "Ollama"


def main() -> None:
    rows = simulate_quantization_table()
    eff = simulate_paged_attention_efficiency()

    q_plot = plot_quantization_tradeoff(rows)
    b_plot = plot_backend_choice()
    p_plot = plot_paged_attention(eff)

    demo_scenarios = {
        "I am a beginner and want local chat quickly": pick_tool("beginner local chat"),
        "I need high-throughput API serving": pick_tool("production api throughput"),
        "I need an embeddable C++ runtime": pick_tool("embedded c++"),
    }

    result = {
        "quantization_rows": rows,
        "paged_attention": eff,
        "tool_recommendations": demo_scenarios,
        "generated_plots": [str(q_plot.name), str(b_plot.name), str(p_plot.name)],
    }

    json_out = OUT_DIR / "page141_157_run_output.json"
    txt_out = OUT_DIR / "page141_157_run_output.txt"

    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "Local Development Demo (Slides 141-157)",
        "=" * 45,
        "",
        "Quantization summary:",
    ]
    for r in rows:
        lines.append(
            f"- {r['format']}: {r['bits']} bit, {r['memory_gb']} GB, quality_drop={r['quality_drop']}, speed_x={r['throughput_gain']}"
        )

    lines += [
        "",
        "PagedAttention simulation:",
        f"- Utilization naive: {eff['utilization_naive']}",
        f"- Utilization paged: {eff['utilization_paged']}",
        f"- Efficiency gain: {eff['efficiency_gain_x']}x",
        "",
        "Tool recommendations:",
    ]
    for k, v in demo_scenarios.items():
        lines.append(f"- {k} -> {v}")

    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {json_out}")
    print(f"Saved: {txt_out}")
    print(f"Saved: {q_plot}")
    print(f"Saved: {b_plot}")
    print(f"Saved: {p_plot}")


if __name__ == "__main__":
    main()
