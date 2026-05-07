"""
page104_109_query_optimization_demo.py

A self-contained simulation of Query Optimization techniques from slides 104-109:
- Query rewriting
- HyDE-like expansion
- Step-back prompting
- Sub-question decomposition
- Multi-query retrieval with Reciprocal Rank Fusion (RRF)
- Semantic query routing (SQL DB vs Vector DB vs Web)

This script produces:
1) Console output showing each optimization step
2) Visualization files to compare retrieval quality before/after optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Tiny in-memory knowledge base
# -----------------------------

DOCUMENTS: Dict[str, str] = {
    "doc_runtime_1": "Python runtime errors include NameError, TypeError, and IndexError.",
    "doc_runtime_2": "Debugging traceback starts by reading the last exception line and stack frame.",
    "doc_runtime_3": "Common syntax errors come from missing colons and unmatched parentheses.",
    "doc_vector_1": "Dense embedding retrieval helps when user wording differs from source docs.",
    "doc_finance_1": "Financial crises often involve leverage, asset bubbles, and liquidity shocks.",
    "doc_finance_2": "The 2008 crisis involved subprime mortgages, securitization risk, and weak oversight.",
    "doc_sql_1": "Weekly speed stats are stored in SQL tables with columns week, speed, and lane.",
    "doc_web_1": "Real-time market news is best fetched from web APIs, not static local documents.",
}


# -----------------------------
# Simple retrieval primitives
# -----------------------------

def normalize(text: str) -> List[str]:
    """Lowercase + basic tokenization for a toy lexical retriever."""
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def lexical_score(query: str, doc: str) -> float:
    """Token overlap ratio (Jaccard-like) used as a lightweight relevance proxy."""
    q = set(normalize(query))
    d = set(normalize(doc))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def retrieve(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Rank documents by lexical score and return top-k."""
    scored = [(doc_id, lexical_score(query, text)) for doc_id, text in DOCUMENTS.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# -----------------------------
# Query optimization techniques
# -----------------------------

def rewrite_query(user_query: str) -> str:
    """
    Query rewriting: make an ambiguous query more explicit.
    This mirrors the slide's idea: ask better before searching.
    """
    if "python error fix" in user_query.lower():
        return "common python runtime errors and traceback debugging solutions"
    return user_query


def hyde_paragraph(user_query: str) -> str:
    """
    HyDE-style expansion: generate a hypothetical answer paragraph,
    then use it as semantic search text.
    """
    if "python" in user_query.lower() and "error" in user_query.lower():
        return (
            "A Python runtime error often happens due to wrong types, undefined names, "
            "or bad indexing. Use traceback analysis, inspect stack frames, and test a "
            "minimal reproducible example to isolate and fix the root cause."
        )
    return f"Detailed explanation for: {user_query}"


def step_back_query(user_query: str) -> str:
    """Generalize specific question into a broader concept search."""
    if "2008" in user_query:
        return "What are common causes of financial crises?"
    return user_query


def decompose_subqueries(user_query: str) -> List[str]:
    """Split comparison requests into separate focused queries."""
    lower = user_query.lower()
    if "compare" in lower and " vs " in lower:
        parts = [p.strip() for p in user_query.split("vs")]
        return [f"details about {p}" for p in parts if p]
    return [user_query]


# -----------------------------
# Query routing + RRF fusion
# -----------------------------

class Route:
    SQL_DB = "sql_db"
    VECTOR_DB = "vector_db"
    WEB_SEARCH = "web"


def route_query(user_query: str) -> str:
    """
    Semantic router:
    - Structured stats/metrics -> SQL
    - Real-time/current info -> Web
    - Everything else -> Vector/semantic docs
    """
    q = user_query.lower()
    if any(w in q for w in ["stat", "stats", "table", "weekly", "speed"]):
        return Route.SQL_DB
    if any(w in q for w in ["today", "latest", "current", "real-time", "news"]):
        return Route.WEB_SEARCH
    return Route.VECTOR_DB


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    RRF: score(doc) = sum(1 / (k + rank_position)).
    Great for merging multiple ranked lists without calibration.
    """
    scores: Dict[str, float] = {}
    for ranking in rank_lists:
        for idx, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + idx)
    fused = list(scores.items())
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# -----------------------------
# Experiment harness
# -----------------------------

@dataclass
class ExperimentResult:
    name: str
    top_docs: List[Tuple[str, float]]


def top_ids(results: List[Tuple[str, float]], n: int = 5) -> List[str]:
    return [doc_id for doc_id, _ in results[:n]]


def avg_topk_score(results: List[Tuple[str, float]], n: int = 5) -> float:
    vals = [score for _, score in results[:n]]
    return float(np.mean(vals)) if vals else 0.0


def plot_quality_comparison(metrics: Dict[str, float], output_path: str) -> None:
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.title("Retrieval Quality Proxy (Average Top-5 Lexical Score)")
    plt.ylabel("Average Score")
    plt.ylim(0, max(values) * 1.2 if values else 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_routing_distribution(routes: List[str], output_path: str) -> None:
    route_labels = [Route.SQL_DB, Route.VECTOR_DB, Route.WEB_SEARCH]
    counts = [routes.count(r) for r in route_labels]

    plt.figure(figsize=(7, 5))
    plt.bar(route_labels, counts)
    plt.title("Semantic Routing Decisions")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    # 1) Baseline ambiguous query
    original_query = "python error fix"
    baseline = retrieve(original_query)

    # 2) Rewrite + HyDE expansion
    rewritten = rewrite_query(original_query)
    hyde_text = hyde_paragraph(original_query)
    rewritten_result = retrieve(rewritten)
    hyde_result = retrieve(hyde_text)

    # 3) Multi-query generation (simulated variants)
    multi_queries = [
        original_query,
        rewritten,
        "python traceback debugging",
        "common python runtime errors",
    ]
    rank_lists = [top_ids(retrieve(q), n=5) for q in multi_queries]
    fused = reciprocal_rank_fusion(rank_lists)

    # 4) Step-back example
    specific_finance = "What caused the 2008 financial crisis?"
    generalized_finance = step_back_query(specific_finance)

    # 5) Sub-question decomposition example
    compare_query = "Compare Python NameError vs TypeError vs IndexError"
    sub_queries = decompose_subqueries(compare_query)

    # 6) Semantic query routing examples
    route_inputs = [
        "Special Week speed stats",
        "Special Week lore",
        "latest market news today",
        original_query,
    ]
    route_outputs = [route_query(q) for q in route_inputs]

    # Console report
    print("=== Query Optimization Demo (Slides 104-109) ===")
    print(f"Original query: {original_query}")
    print("\n[Baseline Top-5]")
    for doc_id, score in baseline:
        print(f"- {doc_id:15s} score={score:.3f}")

    print("\n[Rewrite]")
    print(f"Rewritten query: {rewritten}")
    for doc_id, score in rewritten_result:
        print(f"- {doc_id:15s} score={score:.3f}")

    print("\n[HyDE Expansion]")
    print(f"HyDE paragraph: {hyde_text}")
    for doc_id, score in hyde_result:
        print(f"- {doc_id:15s} score={score:.3f}")

    print("\n[Multi-query + RRF fused Top-5]")
    for doc_id, score in fused[:5]:
        print(f"- {doc_id:15s} rrf_score={score:.4f}")

    print("\n[Step-back Prompting]")
    print(f"Specific:    {specific_finance}")
    print(f"Generalized: {generalized_finance}")

    print("\n[Sub-question Decomposition]")
    for idx, sq in enumerate(sub_queries, start=1):
        print(f"{idx}. {sq}")

    print("\n[Semantic Routing]")
    for q, r in zip(route_inputs, route_outputs):
        print(f"- '{q}' -> {r}")

    # Visual metrics
    metrics = {
        "baseline": avg_topk_score(baseline),
        "rewrite": avg_topk_score(rewritten_result),
        "hyde": avg_topk_score(hyde_result),
    }
    plot_quality_comparison(metrics, "page104_109/page104_109_quality_comparison.png")
    plot_routing_distribution(route_outputs, "page104_109/page104_109_routing_distribution.png")

    # RRF contribution chart for top fused docs
    top_fused = fused[:5]
    labels = [doc for doc, _ in top_fused]
    values = [score for _, score in top_fused]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title("Top Documents After RRF Fusion")
    plt.ylabel("RRF Score")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("page104_109/page104_109_rrf_fused_scores.png", dpi=180)
    plt.close()

    print("\nSaved visualizations:")
    print("- page104_109/page104_109_quality_comparison.png")
    print("- page104_109/page104_109_routing_distribution.png")
    print("- page104_109/page104_109_rrf_fused_scores.png")


if __name__ == "__main__":
    main()
