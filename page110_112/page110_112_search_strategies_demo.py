"""
page110_112_search_strategies_demo.py

A small, self-contained simulation of four search strategies from slides 110-112:
- Semantic Search
- Keyword / BM25-style Search
- Hybrid + RRF
- Agentic Search (iterative retrieve -> observe -> retrieve)

The script generates two visualizations and prints a compact summary table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Keep every output file prefixed as requested.
OUTPUT_DIR = Path(__file__).resolve().parent
PREFIX = "page110_112_"


@dataclass
class Document:
    doc_id: str
    text: str
    dense_vec: np.ndarray


@dataclass
class QueryCase:
    query: str
    target_doc_id: str


def build_corpus() -> List[Document]:
    """Create a toy corpus with both paraphrases and exact technical keywords."""
    return [
        Document("D1", "How to reset your account password", np.array([0.95, 0.05, 0.0])),
        Document("D2", "Steps to change login credentials", np.array([0.92, 0.08, 0.0])),
        Document("D3", "BM25 ranking formula and inverse document frequency", np.array([0.15, 0.85, 0.0])),
        Document("D4", "GPU optimization for transformer inference", np.array([0.05, 0.95, 0.0])),
        Document("D5", "Graph traversal with Breadth-First Search", np.array([0.1, 0.2, 0.95])),
        Document("D6", "Reason-in-documents iterative retrieval notes", np.array([0.25, 0.35, 0.8])),
    ]


def build_queries() -> List[QueryCase]:
    """Query set intentionally mixes paraphrases and exact-term requests."""
    return [
        QueryCase("I forgot my sign-in secret, how can I recover access?", "D1"),
        QueryCase("Explain BM25 idf term weighting", "D3"),
        QueryCase("Guide to optimize GPU transformer serving", "D4"),
        QueryCase("What is breadth first search in graphs?", "D5"),
    ]


def tokenize(text: str) -> List[str]:
    return [tok.strip(".,!?;:()[]\"'").lower() for tok in text.split() if tok.strip()]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def query_embedding(query: str) -> np.ndarray:
    """
    Very small handcrafted embedding logic for demonstration.

    Vector dimensions:
    [account_access_meaning, technical_term_focus, graph_reasoning]
    """
    q = query.lower()
    vec = np.array([0.2, 0.2, 0.2], dtype=float)

    if any(k in q for k in ["password", "recover", "access", "login", "sign-in", "credentials", "secret"]):
        vec += np.array([0.7, 0.05, 0.0])
    if any(k in q for k in ["bm25", "idf", "gpu", "transformer", "serving", "optimize"]):
        vec += np.array([0.05, 0.75, 0.0])
    if any(k in q for k in ["graph", "breadth", "first", "search", "bfs"]):
        vec += np.array([0.0, 0.05, 0.75])

    return vec


def semantic_scores(query: str, corpus: Sequence[Document]) -> Dict[str, float]:
    q_vec = query_embedding(query)
    return {doc.doc_id: cosine_similarity(q_vec, doc.dense_vec) for doc in corpus}


def keyword_scores(query: str, corpus: Sequence[Document]) -> Dict[str, float]:
    """
    Lightweight BM25-like score for readability.

    This is not full BM25; it captures the spirit:
    - count exact token overlap
    - weight rarer query terms more (IDF-like)
    """
    q_tokens = tokenize(query)
    doc_tokens = {doc.doc_id: tokenize(doc.text) for doc in corpus}

    df: Dict[str, int] = {}
    n_docs = len(corpus)
    for token in set(q_tokens):
        df[token] = sum(1 for d in corpus if token in doc_tokens[d.doc_id])

    scores: Dict[str, float] = {}
    for doc in corpus:
        score = 0.0
        tokens = doc_tokens[doc.doc_id]
        for token in q_tokens:
            tf = tokens.count(token)
            if tf == 0:
                continue
            idf = math.log((n_docs + 1) / (df.get(token, 0) + 1)) + 1.0
            score += tf * idf
        scores[doc.doc_id] = score
    return scores


def rank_desc(scores: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def rrf_fuse(rankings: Sequence[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion for Hybrid retrieval."""
    all_doc_ids = {doc_id for ranking in rankings for doc_id in ranking}
    fused = {doc_id: 0.0 for doc_id in all_doc_ids}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + pos)
    return fused


def agentic_search(query: str, corpus: Sequence[Document]) -> Tuple[List[str], int]:
    """
    Simulate iterative behavior:
    1) First query
    2) Observe top hit, expand query with discovered terms
    3) Search again
    """
    sem = semantic_scores(query, corpus)
    key = keyword_scores(query, corpus)

    first_pass = rank_desc(rrf_fuse([rank_desc(sem), rank_desc(key)]))
    top_doc = next((d for d in corpus if d.doc_id == first_pass[0]), None)

    expanded_query = query
    if top_doc is not None:
        # Agent "observes" a document and adds high-signal terms.
        seed_terms = tokenize(top_doc.text)[:3]
        expanded_query = f"{query} {' '.join(seed_terms)}"

    sem2 = semantic_scores(expanded_query, corpus)
    key2 = keyword_scores(expanded_query, corpus)
    second_pass = rank_desc(rrf_fuse([rank_desc(sem2), rank_desc(key2)]))
    return second_pass, 2


def evaluate_top1_accuracy(corpus: Sequence[Document], queries: Sequence[QueryCase]) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    strategies = ["Semantic", "Keyword", "Hybrid+RRF", "Agentic"]
    correct_counts = {s: 0 for s in strategies}
    details: List[Dict[str, object]] = []

    for case in queries:
        sem_scores = semantic_scores(case.query, corpus)
        key_scores = keyword_scores(case.query, corpus)

        sem_rank = rank_desc(sem_scores)
        key_rank = rank_desc(key_scores)
        hyb_rank = rank_desc(rrf_fuse([sem_rank, key_rank]))
        ag_rank, hops = agentic_search(case.query, corpus)

        picked = {
            "Semantic": sem_rank[0],
            "Keyword": key_rank[0],
            "Hybrid+RRF": hyb_rank[0],
            "Agentic": ag_rank[0],
        }

        for strategy, doc_id in picked.items():
            if doc_id == case.target_doc_id:
                correct_counts[strategy] += 1

        details.append(
            {
                "query": case.query,
                "target": case.target_doc_id,
                "semantic": sem_rank[0],
                "keyword": key_rank[0],
                "hybrid": hyb_rank[0],
                "agentic": ag_rank[0],
                "agentic_hops": hops,
            }
        )

    accuracy = {s: correct_counts[s] / len(queries) for s in strategies}
    return accuracy, details


def make_accuracy_chart(accuracy: Dict[str, float]) -> Path:
    labels = list(accuracy.keys())
    values = [accuracy[k] for k in labels]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(labels, values, color=["#2dd4bf", "#fb923c", "#fbbf24", "#a78bfa"])
    plt.ylim(0, 1.05)
    plt.ylabel("Top-1 Accuracy")
    plt.title("Search Strategy Comparison (Slides 110-112 Concept Demo)")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center")

    out = OUTPUT_DIR / f"{PREFIX}strategy_accuracy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def make_flow_chart() -> Path:
    """Visualize complexity tradeoff: quality vs operational complexity."""
    points = {
        "Semantic": (0.72, 0.45),
        "Keyword": (0.58, 0.20),
        "Hybrid+RRF": (0.82, 0.62),
        "Agentic": (0.90, 0.88),
    }

    plt.figure(figsize=(8.5, 6.0))
    for name, (quality, complexity) in points.items():
        plt.scatter(quality, complexity, s=220)
        plt.text(quality + 0.01, complexity + 0.01, name)

    plt.xlabel("Answer Quality (Higher is better)")
    plt.ylabel("System Complexity (Higher is harder)")
    plt.title("Tradeoff Map: Quality vs Complexity")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.1, 1.0)
    plt.grid(alpha=0.3)

    out = OUTPUT_DIR / f"{PREFIX}quality_vs_complexity.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def write_ascii_flow() -> Path:
    diagram = r"""
+----------------------+      +----------------------+      +------------------------+
| User Question        | ---> | Retriever Layer      | ---> | Candidate Documents    |
+----------------------+      +----------------------+      +------------------------+
                                    |                                  |
                                    |                                  v
                                    |                         +------------------+
                                    |                         | Rank / Fuse      |
                                    |                         | (RRF for hybrid) |
                                    |                         +------------------+
                                    |                                  |
                                    v                                  v
                         +----------------------+         +----------------------+
                         | Agent (optional)     | <-----> | Observe + Re-query   |
                         | decides next query   |         | iterative refinement |
                         +----------------------+         +----------------------+
                                    |
                                    v
                           +-------------------+
                           | Final answer path |
                           +-------------------+
""".strip("\n")

    out = OUTPUT_DIR / f"{PREFIX}ascii_flow.txt"
    out.write_text(diagram + "\n", encoding="utf-8")
    return out


def main() -> None:
    corpus = build_corpus()
    queries = build_queries()

    accuracy, details = evaluate_top1_accuracy(corpus, queries)

    print("=== Search Strategy Demo (Slides 110-112) ===")
    print("Top-1 Accuracy Summary")
    for k, v in accuracy.items():
        print(f"- {k:10s}: {v:.2f}")

    print("\nPer-query picks")
    for row in details:
        print(
            f"- Query: {row['query']}\n"
            f"  target={row['target']} | semantic={row['semantic']} | keyword={row['keyword']} "
            f"| hybrid={row['hybrid']} | agentic={row['agentic']} (hops={row['agentic_hops']})"
        )

    acc_path = make_accuracy_chart(accuracy)
    tradeoff_path = make_flow_chart()
    ascii_path = write_ascii_flow()

    print("\nSaved artifacts:")
    print(f"- {acc_path.name}")
    print(f"- {tradeoff_path.name}")
    print(f"- {ascii_path.name}")


if __name__ == "__main__":
    main()
