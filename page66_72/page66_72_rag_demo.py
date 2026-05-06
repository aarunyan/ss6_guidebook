from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

OUTPUT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = Path("/private/tmp/page66_72_mplconfig")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

# Use a non-interactive backend so images are generated in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PREFIX = OUTPUT_DIR / "page66_72"
RUN_LOG = OUTPUT_DIR / "page66_72_run_output.txt"

# Tiny in-memory corpus for the RAG demo. In a real system this would be files, wiki pages, or DB rows.
DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "City EV Policy 2025",
        "text": (
            "Bangkok launched an electric vehicle incentive in 2025. "
            "The program gives tax discounts for low-emission vehicles and expands charging stations."
        ),
        "source": "internal_policy",
    },
    {
        "id": "doc_2",
        "title": "Fleet Maintenance Notes",
        "text": (
            "Our company fleet includes electric cars, delivery vans, and motorcycles. "
            "Battery health checks are scheduled every month."
        ),
        "source": "ops_notes",
    },
    {
        "id": "doc_3",
        "title": "Public Transport Update",
        "text": (
            "The metro extension project started in 2024 and focuses on train frequency. "
            "No new parking policy was announced."
        ),
        "source": "news_digest",
    },
    {
        "id": "doc_4",
        "title": "Sustainability FAQ",
        "text": (
            "A vehicle can reduce emissions when switched from gasoline to electric power. "
            "Charging infrastructure and battery recycling are important."
        ),
        "source": "faq_portal",
    },
]

# Small semantic map to simulate dense retrieval behavior where synonyms are close in embedding space.
SEMANTIC_DIMENSIONS = [
    "transport",
    "electric",
    "policy",
    "battery",
    "rail",
]

WORD_TO_VECTOR: Dict[str, np.ndarray] = {
    "car": np.array([1.0, 0.1, 0.0, 0.0, 0.0]),
    "cars": np.array([1.0, 0.1, 0.0, 0.0, 0.0]),
    "vehicle": np.array([1.0, 0.2, 0.0, 0.0, 0.0]),
    "vehicles": np.array([1.0, 0.2, 0.0, 0.0, 0.0]),
    "van": np.array([0.9, 0.1, 0.0, 0.0, 0.0]),
    "electric": np.array([0.1, 1.0, 0.0, 0.4, 0.0]),
    "battery": np.array([0.0, 0.8, 0.0, 1.0, 0.0]),
    "charging": np.array([0.1, 0.9, 0.0, 0.7, 0.0]),
    "policy": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    "incentive": np.array([0.0, 0.2, 1.0, 0.0, 0.0]),
    "metro": np.array([0.4, 0.0, 0.0, 0.0, 1.0]),
    "train": np.array([0.4, 0.0, 0.0, 0.0, 1.0]),
}


def log(message: str, lines: List[str]) -> None:
    """Print and store messages so we can save an execution transcript."""
    print(message)
    lines.append(message)


def tokenize(text: str) -> List[str]:
    """Basic tokenizer for this educational demo."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def chunk_fixed_size(text: str, chunk_size: int = 14, overlap: int = 4) -> List[str]:
    """Fixed-size chunking: simple but can cut sentence boundaries."""
    words = tokenize(text)
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if part:
            chunks.append(" ".join(part))
    return chunks


def chunk_by_sentence(text: str) -> List[str]:
    """Sentence chunking: preserve sentence boundaries where possible."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return sentences


def chunk_semantic_group(sentences: List[str]) -> List[str]:
    """Very small semantic chunking: merge neighboring sentences with similar topic vectors."""
    if not sentences:
        return []

    sent_vectors = [dense_vector(sentence) for sentence in sentences]
    groups: List[List[str]] = [[sentences[0]]]
    current_centroid = sent_vectors[0].copy()

    for sentence, vector in zip(sentences[1:], sent_vectors[1:]):
        similarity = cosine_similarity(current_centroid, vector)
        if similarity > 0.78:
            groups[-1].append(sentence)
            current_centroid = np.mean(
                [dense_vector(" ".join(groups[-1]))],
                axis=0,
            )
        else:
            groups.append([sentence])
            current_centroid = vector.copy()

    return [" ".join(group) for group in groups]


def build_sparse_index(chunks: List[Dict[str, str]]) -> Tuple[Dict[str, Counter], Dict[str, float], int]:
    """
    Build a tiny TF-IDF style index:
    - term frequency per chunk
    - inverse document frequency per term
    """
    tf_index: Dict[str, Counter] = {}
    doc_freq: Dict[str, int] = defaultdict(int)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        tokens = tokenize(chunk["text"])
        counts = Counter(tokens)
        tf_index[chunk_id] = counts
        for term in counts.keys():
            doc_freq[term] += 1

    total_docs = len(chunks)
    idf = {
        term: math.log((1 + total_docs) / (1 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }
    return tf_index, idf, total_docs


def sparse_score(query: str, chunk_id: str, tf_index: Dict[str, Counter], idf: Dict[str, float]) -> float:
    """Compute simple TF-IDF relevance for one chunk."""
    score = 0.0
    query_counts = Counter(tokenize(query))
    chunk_counts = tf_index[chunk_id]

    for term, qtf in query_counts.items():
        score += qtf * chunk_counts.get(term, 0) * idf.get(term, 0.0)
    return score


def dense_vector(text: str) -> np.ndarray:
    """Average word vectors to get a dense embedding for text."""
    tokens = tokenize(text)
    vectors = [WORD_TO_VECTOR[token] for token in tokens if token in WORD_TO_VECTOR]
    if not vectors:
        return np.zeros(len(SEMANTIC_DIMENSIONS), dtype=float)
    return np.mean(vectors, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for dense retrieval."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def reciprocal_rank_fusion(rank: int, k: int = 60) -> float:
    """RRF score contribution for one rank position (rank starts from 1)."""
    return 1.0 / (k + rank)


def build_chunks() -> List[Dict[str, str]]:
    """Create chunks using sentence boundaries and keep metadata (source, title)."""
    chunks: List[Dict[str, str]] = []
    for doc in DOCUMENTS:
        sentence_chunks = chunk_by_sentence(doc["text"])
        for idx, chunk_text in enumerate(sentence_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_c{idx}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "text": chunk_text,
                }
            )
    return chunks


def save_rag_pipeline_diagram(path: Path) -> None:
    """Visualize slides 67-68 pipeline: index -> retrieve -> augment -> generate."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    boxes = [
        ("1. Indexing\n(split + embed + store)", "#43c6db"),
        ("2. Retrieval\n(query -> top-k)", "#5aa3ff"),
        ("3. Augment\n(context injection)", "#ffb45c"),
        ("4. Generate\n(grounded answer)", "#f0c33c"),
    ]

    x_positions = [0.04, 0.29, 0.54, 0.79]
    for x, (label, color) in zip(x_positions, boxes):
        rect = plt.Rectangle((x, 0.18), 0.18, 0.58, facecolor="#111827", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x + 0.09, 0.47, label, ha="center", va="center", color="white", fontsize=12, weight="bold")

    for start in [0.23, 0.48, 0.73]:
        ax.annotate(
            "",
            xy=(start + 0.05, 0.47),
            xytext=(start, 0.47),
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#9ca3af"},
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sparse_dense_chart(path: Path, labels: List[str], sparse_scores: List[float], dense_scores: List[float]) -> None:
    """Compare sparse and dense retrieval score behavior from slide 69."""
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - width / 2, sparse_scores, width, label="Sparse (TF-IDF)", color="#38bdf8")
    ax.bar(x + width / 2, dense_scores, width, label="Dense (Embedding Cosine)", color="#f59e0b")
    ax.set_title("Sparse vs Dense Retrieval Scores")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hybrid_chart(path: Path, labels: List[str], hybrid_scores: List[float]) -> None:
    """Show hybrid retrieval via Reciprocal Rank Fusion from slide 70."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    bars = ax.bar(labels, hybrid_scores, color="#34d399")
    ax.set_title("Hybrid Retrieval (RRF Fusion of Sparse + Dense Ranks)")
    ax.set_ylabel("RRF Score")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.0004, f"{height:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_chunking_strategy_chart(path: Path, text: str) -> None:
    """Visualize chunking strategies from slide 71."""
    fixed_chunks = chunk_fixed_size(text)
    sentence_chunks = chunk_by_sentence(text)
    semantic_chunks = chunk_semantic_group(sentence_chunks)

    names = ["Fixed-size", "Sentence-based", "Semantic-grouped"]
    counts = [len(fixed_chunks), len(sentence_chunks), len(semantic_chunks)]
    avg_lengths = [
        np.mean([len(tokenize(chunk)) for chunk in fixed_chunks]),
        np.mean([len(tokenize(chunk)) for chunk in sentence_chunks]),
        np.mean([len(tokenize(chunk)) for chunk in semantic_chunks]),
    ]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(10.5, 5.2))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, counts, width=0.5, color="#818cf8", label="Number of chunks")
    line = ax2.plot(x, avg_lengths, marker="o", color="#f97316", linewidth=2.2, label="Avg tokens per chunk")

    ax1.set_title("Chunking Strategy Trade-off")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Chunk count")
    ax2.set_ylabel("Average tokens per chunk")
    ax1.grid(axis="y", alpha=0.2)

    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{int(bar.get_height())}", ha="center", va="bottom")

    fig.legend([bars, line[0]], ["Number of chunks", "Avg tokens per chunk"], loc="upper right", bbox_to_anchor=(0.95, 0.9))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_grounded_answer(query: str, selected_chunks: List[Dict[str, str]]) -> str:
    """Simple answer generator that only uses retrieved context (simulating grounded generation)."""
    context_lines = [f"- {chunk['text']} (source={chunk['source']})" for chunk in selected_chunks]
    answer = (
        f"Question: {query}\n\n"
        "Grounded answer draft:\n"
        "The 2025 policy introduces EV incentives and tax discounts, with charging expansion.\n"
        "Battery health checks are also part of operational best practice.\n\n"
        "Cited context:\n"
        + "\n".join(context_lines)
    )
    return answer


def main() -> None:
    log_lines: List[str] = []
    log("=== page66_72 RAG demo started ===", log_lines)

    # 1) Indexing (offline): create chunks and build both sparse and dense indexes.
    chunks = build_chunks()
    tf_index, idf, _ = build_sparse_index(chunks)
    dense_index = {chunk["chunk_id"]: dense_vector(chunk["text"]) for chunk in chunks}
    log(f"Indexed chunks: {len(chunks)}", log_lines)

    # 2) Retrieval: run a user query through sparse and dense retrieval.
    query = "What electric vehicle policy was launched in 2025?"
    query_dense = dense_vector(query)
    log(f"Query: {query}", log_lines)
    log("", log_lines)

    sparse_scores_by_chunk = {
        chunk["chunk_id"]: sparse_score(query, chunk["chunk_id"], tf_index, idf)
        for chunk in chunks
    }
    dense_scores_by_chunk = {
        chunk["chunk_id"]: cosine_similarity(query_dense, dense_index[chunk["chunk_id"]])
        for chunk in chunks
    }

    sparse_ranked = sorted(chunks, key=lambda c: sparse_scores_by_chunk[c["chunk_id"]], reverse=True)
    dense_ranked = sorted(chunks, key=lambda c: dense_scores_by_chunk[c["chunk_id"]], reverse=True)

    log("Top-3 Sparse retrieval:", log_lines)
    for rank, chunk in enumerate(sparse_ranked[:3], start=1):
        cid = chunk["chunk_id"]
        log(f"  {rank}. {cid} score={sparse_scores_by_chunk[cid]:.3f} text='{chunk['text']}'", log_lines)

    log("Top-3 Dense retrieval:", log_lines)
    for rank, chunk in enumerate(dense_ranked[:3], start=1):
        cid = chunk["chunk_id"]
        log(f"  {rank}. {cid} score={dense_scores_by_chunk[cid]:.3f} text='{chunk['text']}'", log_lines)

    # 3) Hybrid retrieval (RRF): combine sparse and dense ranks.
    sparse_position = {chunk["chunk_id"]: idx + 1 for idx, chunk in enumerate(sparse_ranked)}
    dense_position = {chunk["chunk_id"]: idx + 1 for idx, chunk in enumerate(dense_ranked)}

    hybrid_scores = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        hybrid_scores[cid] = reciprocal_rank_fusion(sparse_position[cid]) + reciprocal_rank_fusion(dense_position[cid])

    hybrid_ranked = sorted(chunks, key=lambda c: hybrid_scores[c["chunk_id"]], reverse=True)
    log("Top-3 Hybrid (RRF) retrieval:", log_lines)
    for rank, chunk in enumerate(hybrid_ranked[:3], start=1):
        cid = chunk["chunk_id"]
        log(f"  {rank}. {cid} rrf={hybrid_scores[cid]:.4f} title='{chunk['title']}'", log_lines)

    # 4) Augment + Generate: build a context bundle and produce a grounded answer.
    top_k = 2
    selected_chunks = hybrid_ranked[:top_k]
    answer = build_grounded_answer(query, selected_chunks)
    log("", log_lines)
    log("Generated grounded answer preview:", log_lines)
    log(answer, log_lines)

    # 5) Visualizations aligned with slide messages.
    save_rag_pipeline_diagram(OUTPUT_PREFIX.with_name("page66_72_rag_pipeline.png"))

    labels = [chunk["chunk_id"] for chunk in chunks]
    sparse_plot_scores = [sparse_scores_by_chunk[label] for label in labels]
    dense_plot_scores = [dense_scores_by_chunk[label] for label in labels]
    hybrid_plot_scores = [hybrid_scores[label] for label in labels]

    save_sparse_dense_chart(
        OUTPUT_PREFIX.with_name("page66_72_sparse_vs_dense.png"),
        labels,
        sparse_plot_scores,
        dense_plot_scores,
    )
    save_hybrid_chart(
        OUTPUT_PREFIX.with_name("page66_72_hybrid_rrf.png"),
        labels,
        hybrid_plot_scores,
    )
    save_chunking_strategy_chart(
        OUTPUT_PREFIX.with_name("page66_72_chunking_strategies.png"),
        " ".join(doc["text"] for doc in DOCUMENTS),
    )

    RUN_LOG.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nSaved run log: {RUN_LOG.name}", log_lines)
    log("=== page66_72 RAG demo finished ===", log_lines)


if __name__ == "__main__":
    main()
