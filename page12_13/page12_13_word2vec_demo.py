from __future__ import annotations

import math
import os
import random
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "page12_13_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / "page12_13_mplconfig"))

import matplotlib

# Keep matplotlib caches inside the project so the script works in restricted environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA


SEED = 7
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_PREFIX = SCRIPT_DIR / "page12_13"


TRAINING_SENTENCES = [
    "king rules a kingdom and queen rules a kingdom",
    "man is a king and woman is a queen",
    "queen is a woman ruler and king is a man ruler",
    "a king is to man as a queen is to woman",
    "prince is a young king and princess is a young queen",
    "paris is the capital of france",
    "berlin is the capital of germany",
    "madrid is the capital of spain",
    "bangkok is the capital of thailand",
    "france has paris as its capital",
    "germany has berlin as its capital",
    "spain has madrid as its capital",
    "thailand has bangkok as its capital",
    "a bank can approve a loan",
    "the bank manages money and credit",
    "people visit the bank to deposit cash",
    "the river bank was muddy after rain",
    "trees grow near the river bank",
    "kids sat on the grassy bank by the river",
    "queen and princess share a royal title",
    "king and prince share a royal title",
]

STOPWORDS = {
    "a", "and", "the", "is", "of", "to", "its", "as", "by", "can", "was", "after", "has",
    "near", "on",
}


def tokenize(sentence: str) -> list[str]:
    """Lowercase and split text into tokens."""
    return sentence.lower().split()


def build_vocabulary(sentences: list[str]) -> tuple[list[str], dict[str, int], list[list[int]]]:
    """Create the word list, index lookup, and tokenized corpus."""
    tokenized = [tokenize(sentence) for sentence in sentences]
    vocab = sorted({token for sentence in tokenized for token in sentence})
    word_to_index = {word: i for i, word in enumerate(vocab)}
    corpus = [[word_to_index[token] for token in sentence] for sentence in tokenized]
    return vocab, word_to_index, corpus


def generate_skipgram_pairs(corpus: list[list[int]], window_size: int) -> list[tuple[int, int]]:
    """Collect (center_word, context_word) training pairs."""
    pairs: list[tuple[int, int]] = []
    for sentence in corpus:
        for center_position, center_index in enumerate(sentence):
            left = max(0, center_position - window_size)
            right = min(len(sentence), center_position + window_size + 1)
            for context_position in range(left, right):
                if context_position == center_position:
                    continue
                pairs.append((center_index, sentence[context_position]))
    return pairs


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


class TinyWord2Vec:
    """Minimal skip-gram word2vec trained with full softmax."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_vectors = np.random.uniform(
            low=-0.5 / embedding_dim,
            high=0.5 / embedding_dim,
            size=(vocab_size, embedding_dim),
        )
        self.output_vectors = np.zeros((vocab_size, embedding_dim))

    def train_one_pair(self, center_index: int, context_index: int, learning_rate: float) -> float:
        """One skip-gram update: predict context from a center word."""
        hidden = self.input_vectors[center_index]
        logits = self.output_vectors @ hidden
        probabilities = softmax(logits)

        target = np.zeros(self.vocab_size)
        target[context_index] = 1.0

        error = probabilities - target
        grad_output = np.outer(error, hidden)
        grad_input = self.output_vectors.T @ error

        self.output_vectors -= learning_rate * grad_output
        self.input_vectors[center_index] -= learning_rate * grad_input

        return -math.log(max(probabilities[context_index], 1e-12))

    def embeddings(self) -> np.ndarray:
        """Average input and output embeddings to get the final word vector."""
        return (self.input_vectors + self.output_vectors) / 2.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def print_top_contexts(
    vocab: list[str],
    pairs: list[tuple[int, int]],
    watched_words: list[str],
) -> None:
    """Show which context words each watched word appears beside."""
    index_to_word = {i: word for i, word in enumerate(vocab)}
    grouped: dict[str, Counter[str]] = {word: Counter() for word in watched_words}

    for center_index, context_index in pairs:
        center_word = index_to_word[center_index]
        if center_word in grouped:
            grouped[center_word][index_to_word[context_index]] += 1

    print("\nContext snapshots from the corpus")
    print("-" * 60)
    for word in watched_words:
        top_items = grouped[word].most_common(5)
        summary = ", ".join(f"{context}:{count}" for context, count in top_items)
        print(f"{word:>8} -> {summary}")


def plot_training_curve(loss_history: list[float], output_file: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(loss_history, color="#118ab2", linewidth=2)
    plt.title("Word2Vec Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average negative log-likelihood")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def build_projection_frames(
    checkpoints: list[tuple[int, np.ndarray]],
    vocab: list[str],
    focus_words: list[str],
    output_gif: Path,
) -> None:
    """Animate how selected words move through vector space during training."""
    focus_indices = [vocab.index(word) for word in focus_words]
    projected_frames: list[tuple[int, np.ndarray]] = []
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")

    for epoch, embedding_matrix in checkpoints:
        projection = PCA(n_components=2).fit_transform(embedding_matrix[focus_indices])
        projected_frames.append((epoch, projection))
        x_min = min(x_min, float(np.min(projection[:, 0])))
        x_max = max(x_max, float(np.max(projection[:, 0])))
        y_min = min(y_min, float(np.min(projection[:, 1])))
        y_max = max(y_max, float(np.max(projection[:, 1])))

    padding = 0.8
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    def update(frame_index: int) -> None:
        ax.clear()
        epoch, projection = projected_frames[frame_index]
        colors = [
            "#ef476f" if word in {"king", "queen", "prince", "princess", "man", "woman"} else
            "#118ab2" if word in {"france", "paris", "germany", "berlin", "thailand", "bangkok"} else
            "#06d6a0"
            for word in focus_words
        ]
        ax.scatter(projection[:, 0], projection[:, 1], c=colors, s=90)
        for word, (x, y) in zip(focus_words, projection):
            ax.text(x + 0.03, y + 0.03, word, fontsize=10)
        ax.set_title(f"How word vectors move during training (epoch {epoch})")
        ax.set_xlabel("PCA axis 1")
        ax.set_ylabel("PCA axis 2")
        ax.grid(alpha=0.25)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    animation = FuncAnimation(fig, update, frames=len(projected_frames), interval=900, repeat=True)
    animation.save(output_gif, writer=PillowWriter(fps=1))
    plt.close(fig)


def plot_final_embeddings(embeddings: np.ndarray, vocab: list[str], output_file: Path) -> None:
    focus_words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "france", "paris", "germany", "berlin", "thailand", "bangkok",
        "bank", "river", "money", "loan",
    ]
    indices = [vocab.index(word) for word in focus_words]
    projection = PCA(n_components=2).fit_transform(embeddings[indices])

    plt.figure(figsize=(8.5, 6.5))
    colors = [
        "#ef476f" if word in {"king", "queen", "prince", "princess", "man", "woman"} else
        "#118ab2" if word in {"france", "paris", "germany", "berlin", "thailand", "bangkok"} else
        "#06d6a0"
        for word in focus_words
    ]
    plt.scatter(projection[:, 0], projection[:, 1], c=colors, s=90)
    for word, (x, y) in zip(focus_words, projection):
        plt.text(x + 0.03, y + 0.03, word, fontsize=10)
    plt.title("Final word2vec-style embedding map")
    plt.xlabel("PCA axis 1")
    plt.ylabel("PCA axis 2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def plot_similarity_bars(
    embeddings: np.ndarray,
    vocab: list[str],
    anchor_word: str,
    candidate_words: list[str],
    output_file: Path,
) -> None:
    anchor_vector = embeddings[vocab.index(anchor_word)]
    scores = [cosine_similarity(anchor_vector, embeddings[vocab.index(word)]) for word in candidate_words]

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(candidate_words, scores, color="#06d6a0")
    plt.ylim(0, 1)
    plt.title(f"Cosine similarity to '{anchor_word}'")
    plt.ylabel("Similarity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def analogy(
    embeddings: np.ndarray,
    vocab: list[str],
    positive_words: list[str],
    negative_words: list[str],
    top_k: int = 5,
    banned_words: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Return nearest words for vector arithmetic like king - man + woman."""
    word_to_index = {word: i for i, word in enumerate(vocab)}
    query = np.zeros(embeddings.shape[1])
    excluded = set(positive_words + negative_words)
    if banned_words:
        excluded |= banned_words

    for word in positive_words:
        query += embeddings[word_to_index[word]]
    for word in negative_words:
        query -= embeddings[word_to_index[word]]

    scores = []
    for word in vocab:
        if word in excluded:
            continue
        score = cosine_similarity(query, embeddings[word_to_index[word]])
        scores.append((word, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


def describe_polysemy(
    embeddings: np.ndarray,
    vocab: list[str],
    output_file: Path,
) -> None:
    """Visualize the static-embedding limitation for the word 'bank'."""
    bank_vector = embeddings[vocab.index("bank")]
    comparison_words = ["money", "loan", "cash", "river", "muddy", "trees"]
    scores = [cosine_similarity(bank_vector, embeddings[vocab.index(word)]) for word in comparison_words]

    plt.figure(figsize=(8.5, 4.8))
    bars = plt.bar(comparison_words, scores, color=["#118ab2"] * 3 + ["#ef476f"] * 3)
    plt.ylim(0, 1)
    plt.title("One static vector for 'bank' mixes two meanings")
    plt.ylabel("Similarity to 'bank'")
    plt.grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def main() -> None:
    print("page12_13_word2vec_demo")
    print("=" * 60)
    print("Goal: show how word2vec learns similar meanings from similar contexts.")
    print("Bonus: connect the result to vector arithmetic and the static 'bank' limitation.\n")

    vocab, word_to_index, corpus = build_vocabulary(TRAINING_SENTENCES)
    pairs = generate_skipgram_pairs(corpus, window_size=2)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training sentences: {len(TRAINING_SENTENCES)}")
    print(f"Skip-gram pairs: {len(pairs)}")

    print_top_contexts(vocab, pairs, watched_words=["king", "queen", "bank", "paris", "bangkok"])

    model = TinyWord2Vec(vocab_size=len(vocab), embedding_dim=12)
    learning_rate = 0.06
    epochs = 220

    loss_history: list[float] = []
    checkpoints: list[tuple[int, np.ndarray]] = []
    focus_words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "france", "paris", "germany", "berlin", "thailand", "bangkok",
        "bank", "river", "money", "loan",
    ]

    print("\nTraining progress")
    print("-" * 60)
    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        for center_index, context_index in pairs:
            total_loss += model.train_one_pair(center_index, context_index, learning_rate)
        average_loss = total_loss / len(pairs)
        loss_history.append(average_loss)

        if epoch in {1, 20, 60, 120, 180, 220}:
            checkpoints.append((epoch, model.embeddings().copy()))

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3} | average loss = {average_loss:.4f}")

    final_embeddings = model.embeddings()

    print("\nNearest words to 'king'")
    print("-" * 60)
    king_vector = final_embeddings[word_to_index["king"]]
    similarities = []
    for candidate in vocab:
        if candidate == "king" or candidate in STOPWORDS:
            continue
        score = cosine_similarity(king_vector, final_embeddings[word_to_index[candidate]])
        similarities.append((candidate, score))
    for word, score in sorted(similarities, key=lambda item: item[1], reverse=True)[:6]:
        print(f"{word:>10} : {score:.3f}")

    print("\nAnalogy experiments")
    print("-" * 60)
    for label, positive_words, negative_words in [
        ("king - man + woman", ["king", "woman"], ["man"]),
        ("paris - france + thailand", ["paris", "thailand"], ["france"]),
    ]:
        answer = analogy(
            final_embeddings,
            vocab,
            positive_words=positive_words,
            negative_words=negative_words,
            top_k=5,
            banned_words=STOPWORDS,
        )
        best_words = ", ".join(f"{word} ({score:.3f})" for word, score in answer)
        print(f"{label:<28} -> {best_words}")

    print("\nWhy page 13 says 'bank' is a problem")
    print("-" * 60)
    bank_vector = final_embeddings[word_to_index["bank"]]
    for comparison_word in ["money", "loan", "river", "muddy"]:
        score = cosine_similarity(bank_vector, final_embeddings[word_to_index[comparison_word]])
        print(f"similarity(bank, {comparison_word:>5}) = {score:.3f}")
    print("Observation: one static vector tries to cover both the finance and river meanings.")

    plot_training_curve(loss_history, OUTPUT_PREFIX.with_name("page12_13_training_loss.png"))
    build_projection_frames(
        checkpoints,
        vocab,
        focus_words,
        OUTPUT_PREFIX.with_name("page12_13_embedding_journey.gif"),
    )
    plot_final_embeddings(final_embeddings, vocab, OUTPUT_PREFIX.with_name("page12_13_embedding_map.png"))
    plot_similarity_bars(
        final_embeddings,
        vocab,
        anchor_word="king",
        candidate_words=["queen", "prince", "woman", "paris", "bank", "river"],
        output_file=OUTPUT_PREFIX.with_name("page12_13_king_similarity.png"),
    )
    describe_polysemy(final_embeddings, vocab, OUTPUT_PREFIX.with_name("page12_13_bank_problem.png"))

    print("\nSaved files")
    print("-" * 60)
    for filename in [
        "page12_13_training_loss.png",
        "page12_13_embedding_journey.gif",
        "page12_13_embedding_map.png",
        "page12_13_king_similarity.png",
        "page12_13_bank_problem.png",
    ]:
        print(filename)


if __name__ == "__main__":
    main()
