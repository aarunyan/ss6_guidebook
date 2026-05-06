from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

OUTPUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "page11_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "page11_cache"))

import matplotlib

# Use a writable config directory and a non-interactive backend for headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RUN_LOG = OUTPUT_DIR / "page11_run_output.txt"


def log(message: str, lines: List[str]) -> None:
    """Print and remember the message so we can save the run transcript."""
    print(message)
    lines.append(message)


def tokenize_sentence(sentence: str) -> List[str]:
    """Lowercase and split a sentence into word tokens."""
    return sentence.lower().split()


def build_ngram_counts(
    sentences: Sequence[str], n: int = 2
) -> Tuple[Counter[Tuple[str, ...]], Counter[Tuple[str, ...]]]:
    """
    Build context counts and full n-gram counts.

    For a bigram model (n=2):
    - context is the previous 1 word
    - n-gram is (previous_word, current_word)
    """
    context_counts: Counter[Tuple[str, ...]] = Counter()
    ngram_counts: Counter[Tuple[str, ...]] = Counter()

    for sentence in sentences:
        tokens = ["<s>"] + tokenize_sentence(sentence) + ["</s>"]
        for index in range(n - 1, len(tokens)):
            context = tuple(tokens[index - (n - 1) : index])
            gram = tuple(tokens[index - (n - 1) : index + 1])
            context_counts[context] += 1
            ngram_counts[gram] += 1

    return context_counts, ngram_counts


def conditional_probability(
    context: Tuple[str, ...],
    next_word: str,
    context_counts: Counter[Tuple[str, ...]],
    ngram_counts: Counter[Tuple[str, ...]],
) -> float:
    """Compute P(next_word | context) from raw frequency counts."""
    full_ngram = context + (next_word,)
    numerator = ngram_counts[full_ngram]
    denominator = context_counts[context]
    if denominator == 0:
        return 0.0
    return numerator / denominator


def predict_next_words(
    context_word: str,
    vocabulary: Iterable[str],
    context_counts: Counter[Tuple[str, ...]],
    ngram_counts: Counter[Tuple[str, ...]],
) -> List[Tuple[str, float, int]]:
    """Rank candidate next words for a single-word context."""
    context = (context_word,)
    results: List[Tuple[str, float, int]] = []

    for word in sorted(vocabulary):
        count = ngram_counts[(context_word, word)]
        if count == 0:
            continue
        probability = conditional_probability(context, word, context_counts, ngram_counts)
        results.append((word, probability, count))

    return sorted(results, key=lambda item: (-item[1], -item[2], item[0]))


def make_probability_table(
    contexts: Sequence[str],
    vocabulary: Sequence[str],
    context_counts: Counter[Tuple[str, ...]],
    ngram_counts: Counter[Tuple[str, ...]],
) -> pd.DataFrame:
    """Create a matrix where rows are context words and columns are next words."""
    data: Dict[str, List[float]] = defaultdict(list)

    for word in vocabulary:
        for context_word in contexts:
            probability = conditional_probability(
                (context_word,), word, context_counts, ngram_counts
            )
            data[context_word].append(probability)

    return pd.DataFrame(data, index=vocabulary).T


def save_heatmap(probability_table: pd.DataFrame, destination: Path) -> None:
    """Save a heatmap-style image of bigram probabilities."""
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(probability_table.values, cmap="YlOrRd", aspect="auto")

    ax.set_title("Bigram probabilities: P(next word | previous word)")
    ax.set_xlabel("Next word")
    ax.set_ylabel("Previous word")
    ax.set_xticks(range(len(probability_table.columns)))
    ax.set_xticklabels(probability_table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(probability_table.index)))
    ax.set_yticklabels(probability_table.index)

    for row_index, context_word in enumerate(probability_table.index):
        for column_index, next_word in enumerate(probability_table.columns):
            value = probability_table.loc[context_word, next_word]
            label = f"{value:.2f}" if value > 0 else "0"
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def save_prediction_bar_chart(
    predictions: Sequence[Tuple[str, float, int]], context_word: str, destination: Path
) -> None:
    """Save a bar chart for the predicted next-word distribution."""
    words = [word for word, _, _ in predictions]
    probabilities = [probability for _, probability, _ in predictions]
    counts = [count for _, _, count in predictions]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(words, probabilities, color="#ff9f43")
    ax.set_title(f'Next-word prediction after "{context_word}"')
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(probabilities) + 0.15)

    for bar, count, probability in zip(bars, counts, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"count={count}\nP={probability:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def save_limitations_chart(
    context_counts: Counter[Tuple[str, ...]],
    ngram_counts: Counter[Tuple[str, ...]],
    destination: Path,
) -> None:
    """Visualize how sparse a bigram table becomes."""
    observed_contexts = sorted(context[0] for context in context_counts if context[0] != "<s>")
    observed_next_words = sorted(
        word for *_, word in ngram_counts if word not in {"</s>"}
    )

    possible_pairs = len(observed_contexts) * len(observed_next_words)
    seen_pairs = sum(
        1
        for context in observed_contexts
        for word in observed_next_words
        if ngram_counts[(context, word)] > 0
    )
    unseen_pairs = possible_pairs - seen_pairs

    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = [seen_pairs, unseen_pairs]
    labels = ["Seen pairs", "Unseen pairs"]
    colors = ["#2ecc71", "#34495e"]
    bars = ax.bar(labels, values, color=colors)

    ax.set_title("Vocabulary explosion: most word pairs are never seen")
    ax.set_ylabel("Number of bigram pairs")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def save_runtime_ascii_art(
    predictions: Sequence[Tuple[str, float, int]], context_word: str, lines: List[str]
) -> None:
    """Print a small text-based visualization so the model is visible while the code runs."""
    log("", lines)
    log(f'Runtime view for context "{context_word}":', lines)
    for word, probability, count in predictions:
        bar = "#" * int(probability * 20)
        log(f"  {context_word:>4} -> {word:<8} {bar:<20} P={probability:.2f} count={count}", lines)


def main() -> None:
    lines: List[str] = []

    # A tiny training corpus chosen to make the counting logic easy to inspect.
    training_sentences = [
        "I eat rice",
        "I eat noodles",
        "I eat rice",
        "You eat rice",
        "We eat rice",
        "They eat noodles",
        "Cats chase mice",
        "Dogs chase cats",
    ]

    log("page11 n-gram demo", lines)
    log("===================", lines)
    log("Training sentences:", lines)
    for sentence in training_sentences:
        log(f"  - {sentence}", lines)

    context_counts, ngram_counts = build_ngram_counts(training_sentences, n=2)
    vocabulary = sorted(
        {
            token
            for sentence in training_sentences
            for token in tokenize_sentence(sentence)
        }
    )

    log("", lines)
    log("Key bigram counts:", lines)
    interesting_pairs = [("i", "eat"), ("eat", "rice"), ("eat", "noodles"), ("chase", "cats")]
    for first, second in interesting_pairs:
        log(f'  Count("{first} {second}") = {ngram_counts[(first, second)]}', lines)

    log("", lines)
    log('Bigram formula: P(next | previous) = Count(previous, next) / Count(previous)', lines)
    eat_count = context_counts[("eat",)]
    log(f'  Count("eat") = {eat_count}', lines)
    log(f'  P("rice" | "eat") = {ngram_counts[("eat", "rice")]}/{eat_count} = '
        f'{conditional_probability(("eat",), "rice", context_counts, ngram_counts):.2f}', lines)
    log(f'  P("noodles" | "eat") = {ngram_counts[("eat", "noodles")]}/{eat_count} = '
        f'{conditional_probability(("eat",), "noodles", context_counts, ngram_counts):.2f}', lines)

    predictions_after_eat = predict_next_words("eat", vocabulary + ["</s>"], context_counts, ngram_counts)
    save_runtime_ascii_art(predictions_after_eat, "eat", lines)

    contexts_for_chart = ["i", "you", "we", "they", "eat", "chase"]
    next_words_for_chart = ["eat", "rice", "noodles", "chase", "cats", "mice", "</s>"]
    probability_table = make_probability_table(
        contexts_for_chart, next_words_for_chart, context_counts, ngram_counts
    )

    heatmap_path = OUTPUT_DIR / "page11_bigram_heatmap.png"
    bar_chart_path = OUTPUT_DIR / "page11_prediction_after_eat.png"
    limits_chart_path = OUTPUT_DIR / "page11_vocabulary_explosion.png"

    save_heatmap(probability_table, heatmap_path)
    save_prediction_bar_chart(predictions_after_eat, "eat", bar_chart_path)
    save_limitations_chart(context_counts, ngram_counts, limits_chart_path)

    log("", lines)
    log("What the demo shows:", lines)
    log("  1. Short context: the model only looks one word back in a bigram model.", lines)
    log("  2. Vocabulary explosion: many possible word pairs are never observed.", lines)
    log('  3. No meaning: "eat" is predicted from counts, not from understanding.', lines)

    RUN_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
