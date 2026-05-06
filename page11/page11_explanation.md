# page11 n-gram language model

## Start with an analogy

Think of an n-gram language model like a cashier who finishes your sentence based only on the last few words they hear all day.

If customers often say:

- "I eat rice"
- "I eat noodles"
- "We eat rice"

then the cashier learns a habit: after hearing `"eat"`, the next word is usually `"rice"`.

That is basically what the code does. It does not understand food, grammar, or meaning. It just keeps score and says, "What usually comes next after this context?"

Another analogy: it is like predictive text with a very short memory. A bigram model remembers only **one previous word**. A trigram model remembers **two previous words**. The larger `n` gets, the longer the memory, but the rarer the exact matches become.

## Draw a diagram

```text
Training sentences
    |
    v
Split into word pairs (bigrams)

"I eat rice"      -> ("i","eat"), ("eat","rice")
"I eat noodles"   -> ("i","eat"), ("eat","noodles")
"We eat rice"     -> ("we","eat"), ("eat","rice")

    |
    v
Count how often each pair appears

("eat","rice")    = 4
("eat","noodles") = 2

    |
    v
Convert counts to probabilities

P("rice" | "eat")    = Count("eat rice") / Count("eat")
P("noodles" | "eat") = Count("eat noodles") / Count("eat")

    |
    v
Pick the most likely next word

Context: "eat"  ->  predict: "rice"
```

## Walk through the code

### 1. We prepare a tiny training corpus

The script stores a short list of sentences such as `"I eat rice"` and `"They eat noodles"`.

Why keep it small? Because with n-grams, the whole idea is counting patterns, and a tiny dataset makes the counting easy to see.

### 2. We tokenize each sentence

`tokenize_sentence()` lowercases the sentence and splits it into words.

So:

```text
"I eat rice" -> ["i", "eat", "rice"]
```

The code also adds special boundary tokens:

```text
["<s>", "i", "eat", "rice", "</s>"]
```

`<s>` means start of sentence. `</s>` means end of sentence.

### 3. We build bigram counts

`build_ngram_counts(..., n=2)` is the core of the demo.

For each sentence, the code slides a window across the tokens:

```text
("<s>", "i")
("i", "eat")
("eat", "rice")
("rice", "</s>")
```

It keeps two count tables:

- `context_counts`: how often a context appears
- `ngram_counts`: how often a full n-gram appears

For a bigram model:

- context = previous 1 word
- n-gram = previous word + current word

So if the model sees `"eat"` six times total:

```text
Count("eat") = 6
```

and `"eat rice"` happens four times:

```text
Count("eat rice") = 4
```

then:

```text
P("rice" | "eat") = 4 / 6 = 0.67
```

### 4. We compute conditional probability

`conditional_probability()` uses the slide’s formula directly:

```text
P(word | previous n-1 words)
```

In the bigram case, that becomes:

```text
P(next_word | previous_word)
```

The function does:

```text
Count(previous_word, next_word) / Count(previous_word)
```

No neural network. No embeddings. Just counting and division.

### 5. We rank possible next words

`predict_next_words("eat", ...)` checks which words have actually followed `"eat"` in the training data.

If the data says:

- `"eat rice"` appears 4 times
- `"eat noodles"` appears 2 times

then the model ranks `"rice"` above `"noodles"`.

That is why n-grams feel intuitive: they mimic "what usually came next before?"

### 6. We visualize the result in two ways

The script gives you both runtime and saved visuals:

- `page11_run_output.txt`: a text-based live view with ASCII bars
- `page11_bigram_heatmap.png`: a heatmap of `P(next word | previous word)`
- `page11_prediction_after_eat.png`: a bar chart for predictions after `"eat"`
- `page11_vocabulary_explosion.png`: a chart showing how many word pairs were never seen

The runtime ASCII section is useful because it lets you watch the probability distribution while the script runs, like this:

```text
eat -> rice      #############       P=0.67
eat -> noodles   ######              P=0.33
```

### 7. We connect back to the slide’s three problems

The script intentionally illustrates the same limitations shown on page 11:

#### Problem 1: Short context

A bigram only looks back **one word**.

That means it cannot use longer meaningfully connected context very well.

Analogy: it is like trying to guess the rest of a joke after hearing only the last word.

#### Problem 2: Vocabulary explosion

As vocabulary grows, the number of possible word combinations grows very fast.

Most combinations are never seen, which makes the count table sparse.

Analogy: imagine a restaurant menu where you try to memorize every possible two-dish combo. Most combos will never be ordered, so your notebook is mostly empty pages.

#### Problem 3: No meaning

The model has no semantic understanding.

It does not know whether `"cat"` and `"dog"` are similar animals. If one appears often after a word and the other does not, that is all it cares about.

It counts patterns. It does not understand concepts.

## Highlight a gotcha

A common misconception is: "If I increase `n`, the model automatically gets smarter."

Not exactly.

Increasing `n` gives the model a longer memory, but it also makes exact phrase matches much rarer. That means you often get **zero counts** for sequences you care about.

So a higher-order n-gram can become more brittle unless you add smoothing or collect a lot more data.

Another gotcha: beginners often forget that

```text
P("rice" | "eat")
```

is **not** the same as

```text
P("eat" | "rice")
```

The order matters. Language models predict the next word from the previous context, not the other way around.

## Files produced

- `page11_ngram_demo.py`
- `page11_run_output.txt`
- `page11_bigram_heatmap.png`
- `page11_prediction_after_eat.png`
- `page11_vocabulary_explosion.png`
