# page12_13 word2vec explanation

## Start with an analogy

Think of `word2vec` like seating guests at a dinner party and quietly watching who keeps ending up at the same table.

If **king**, **queen**, **prince**, and **princess** keep appearing around the same kinds of neighbors, you start to suspect they belong to the same social circle. If **paris** often appears near **france** and **capital**, while **bangkok** appears near **thailand** and **capital**, you start to see another pattern.

That is the heart of word2vec: it does not begin with dictionary definitions. It learns by watching **which words travel together**.

Another analogy: imagine every word gets a home address in a giant city. Words with similar jobs get houses in the same neighborhood. Royal words settle near royal words. Country and capital words settle near each other. The final vector is basically the word's address.

## Draw a diagram

```text
Training sentences
    |
    v
Build skip-gram pairs

"king rules a kingdom"
      ^
      center word tries to predict nearby words

    |
    v
Neural model adjusts word vectors

king ----\
queen ----> nearby points in vector space
prince --/

france ---\
paris  ----> another nearby cluster
thailand -/
bangkok --/

    |
    v
Vector arithmetic becomes meaningful

king - man + woman  ~= queen
paris - france + thailand ~= bangkok

    |
    v
Gotcha with static embeddings

"bank" (money) ----\
"bank" (river) -----> same word, same vector
"bank" (company) --/
```

## Walk through the code

### 1. We create a tiny corpus on purpose

The file [page12_13_word2vec_demo.py](/Users/aarunyan/Documents/project/ss6_guidebook/page12_13/page12_13_word2vec_demo.py) uses a small set of sentences instead of a giant dataset.

That makes the learning process visible:

- royal words appear in royal contexts
- country and capital words appear in similar contexts
- `bank` appears in both finance and river contexts

This mirrors slides 12 and 13 directly.

### 2. We turn text into training pairs

`generate_skipgram_pairs()` slides a small window over each sentence.

For example:

```text
"paris is the capital of france"
```

creates pairs such as:

```text
(paris -> is)
(paris -> the)
(capital -> the)
(capital -> of)
(france -> capital)
```

This is the skip-gram idea: the **center word** tries to predict nearby **context words**.

### 3. We build a very small neural version of word2vec

`TinyWord2Vec` stores two matrices:

- `input_vectors`: how words behave when they are the center word
- `output_vectors`: how words behave when they are the target context

During `train_one_pair()`:

1. the script picks one center word
2. it computes scores for every possible context word
3. it applies `softmax()` to turn scores into probabilities
4. it compares the prediction with the real context word
5. it nudges the vectors so the correct context becomes more likely next time

That repeated nudging is how meaning slowly appears.

### 4. We print live progress while the code runs

The `main()` function prints:

- vocabulary size
- example context snapshots
- training loss every 20 epochs
- nearest neighbors for `king`
- analogy results like `king - man + woman`
- the `bank` similarity check

So the script is not just saving files silently. It gives you a running narrative while it trains.

### 5. We save visualizations after training

The script generates these outputs:

- `page12_13_training_loss.png`: shows whether training is improving
- `page12_13_embedding_journey.gif`: animates how words move in vector space over time
- `page12_13_embedding_map.png`: a final 2D map of selected word vectors
- `page12_13_king_similarity.png`: shows which words are closest to `king`
- `page12_13_bank_problem.png`: shows why one static vector for `bank` is limiting

The GIF is especially useful because it shows the slide 12 idea visually: words that share context gradually move closer together.

### 6. We test vector arithmetic

The `analogy()` function implements the classic word2vec-style pattern:

```text
vector("king") - vector("man") + vector("woman")
```

Then it finds the nearest real word to that new vector.

This is the code version of the slide's claim that vectors can encode relationships, not just isolated meanings.

### 7. We connect to the page 13 “bank” problem

The script intentionally mixes these sentences:

- finance contexts: `loan`, `money`, `cash`
- river contexts: `muddy`, `trees`, `river`

Because plain word2vec gives each token only **one fixed vector**, the same vector has to represent both senses of `bank`.

That is exactly why slide 13 compares **static embeddings** with **contextualized embeddings** like BERT.

## Highlight a gotcha

A common misconception is: "word2vec understands meaning the way humans do."

Not really.

It learns **distributional meaning**:

- words used in similar contexts get similar vectors
- relationships can emerge from those patterns

But it still has blind spots.

The biggest one on these slides is polysemy:

```text
bank = financial institution
bank = side of a river
```

Classic word2vec gives both senses the **same address** in vector space. That is why the `page12_13_bank_problem.png` chart matters.

Another gotcha: the 2D map is only a **projection** of higher-dimensional vectors. If two words look a little farther apart on the plot than you expected, it does not necessarily mean the model thinks they are unrelated. PCA is compressing 12 dimensions down to 2 for visualization.

## Multiple analogies for the hard part

If the dinner-party analogy does not click, here are two more:

### Analogy 1: Music playlists

If two songs keep appearing in the same playlists, even if they are not identical, you assume they have a similar vibe.

Word2vec does the same with words and contexts.

### Analogy 2: Store shelves

If two products are repeatedly stocked near the same kinds of items, you infer they serve related purposes.

Words near similar context words get nearby vectors for the same reason.

## How to run it

From the project root:

```bash
./venv/bin/python page12_13/page12_13_word2vec_demo.py
```

The script prints progress to the terminal and saves the generated visuals into the same folder.
