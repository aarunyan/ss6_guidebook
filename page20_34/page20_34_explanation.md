# page20_34 Attention mechanism explanation

This guide follows slide pages 20 to 34 of [AI_Full_Curriculum_EN.pptx.pptx](/Users/aarunyan/Documents/project/ss6_guidebook_new/AI_Full_Curriculum_EN.pptx.pptx). Those slides move through four connected ideas:

- slides 20-23: why attention replaced the old bottlenecked seq2seq approach
- slides 24-28: one token can look at all other tokens, not just one compressed sentence vector
- slides 29-31: transformer architectures are built from attention
- slides 32-34: scale and training changed later, but attention is the core mechanism that unlocked the jump

The runnable code is [page20_34_attention_demo.py](/Users/aarunyan/Documents/project/ss6_guidebook_new/page20_34/page20_34_attention_demo.py).

## Start with an analogy

Think of attention like a group study table.

- Every word in the sentence is a student.
- When one student wants to answer a question, they quickly look around the whole table.
- They do not ask just the person sitting next to them.
- They ask: "Who here has the most relevant clue for me right now?"

That is why the slides say attention changed everything. Instead of squeezing the whole sentence into one summary first, each token can look directly at the parts it needs.

Another analogy:

- Old seq2seq without attention is like writing an entire meeting onto one sticky note.
- Attention is like keeping the full meeting transcript open and highlighting the lines that matter for the current question.

For the sentence from slide 23:

`"The animal didn't cross the street because it was too tired"`

the token `it` is like someone asking, "Who are we talking about here?"  
Attention lets `it` look across the full sentence and place more weight on `animal` than on `street`.

## Draw a diagram

### 1. Old bottleneck idea

```text
whole sentence
      |
      v
[ one compressed vector ]
      |
      v
answer / next token
```

### 2. Attention idea

```text
query token: "it"
      |
      v
compare against every token key

"The"    "animal"   "didn't"   "cross"   "street"   "tired"
   \         |          |          |          |         /
    \        |          |          |          |        /
     \-------+----------+----------+----------+-------/
                    weighted focus
                         |
                         v
                new context for "it"
```

### 3. Query, key, value flow

```text
token embeddings
      |
      +--> Query matrix --> Q
      +--> Key matrix   --> K
      +--> Value matrix --> V

scores = Q x K^T
weights = softmax(scores)
output = weights x V
```

### 4. Transformer variants from the later slides

```text
Encoder-only:     token <--> token <--> token
Decoder-only:     token --> token --> token
Encoder-Decoder:  input tokens ==> encoder ==> decoder ==> output tokens
```

## Walk through the code

### 1. The script starts with a sentence chosen from the slides

The token list is:

```text
["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
```

This is deliberate. Slide 23 uses this exact style of example to show that attention helps resolve references like `it`.

### 2. Each token gets a tiny handcrafted embedding

The `EMBEDDINGS` matrix is not trying to be a real pretrained model. It is a teaching version.

Each row gives a token a few simple features:

- entity-ness
- action-ness
- location-ness
- pronoun-ness
- tiredness / condition

That keeps the math small enough to explain while still letting the model discover useful focus patterns.

### 3. The code builds queries, keys, and values

In `build_qkv()`, the script multiplies the embeddings by three weight matrices:

- `w_query`
- `w_key`
- `w_value`

That creates:

- `Q`: what each token is looking for
- `K`: what each token offers as an addressable clue
- `V`: what information each token contributes if selected

Conversationally: the query is a question, the key is a label on a folder, and the value is the content inside the folder.

### 4. The code computes attention scores

In `scaled_dot_attention()`, the main line is:

```text
scores = (queries @ keys.T) / sqrt(d)
```

That means:

- take one query
- compare it with every key
- produce a similarity score

Larger scores mean stronger relevance.

Then `softmax()` turns those scores into weights that sum to `1.0`.

So instead of saying "this token matters" in a vague way, the model says:

- `animal`: 0.31
- `it`: 0.18
- `tired`: 0.15
- and so on

### 5. The runtime trace shows attention live

When you run the script, `print_attention_walkthrough()` prints a line for each token being considered by the query token `it`.

It shows:

- the raw dot-product score
- the normalized attention probability
- an ASCII bar

So the run output is a live visualization, not just a final chart dump.

### 6. The script also saves image-based visualizations

It produces these files:

- `page20_34_attention_heatmap.png`
- `page20_34_it_focus.png`
- `page20_34_context_vectors.png`
- `page20_34_transformer_variants.png`
- `page20_34_mask_comparison.png`
- `page20_34_attention_scan.gif`

What they show:

- `attention_heatmap`: every token attending to every other token
- `it_focus`: a zoomed-in look at where `it` sends its attention
- `context_vectors`: how weighted attention mixes information into a new representation
- `transformer_variants`: encoder-only, decoder-only, and encoder-decoder patterns from slides 29-31
- `mask_comparison`: why decoder-only models cannot look into the future
- `attention_scan.gif`: a step-by-step scan across the candidate tokens

### 7. The code contrasts bidirectional attention with masked attention

The later slides introduce transformer families, so the demo includes both:

- bidirectional attention, like encoder-style understanding
- causal masked attention, like decoder-style generation

The same `it` token is evaluated twice:

- once with permission to inspect all tokens
- once with future tokens hidden

That mirrors the difference between models like `BERT` and models like `GPT`.

## Highlight a gotcha

A common misconception is:

> attention is just “finding the single most important word”

That is too simplistic.

Attention usually spreads probability across several useful tokens. In this example, `it` may focus strongly on `animal`, but it can also keep some weight on `tired` because that word contributes the condition being discussed.

Another gotcha:

> attention weights are the final answer

Not exactly. The weights decide **where to read from**, but the final representation comes from the weighted combination of the **value vectors**, not from the weights alone.

## How to run it

From the project root:

```bash
python3 page20_34/page20_34_attention_demo.py
```

If you want to use the bundled runtime that already exists in this workspace:

```bash
"/Users/aarunyan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3" page20_34/page20_34_attention_demo.py
```

## Slide connection summary

- slide 23: the pronoun-resolution example is the heart of the demo
- slides 24-28: the heatmap shows the move from one sentence vector to token-to-token interaction
- slides 29-31: the mask comparison and architecture diagram connect the attention mechanism to transformer variants
- slides 32-34: those slides discuss training, scale, and MoE, but attention is still the mechanism that makes the token interaction powerful enough to scale
