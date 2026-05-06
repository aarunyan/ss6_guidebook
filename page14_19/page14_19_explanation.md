# page14_19 RNN, LSTM, and GRU explanation

This guide follows slides 14 to 19 of [AI_Full_Curriculum_EN.pptx.pdf](/Users/aarunyan/Documents/project/ss6_guidebook_new/AI_Full_Curriculum_EN.pptx.pdf). Those slides emphasize three ideas:

- `RNN`, `LSTM`, and `GRU` process information in sequence.
- `LSTM` and `GRU` were invented to remember useful things longer than a plain `RNN`.
- `seq2seq` translation with recurrent models can hit a **bottleneck**, because one fixed-size vector has to summarize the whole input sentence.

## Start with an analogy

Think of a plain `RNN` like a person trying to retell a story after hearing it once, one word at a time.

- Every new word slightly overwrites the previous memory.
- If the sentence gets long, the earlier details get fuzzy.

That is why slide 14 says it "remembers sequences but slow + forgets."

Now switch to an `LSTM`.

An `LSTM` is like carrying a notebook with three helpers:

- one helper decides what old notes to erase
- one helper decides what new notes are worth writing down
- one helper decides what part of the notebook to read out loud right now

That is what the gates do.

`GRU` is a similar idea, but lighter. It is like using a smaller notebook with fewer rules:

- one control decides how much old memory to keep
- one control decides how much the new information should replace it

For the translation slides, another analogy helps.

The old `seq2seq` encoder-decoder setup is like trying to summarize a whole lecture on one sticky note before handing it to someone else. If the lecture is long, that sticky note becomes the bottleneck.

## Draw a diagram

### 1. Plain recurrent flow

```text
token_1 --> [RNN state] --> token_2 --> [RNN state] --> token_3 --> [RNN state]
                |                           |                           |
             remembers                   remembers                   remembers
             a little                    a little                    a little
             and updates                 and updates                 and updates
```

### 2. Why LSTM is more careful

```text
previous memory ----> [forget gate] ----\
                                         +--> new cell state --> [output gate] --> hidden state
current token -----> [input gate] --> [candidate memory] --/
```

### 3. Why GRU is simpler

```text
previous hidden ----> [reset gate] ----> candidate state
         \                                   |
          \-----> [update gate] -------------+----> next hidden state
```

### 4. Seq2Seq bottleneck from slides 16-18

```text
source sentence
"I saw a cute cat"
      |
      v
[Encoder] ---> [one fixed-size context vector] ---> [Decoder] ---> "I saw a cat"
                     ^
                     |
               bottleneck here
```

## Walk through the code

The main script is [page14_19_rnn_lstm_gru_demo.py](/Users/aarunyan/Documents/project/ss6_guidebook_new/page14_19/page14_19_rnn_lstm_gru_demo.py).

### 1. We create a tiny sentence on purpose

The script uses:

```text
the cat sat on the mat while purring
```

Each token gets a small importance score.

- `cat` and `mat` get stronger signals
- filler words like `the` and `on` get weaker signals

This gives the models something to remember and something to ignore.

### 2. The plain RNN does one simple update

In `simulate_rnn()`, the hidden state is updated like this:

```text
new_state = tanh(old_state * something + new_signal * something)
```

That means the new state is always a blend of:

- what the model already remembered
- what just arrived

This is simple, but it also means old information is repeatedly mixed and can fade away.

### 3. The LSTM adds gates

In `simulate_lstm()`, the script computes:

- `forget_gate`
- `input_gate`
- `candidate`
- `output_gate`

Unlike the first draft of the demo, these gates now depend on both:

- the current token signal
- the previous hidden state

That makes the recurrence much closer to a real LSTM, where the next step depends on what the model was already thinking.

Then it updates:

```text
cell = forget_gate * old_cell + input_gate * candidate
hidden = output_gate * tanh(cell)
```

That extra cell state is the big idea.

It acts like a more protected memory lane, so useful information has a better chance of surviving across many steps.

### 4. The GRU keeps the same goal but with fewer moving parts

In `simulate_gru()`, the script computes:

- `reset_gate`
- `update_gate`
- `candidate`

Those gate values now also depend on the previous hidden state, which is how a real GRU decides whether to keep or overwrite what it already knows.

Then it blends:

```text
hidden = (1 - update_gate) * old_hidden + update_gate * candidate
```

So GRU is still trying to decide:

- how much old memory to keep
- how much new memory to write

but it does that with fewer gates than LSTM.

### 5. The script prints live text visualizations while running

When you run the file, it prints rows like:

```text
t= 2 token=cat      signal=1.00 state=+0.XXX [########....]
```

Those bars are not decoration. They make the sequential behavior visible in real time:

- one token arrives
- the state changes
- the next token arrives
- the state changes again

That directly matches the "sequential but limited" message in the slides.

### 6. The script saves chart-based visualizations

It generates:

- `page14_19_memory_trace.png`
- `page14_19_gate_heatmap.png`
- `page14_19_bottleneck.png`
- `page14_19_context_map.png`
- `page14_19_sequential_flow.gif`

What each one shows:

- `memory_trace`: how RNN, LSTM, and GRU carry memory through the same sentence
- `gate_heatmap`: how strongly LSTM and GRU open or close their gates
- `bottleneck`: the encoder-decoder compression problem from slides 16-18
- `context_map`: many source token vectors getting squeezed into one final summary vector
- `sequential_flow.gif`: a step-by-step animation showing that recurrent models process tokens one at a time

### 7. The seq2seq section recreates the bottleneck idea

The script uses a tiny source sentence:

```text
I saw a cute cat
```

Then it builds a sequence of encoder states and keeps only the final context vector.

That is the code version of the slide idea:

- the encoder reads the whole source sentence
- the decoder receives a fixed-size summary
- if the sentence is long, a lot of detail has to compete for limited space

## Highlight a gotcha

A common misconception is:

> "LSTM solves the memory problem completely."

Not really.

`LSTM` helps a lot, especially compared with a plain `RNN`, but it does **not** remove every problem.

Slides 18 and 19 make that clear:

- recurrent models are still sequential, so training is hard to parallelize
- they still struggle with very long-range context
- seq2seq models still hit bottlenecks when one fixed vector must summarize a long sentence

Another gotcha: this script is a **teaching simulation**, not a trained production model.

That is intentional.

The point here is to make the mechanisms visible:

- how state gets updated
- how gates behave
- how the encoder bottleneck appears

## Multiple analogies for the hard part

If the notebook analogy does not click, try these:

### Analogy 1: Cooking

A plain `RNN` is like tasting a soup over and over while new ingredients keep getting added. Earlier flavors are still there, but each new ingredient can drown them out.

An `LSTM` is like writing down the important ingredients before the pot gets too crowded.

### Analogy 2: Group chat

A plain `RNN` is like scrolling through a long chat and trying to remember the first useful message without pinning it.

An `LSTM` is like pinning key messages so they are harder to lose.

The seq2seq bottleneck is like asking someone to summarize the whole chat in one sentence. That works for short chats, but it breaks down once the conversation gets long.

## How to run it

From the project root:

```bash
./venv/bin/python page14_19/page14_19_rnn_lstm_gru_demo.py
```

The script prints a live walkthrough and saves all generated visuals into [page14_19](/Users/aarunyan/Documents/project/ss6_guidebook_new/page14_19).
