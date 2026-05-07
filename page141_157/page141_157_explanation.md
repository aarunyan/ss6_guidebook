# Local Development (Slides 141-157)

## 4.1 Start with an analogy
Think of local LLM deployment like moving furniture into your apartment.

- FP16 is bringing the full furniture set as-is: best quality, but it might not fit through the door (VRAM limit).
- Quantization is vacuum-packing furniture boxes: a little less perfect, but now it fits and is easier to move.
- Tools like Ollama and vLLM are different moving services: one is easier for daily use, one is optimized for heavy traffic.

Second analogy: KV cache in vLLM is like parking lots.

- Naive serving reserves giant parking spots for every possible car size, so most space sits empty.
- PagedAttention breaks parking into reusable slots, so only occupied slots are allocated.

## 4.2 Draw a diagram
```text
[Model Weights FP16]
        |
        v
[Quantization]
(FP16 -> INT8 -> INT4)
        |
        v
[Format Choice]
GGUF / GPTQ / AWQ
        |
        v
[Runtime Choice]
Ollama | llama.cpp | vLLM | TGI | LM Studio
        |
        v
[Local Inference]
(chat app / API / embedded app)

KV Cache (naive):
[Reserved........................................]
[Used.....]

KV Cache (paged):
[Page][Page][Page]...[Page]
  used pages only + page sharing
```

## 4.3 Walk through the code
1. `simulate_quantization_table()` creates a comparison table for FP16, GGUF Q4_K_M, GPTQ 4-bit, and AWQ 4-bit.
2. `estimate_memory_gb()` computes estimated weight memory from model size and bit-width.
3. `plot_quantization_tradeoff()` draws memory vs quality-drop, showing why INT4 formats help local deployment.
4. `plot_backend_choice()` visualizes speed/ease trade-offs across Ollama, llama.cpp, vLLM, TGI, and LM Studio.
5. `simulate_paged_attention_efficiency()` models token memory reservation under:
   - Naive allocation (reserve max length for every request)
   - Paged allocation (reserve only needed pages)
6. `plot_paged_attention()` turns the simulation into a chart showing lower waste with paged allocation.
7. `pick_tool()` gives simple recommendations based on use case text (beginner chat, production API, embedded C++).
8. `main()` runs all steps and saves:
   - `page141_157_run_output.json`
   - `page141_157_run_output.txt`
   - three visualization PNG files

## 4.4 Highlight a gotcha
Common misconception: "4-bit always means best choice."

Reality:
- 4-bit helps memory a lot, but format/runtime compatibility matters.
- GGUF is usually easiest for broad local use, GPTQ can be very fast on CUDA-heavy setups, and AWQ often preserves quality better.
- Tool choice (Ollama vs vLLM vs llama.cpp) can matter as much as quantization choice.

A practical rule: choose the format that your serving stack supports reliably first, then optimize for quality/speed.
