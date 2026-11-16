# MacBook Neo measurements

November 16, 2026. Apple A18 Pro, 8 GiB unified memory, macOS 26.6.2, MLX 0.32.4 and MLX-LM 0.32.2. The Mac was on battery with Low Power Mode enabled. Times below are warmed request times, excluding checkpoint load, unless stated otherwise.

With Docker Desktop stopped:

| Runner | Prompt | Repetitions | First token | Decode | Peak MLX |
|---|---:|---:|---:|---:|---:|
| [Upstream MLX-LM](stock-short.json) | 58 tokens | 3 | 1.15 s | 7.22 tok/s | 3,401 MiB |
| [Default packed projections + disk-backed embedding](lean-first.json) | 58 tokens | 2 | 1.00 s | 7.38 tok/s | 3,009 MiB |
| [Same weights, first token from the last prefill block](lean-first.json) | 58 tokens | 2 | 1.01 s | 6.12 tok/s | 3,011 MiB |

Upstream emitted 32 tokens per sample; the two lean profiles emitted 16. The lean split and combined runs generated identical token IDs. The resident default uses split prefill based on this comparison. This is application speed in the recorded conditions; LPM, warm state, input size and background memory change the result.

[Lookup copy benchmark](lookup-copy.json): 343 input tokens and 24 output tokens. Greedy decode ran at 6.67 and 6.98 tok/s; lookup ran at 2.83 and 14.90 tok/s. All four samples generated the same token IDs. Lookup is opt-in.

[Live API check](api-smoke.json): JSON and streamed responses produced the same text. The first request took 30.38 s to its first token on this memory-pressured system. A repeated request reused 40 of 41 prompt tokens and took 0.26 s to its first token.

The [DSpark comparison baseline](dspark-baseline.json) ran at 6.58–7.21 tok/s. The official 2-bit draft head and target loaded when packed separately, but the end-to-end DSpark trial exceeded its 85 s budget before reporting a completion. No DSpark speedup is claimed on this Mac.

With Docker Desktop's 4 GiB VM running, the fully resident benchmark stalled before completing its first warmup. The following weight streaming runs used an identical 58-token prompt and 5-token output budget. Medians are from two runs per profile without warmup:

| Layers kept resident | First token | Decode | Peak MLX |
|---:|---:|---:|---:|
| [0](stream-keep0.json) | 3.75 s | 0.28 tok/s | 440 MiB |
| [8](stream-keep8.json) | 4.54 s | 0.33 tok/s | 1,340 MiB |
| [12](stream-keep12.json) | 4.37 s | 0.39 tok/s | 1,771 MiB |
| [16](stream-keep16.json) | 4.26 s | 0.62 tok/s | 2,215 MiB |
| [20](stream-keep20.json) | 2.88 s | 1.00 tok/s | 2,659 MiB |

The streamed profiles produced identical token IDs. GPU allocation is one part of the system's unified memory usage; these figures exclude CPU file cache, other processes, and OS swap.
