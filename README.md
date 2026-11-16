# Nanbeige Fast

A local Metal inference harness for the exact [MercuriusDream/Nanbeige4.2-3B-mlx-6bit checkpoint](https://huggingface.co/MercuriusDream/Nanbeige4.2-3B-mlx-6bit/tree/0ed232c61a153d4596bf3cf346f5a1a9b1acd870).
The launcher uses locked native arm64 packages. It downloads the weights, tokenizer and template, then runs offline. The model keeps its original affine 6-bit weights and both passes through the 22-layer stack.

You need an Apple Silicon Mac and `uv`. Run these commands from the checkout.

```sh
./nanbeige setup
./nanbeige chat --stats
./nanbeige run 'Explain what a torque wrench does in one sentence.' --stats
```

If the launcher is installed on your PATH, use `nanbeige chat --stats` from any directory.

Use `chat` to keep the weights resident across turns. It reuses the exact common token prefix, including previously generated text when the template produces the same IDs. The model uses the repository's direct-answer template and greedy decoding by default. `--think` enables its reasoning template; allow more output tokens for longer answers. Use `--temperature 0.6 --top-p 0.95` for sampled output.

The default context limit is 2,048 tokens, **including the requested output**. Full precision attention caches use about 176 KiB per token before allocation padding. Only raise `--context` as needed. `--kv-bits 8` or `4` is an independent, opt-in compression of the attention cache; it can change outputs and can cost speed even when it saves RAM. Neither changes the model's 6-bit weights.

On this 8 GB MacBook Neo, the default keeps the main weights resident, packs parallel projections, and reads only the needed rows of the 395.5 MiB input embedding. It measured **7.38 tokens/s** on the short test, with **3,009 MiB peak MLX allocation**. See [the benchmark notes](benchmarks/README.md) for the timings and conditions. Packing preserves the original weights, although batching and packing can choose different floating-point kernels. Use `--resident` to keep the input embedding in MLX memory as well.

If a VM is running and unified memory is tight, the launcher can switch to streaming groups of weights and keeping only some layers resident. This is slower but lets the exact model run when a normal resident model stalls. You can select it explicitly with `--stream-weights 2 --resident-layers 20`; lower the retained layer count if you need headroom. For highest sustained speed, close large VMs and other heavy applications, plug in, and turn off Low Power Mode.

## Memory

`chat` keeps recent turns in context and can recall short excerpts from older turns as the context fills. Recall runs locally using keyword matching and recency, so it works offline without an embedding model. The most recent user message and requested output still need to fit the context limit.

`chat` saves memory across sessions by default in `~/nanbeige-memory.sqlite3`. Start a chat with `./nanbeige chat` and use `/remember My workshop project is called Cedar.` to save a note. Quit and start another chat to recall it.

Use `./nanbeige chat --session-only` for a temporary chat with its own memory in RAM. Use `--memory PATH` to select a different database or to give a one-shot `run` access to a database.

Use `/memory` to list explicit notes, `/forget ID` to remove a note, and `/clear-memory` to erase notes and saved turns and start a new chat. `/reset` starts a new chat while keeping the database. In a session started with `--session-only`, `/reset` also clears its temporary memory. Forgetting a note does not remove separate turns that mention it.

The database stores explicit notes and bounded excerpts from completed turns as local, unencrypted text. You can use `./nanbeige memory PATH list` to inspect notes or `./nanbeige memory PATH history` to inspect saved turns without loading the model. That command also accepts `remember TEXT`, `forget ID`, and `clear`. Use different paths for separate memories. The local API stays stateless; its clients supply their conversation history as before.

## Local API

```sh
./nanbeige serve --port 8080
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MercuriusDream/Nanbeige4.2-3B-mlx-6bit","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}'
```

Set the base URL to `http://127.0.0.1:8080/v1` and use the model ID shown above. This is a text chat subset of the OpenAI API: model discovery, JSON responses, streaming with `"stream":true`, and optional `"stream_options":{"include_usage":true}`. The server keeps one model and handles one request at a time. It binds to loopback only. The response includes measured latency and token throughput.

## Measure

```sh
./nanbeige bench --suite compare --repeat 3 --json /tmp/nanbeige-compare.json
./nanbeige bench --suite cache --repeat 3 --json /tmp/nanbeige-cache.json
./nanbeige bench --suite prefill --repeat 2
./nanbeige bench --suite kv --repeat 2
./nanbeige bench --suite lookup --disk-embedding --fused
```

The comparison uses MLX-LM's upstream generation loop, then the local loop with the original projections, packed projections, and the disk-backed embedding. A warmup precedes each profile. The JSON records individual timings, medians, model revision, dependency versions, power mode, swap and greedy token hashes. Time to first token measures the first sampled token, which can precede the first complete piece of text. Peak MLX allocation is separate from total process memory and system swap.

Add `--lookup 4` to `run`, `chat`, or `serve` to have the 6-bit target verify guesses from repeated token patterns. It is most useful for copying, templated text and repeated code. It runs only with greedy decoding. The official [Nanbeige DSpark head](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-DSpark) is an experimental option. It uses about 455 MiB of additional packed weights at 4 bits (about 253 MiB at 2 bits) and more memory while packing. The CLI installs its separate locked dependencies only when invoked.

```sh
./nanbeige setup --dspark
./nanbeige bench-spec --modes baseline --disk-embedding --fused --json /tmp/nanbeige-base.json
./nanbeige bench-spec --modes dspark --disk-embedding --fused --draft-bits 2 --json /tmp/nanbeige-spec.json
./nanbeige spec --mode dspark --disk-embedding --fused --draft-bits 2 --stats 'Write a Python function to flatten a nested list.'
```

Run the tests with `uv run --locked python -m unittest discover -s tests -v`.

The implementation uses [Apple MLX and MLX-LM](https://github.com/ml-explore/mlx-lm/tree/v0.32.2) for the original Nanbeige layers and [mlx-dspark](https://github.com/ARahim3/mlx-dspark/tree/v0.20.4) for the optional draft head. See the [third-party notices](THIRD_PARTY_NOTICES.md) for MLX-LM attribution.
