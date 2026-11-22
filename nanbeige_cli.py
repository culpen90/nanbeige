"""Local chat, prompt, benchmarks and API for the pinned 6-bit checkpoint."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sqlite3
from statistics import median
import subprocess
import sys
from time import perf_counter

from nanbeige_context import prepare_chat
from nanbeige_engine import MODEL_ID, REVISION, CheckpointUnavailableError, Engine, Settings, model_path
from nanbeige_gemini import DEFAULT_GEMINI_MODEL, GeminiClient
from nanbeige_memory import MemoryStore


def stderr(*args):
    print(*args, file=sys.stderr, flush=True)


def load_engine(settings, *, download_missing=False):
    started = perf_counter()
    stderr(f"Loading {MODEL_ID} on Metal…")
    try:
        engine = Engine(settings)
    except CheckpointUnavailableError:
        if not download_missing:
            raise
        stderr("Downloading the pinned checkpoint from Hugging Face (about 3.2 GiB for a fresh install)…")
        model_path(download=True)
        engine = Engine(settings)
    stderr(f"Ready in {perf_counter() - started:.2f}s")
    return engine


def show_stats(stats):
    stderr(
        f"[{stats.generated_tokens} tokens | first {stats.ttft_s:.2f}s | "
        f"{stats.decode_tps:.2f} tok/s | {stats.cached_tokens}/{stats.prompt_tokens} "
        f"prompt tokens reused | {stats.peak_mib:.0f} MiB peak MLX]"
    )


def settings_from(args):
    stream = args.stream_weights
    retained = args.resident_layers
    fused = args.fused
    disk = args.disk_embedding
    if (
        args.command in ("chat", "run", "serve")
        and not any((stream, args.layer_chunk, args.impl == "stock"))
    ):
        memory = sysinfo(["sysctl", "-n", "hw.memsize"])
        wired = wired_bytes()
        low_ram = bool(memory and memory.isdecimal() and int(memory) <= 8 * 2**30)
        docker = low_ram and sysinfo(["pgrep", "-f", "/Applications/Docker.app/Contents/MacOS/com.docker.virtualization"])
        if (low_ram and not args.resident and not (fused or disk) and wired
                and (wired > 4.0 * 2**30 or (docker and wired > 2.6 * 2**30))):
            stream = 2
            gib = wired / 2**30
            retained = 20 if gib < 3.5 else 12 if gib < 4.0 else 4 if gib < 4.5 else 0
            retained = max(0, retained - max(0, args.context - 2048) // 640)
            stderr(f"{gib:.1f} GiB wired on this 8 GiB Mac; streaming weights with {retained}/22 layers resident.")
        if not stream:
            fused = True if fused is None else fused
            disk = low_ram and not args.resident if disk is None else disk
    return Settings(
        context=args.context, prefill=args.prefill, kv_bits=args.kv_bits,
        cache_mib=args.cache_mib if args.cache_mib is not None else (32 if stream else 128),
        impl=args.impl, fused=bool(fused), disk_embedding=bool(disk), layer_chunk=args.layer_chunk,
        stream_weights=stream, resident_layers=retained, lookup=args.lookup,
        last_row_prefill=args.last_row_prefill,
    )


def messages_for(prompt, system):
    messages = [{"role": "system", "content": system}] if system is not None else []
    return messages + [{"role": "user", "content": prompt}]


def display(engine, messages, args, store=None, *, summary=""):
    output = []
    options = dict(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    if store is None:
        events = engine.generate(messages, thinking=args.think, **options)
    else:
        started = perf_counter()
        ids, messages = prepare_chat(
            engine, messages, max_tokens=args.max_tokens, thinking=args.think, store=store, summary=summary,
        )
        events = engine.generate_ids(ids, started=started, **options)
    try:
        for event in events:
            if event.text:
                print(event.text, end="", flush=True)
                output.append(event.text)
            if event.stats and args.stats:
                show_stats(event.stats)
    finally:
        events.close()
    print()
    return "".join(output), messages


def visible_reply(reply, thinking):
    _, closed, answer = reply.rpartition("</think>")
    if closed:
        return answer.lstrip("\n")
    return "" if thinking else reply


def read_prompt(args):
    if args.prompt is None and sys.stdin.isatty():
        raise ValueError("Pass a prompt or pipe text to stdin.")
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        raise ValueError("The prompt is blank.")
    return prompt


def note_id(value):
    try:
        number = int(value)
    except ValueError:
        number = 0
    if not 0 < number < 2**63:
        raise ValueError("Use a positive note ID from the memory list.")
    return number


def memory_command(store, action, value=None):
    if action == "remember":
        print(f"Saved note {store.remember(value)}.")
    elif action == "list":
        notes = store.notes()
        if not notes:
            print("No saved notes.")
        for number, text in notes:
            print(f"{number}\t{text!r}")
    elif action == "history":
        turns = store.history()
        if not turns:
            print("No saved turns.")
        for user, assistant in turns:
            print(f"you> {user!r}\nnanbeige> {assistant!r}")
    elif action == "forget":
        number = note_id(value)
        if not store.forget(number):
            raise ValueError(f"No saved note {number}.")
        print(f"Removed note {number}.")
    elif action == "clear":
        store.clear()
        print("Memory cleared.")


_CHAT_MEMORY_COMMANDS = {
    "/remember": "remember", "/memory": "list", "/forget": "forget", "/clear-memory": "clear",
}


def compression_client(args):
    if not args.compress_context:
        return None
    if args.compression_provider == "openrouter":
        if args.gemini_model is not None:
            raise ValueError("--gemini-model is for the Gemini compression provider.")
        if not args.openrouter_model:
            raise ValueError("Select an OpenRouter model using --openrouter-model MODEL.")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OpenRouter context compression needs OPENROUTER_API_KEY.")
        from nanbeige_openrouter import OpenRouterClient

        return OpenRouterClient(key, args.openrouter_model)

    if args.openrouter_model is not None:
        raise ValueError("--openrouter-model requires --compression-provider openrouter.")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("--compress-context needs a Google AI Studio key in GEMINI_API_KEY or GOOGLE_API_KEY.")
    model = DEFAULT_GEMINI_MODEL if args.gemini_model is None else args.gemini_model
    return GeminiClient(key, model)


def chat(engine, args, store, watcher=None):
    messages = messages_for("", args.system)[:-1]
    stderr("Enter a prompt. /remember TEXT saves a note; /memory lists notes; /forget ID removes a note.")
    stderr("/reset starts a new chat; /clear-memory clears the chat and saved memory; /quit exits.")
    if args.memory is not None:
        stderr(f"Saving chat memory to {str(args.memory)!r}.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if not prompt:
            continue
        command, *values = prompt.split(maxsplit=1)
        if command == "/reset" or command in _CHAT_MEMORY_COMMANDS:
            try:
                takes_value = command in ("/remember", "/forget")
                if takes_value != bool(values):
                    usage = {"/remember": " TEXT", "/forget": " ID"}
                    raise ValueError(f"Usage: {command}{usage.get(command, '')}")
                if command == "/reset":
                    if args.memory is None:
                        store.clear()
                else:
                    memory_command(store, _CHAT_MEMORY_COMMANDS[command], values[0] if values else None)
                if command in ("/reset", "/clear-memory"):
                    engine.reset()
                    messages = messages_for("", args.system)[:-1]
                    if watcher is not None:
                        watcher.reset()
            except (OSError, ValueError, sqlite3.Error) as exc:
                stderr(exc)
            continue
        if watcher is not None:
            messages = watcher.apply(messages)
        turn = [*messages, {"role": "user", "content": prompt}]
        try:
            print("nanbeige> ", end="", flush=True)
            reply, retained = display(engine, turn, args, store, summary=watcher.summary if watcher else "")
        except KeyboardInterrupt:
            print()
            stderr("Generation interrupted.")
            continue
        except (OSError, ValueError, sqlite3.Error) as exc:
            print()
            stderr(exc)
            continue
        answer = visible_reply(reply, args.think)
        assistant = {"role": "assistant", "content": answer}
        messages = [*retained, assistant] if watcher is None else watcher.observe([*turn, assistant])
        try:
            store.record(prompt, answer)
        except (OSError, ValueError, sqlite3.Error) as exc:
            stderr(f"Couldn't save this turn: {exc}")


def sysinfo(cmd):
    try:
        return subprocess.check_output(cmd, text=True, timeout=3, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def wired_bytes():
    stats = sysinfo(["vm_stat"]) or ""
    page = re.search(r"page size of (\d+) bytes", stats)
    wired = re.search(r"Pages wired down:\s+(\d+)\.", stats)
    if page and wired:
        return int(page[1]) * int(wired[1])
    return None


PROMPTS = {
    "short": "Explain how a bicycle gear works in six short numbered steps. "
             "Use a concrete example with small and large sprockets.",
    "copy": "Print the following sequence exactly, starting with the first number and with no explanation:\n"
            + " ".join(str(n) for n in range(1, 101)),
    "long": "Here is a log of observations from a weather station:\n"
            + "\n".join(
                f"Day {i}: temperature {10 + i % 17} C, wind {4 + i % 11} km/h, "
                f"rain {i % 8} mm." for i in range(1, 58)
            ) + "\nWrite a brief report on patterns in the log. Give several examples with specific values.",
}


def record(engine, ids, maximum, reuse):
    tokens = []
    for event in engine.generate_ids(ids, max_tokens=maximum, reuse=reuse):
        if event.token is not None:
            tokens.append(event.token)
        if event.stats:
            stats = asdict(event.stats)
            stats["finish_reason"] = event.finish_reason
    stats["token_sha256"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    return stats


def new_report(args, suite):
    power = sysinfo(["pmset", "-g", "batt"])
    return {
        "model": MODEL_ID, "revision": REVISION,
        "hardware": sysinfo(["sysctl", "-n", "hw.model", "machdep.cpu.brand_string", "hw.memsize"]),
        "os": sysinfo(["sw_vers", "-productVersion"]),
        "power": re.sub(r" \(id=\d+\)", "", power) if power else power,
        "low_power_mode": next((line.strip() for line in (sysinfo(["pmset", "-g"]) or "").splitlines()
                                if "lowpowermode" in line), None),
        "swap_before": sysinfo(["sysctl", "-n", "vm.swapusage"]),
        "wired_bytes": wired_bytes(),
        "versions": {p: version(p) for p in ("mlx", "mlx-lm", "transformers")},
        "max_tokens": args.max_tokens, "suite": suite, "runs": [],
    }


def append_results(report, profile, workload, settings, runs, path):
    report["runs"].append({
        "profile": profile, "workload": workload, "settings": asdict(settings),
        "samples": runs,
        "median": {stat: median(run[stat] for run in runs) for stat in (
            "ttft_s", "decode_tps", "prefill_tps", "wall_s", "peak_mib",
        )},
    })
    if path:
        report["swap_after"] = sysinfo(["sysctl", "-n", "vm.swapusage"])
        path.write_text(json.dumps(report, indent=2) + "\n")


def bench(engine, args):
    initial = engine.settings
    name = initial.impl + ("-fused" if initial.fused else "") + ("-disk" if initial.disk_embedding else "")
    if initial.stream_weights:
        name += f"-stream-{initial.stream_weights}-keep-{initial.resident_layers}"
    profiles = [(name, initial, args.workload)]
    if args.suite == "compare":
        if initial.fused or initial.disk_embedding or initial.stream_weights:
            raise ValueError("The compare suite starts from original weight placement and projections.")
        fast = replace(initial, impl="fast")
        profiles = [
            ("stock-default", replace(
                initial, impl="stock", prefill=2048, kv_bits=None, layer_chunk=0, lookup=0
            ), args.workload),
            ("fast", fast, args.workload),
            ("fast-fused", replace(fast, fused=True), args.workload),
            ("fast-fused-disk", replace(fast, fused=True, disk_embedding=True), args.workload),
        ]
    elif args.suite == "prefill":
        profiles = [
            (f"prefill-{size}", replace(initial, prefill=size), "long")
            for size in (128, 256, 512, 1024)
        ]
    elif args.suite == "kv":
        profiles = [
            (f"kv-{bits or 'full'}", replace(initial, kv_bits=bits), "long")
            for bits in (None, 8, 4)
        ]
    elif args.suite == "cache":
        profiles = [("cold-prefix", initial, "long"), ("warm-prefix", initial, "long")]
    elif args.suite == "lookup":
        profiles = [
            ("greedy", replace(initial, lookup=0), "copy"),
            ("lookup", replace(initial, lookup=initial.lookup or 4), "copy"),
        ]
    elif args.suite == "first":
        profiles = [
            ("split-prefill", replace(initial, last_row_prefill=False), args.workload),
            ("last-row-prefill", replace(initial, last_row_prefill=True), args.workload),
        ]

    report = new_report(args, args.suite)
    if not 1 <= args.repeat <= 10:
        raise ValueError("Use between 1 and 10 benchmark repetitions.")
    for label, settings, workload in profiles:
        engine.configure(settings)
        ids = engine.encode(messages_for(PROMPTS[workload], None), False)
        stderr(f"{label}: {len(ids)} prompt tokens")
        if not args.no_warmup:
            record(engine, ids, 2, False)
        if label == "warm-prefix":
            # Seed with a long shared prefix and a different question. The timed
            # request must actually compare and trim tokens rather than match text.
            seed = engine.encode(messages_for(
                PROMPTS[workload].rsplit("\n", 1)[0] + "\nSummarize the log in one sentence.",
                None,
            ), False)
            record(engine, seed, 4, True)
        runs = []
        for attempt in range(args.repeat):
            if label == "warm-prefix" and attempt:
                engine.reset()
                record(engine, seed, 4, True)
            stats = record(engine, ids, args.max_tokens, label == "warm-prefix")
            runs.append(stats)
            stderr(
                f"{label:16s} {attempt + 1}: prefill {stats['prefill_tps']:7.1f} tok/s  "
                f"first {stats['ttft_s']:6.3f}s  decode {stats['decode_tps']:6.2f} tok/s  "
                f"peak {stats['peak_mib']:.0f} MiB  prefix {stats['cached_tokens']}"
            )
        append_results(report, label, workload, settings, runs, args.json)
    if len(profiles) > 1:
        hashes = {sample["token_sha256"] for row in report["runs"] for sample in row["samples"]}
        stderr("Greedy token IDs match across runs." if len(hashes) == 1 else "Greedy token IDs differ across runs.")
    return 0


def preload_draft(bits):
    from huggingface_hub.errors import LocalEntryNotFoundError
    from nanbeige_dspark import load_compact_drafter

    stderr(f"Packing the official DSpark head to {bits} bits…")
    try:
        return load_compact_drafter(bits)
    except LocalEntryNotFoundError as exc:
        raise ValueError("Official drafter missing. Run ./nanbeige setup --dspark first.") from exc


def spec_runner(engine, mode, cap, draft_bits, drafter=None):
    from nanbeige_dspark import DSparkRunner
    from huggingface_hub.errors import LocalEntryNotFoundError

    settings = engine.settings
    if settings.layer_chunk or settings.impl != "fast":
        raise ValueError("Speculation uses its own layer scheduling; omit --layer-chunk and --impl stock.")
    try:
        return DSparkRunner(
            engine.model, engine.tokenizer, mode=mode, draft_cap=cap,
            context=settings.context, prefill=settings.prefill,
            kv_bits=settings.kv_bits, prefix_slots=0, local_only=True,
            streamed=getattr(engine, "streaming_weights", None), draft_bits=draft_bits,
            preloaded_drafter=drafter,
        )
    except LocalEntryNotFoundError as exc:
        raise ValueError("Official drafter missing. Run ./nanbeige setup --dspark first.") from exc


def spec_wiring(engine):
    return (nullcontext() if engine.settings.stream_weights
            else engine.upstream.wired_limit(engine.model))


def bench_spec(engine, args, drafter=None):
    if not 1 <= args.repeat <= 10:
        raise ValueError("Use between 1 and 10 benchmark repetitions.")
    report = new_report(args, "spec")
    report["versions"]["mlx-dspark"] = version("mlx-dspark")
    report["draft_cap"] = args.draft_cap
    report["draft_bits"] = args.draft_bits
    ids = engine.prepare(messages_for(PROMPTS[args.workload], None), args.max_tokens, False)
    reference = None
    for mode in args.modes:
        stderr(f"{mode}: {len(ids)} prompt tokens")
        runner = spec_runner(engine, mode, args.draft_cap, args.draft_bits, drafter)
        if not args.no_warmup:
            with spec_wiring(engine):
                runner.generate(ids, max_tokens=2, use_cache=False)
        runs = []
        for _ in range(args.repeat):
            engine.mx.reset_peak_memory()
            started = perf_counter()
            with spec_wiring(engine):
                result = runner.generate(ids, max_tokens=args.max_tokens, use_cache=False)
            wall = perf_counter() - started
            digest = hashlib.sha256(json.dumps(result.token_ids).encode()).hexdigest()
            if reference is None and mode == "baseline":
                reference = digest
            sample = dict(
                prompt_tokens=len(ids), generated_tokens=result.num_tokens,
                ttft_s=result.ttft_seconds, prefill_tps=len(ids) / max(result.prefill_seconds, 1e-9),
                decode_tps=result.decode_tokens_per_sec, wall_s=wall,
                peak_mib=engine.mx.get_peak_memory() / 2**20, token_sha256=digest,
                exact_match=None if reference is None else digest == reference,
                mean_accept=result.mean_accept_len, finish_reason=result.finish_reason,
            )
            runs.append(sample)
            stderr(f"{mode}: {sample['decode_tps']:.2f} tok/s, {wall:.2f}s total, "
                   f"{sample['peak_mib']:.0f} MiB peak, exact match: {sample['exact_match']}")
        append_results(report, mode, args.workload, engine.settings, runs, args.json)
        del runner
        engine.mx.clear_cache()


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--context", type=int, default=2048, help="total prompt + output token budget")
    common.add_argument("--prefill", type=int, default=256, help="maximum prompt chunk in tokens")
    common.add_argument("--cache-mib", type=int, help="MLX allocator reuse buffer limit")
    common.add_argument("--kv-bits", type=int, choices=[4, 8], help="opt-in compressed attention cache")
    common.add_argument("--impl", choices=["fast", "stock"], default="fast")
    common.add_argument("--fused", action=argparse.BooleanOptionalAction,
                        help="pack parallel QKV and gate/up projections")
    common.add_argument("--disk-embedding", action=argparse.BooleanOptionalAction,
                        help="read only needed embedding rows, reducing Metal memory by 395 MiB")
    common.add_argument("--layer-chunk", type=int, default=0,
                        help="synchronize between groups of this many layers under memory pressure")
    common.add_argument("--stream-weights", type=int, default=0,
                        help="reduce peak RAM by reading weights in groups of 1 to 4 layers")
    common.add_argument("--resident-layers", type=int, default=0,
                        help="retain up to 20 layers when streaming weights")
    common.add_argument("--resident", action="store_true", help="disable automatic weight streaming")
    common.add_argument("--lookup", type=int, default=0,
                        help="verify up to this many token guesses from the prompt/history per round")
    common.add_argument("--last-row-prefill", action=argparse.BooleanOptionalAction, default=False,
                        help="compute first-token logits in the final prefill block")

    generation = argparse.ArgumentParser(add_help=False)
    generation.add_argument("--max-tokens", type=int, default=256)
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-p", type=float, default=1.0)
    generation.add_argument("--think", action="store_true", help="enable the model's reasoning template")
    generation.add_argument("--system")
    generation.add_argument("--stats", action="store_true")
    parser = argparse.ArgumentParser(description=__doc__, epilog="Run without a command to start chat.")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="download the pinned checkpoint to the Hugging Face cache")
    setup.add_argument("--dspark", action="store_true", help="also download the official speculative draft head")
    run = sub.add_parser("run", parents=[common, generation], help="answer a prompt (or read stdin)")
    run.add_argument("prompt", nargs="?")
    run.add_argument("--memory", type=Path, metavar="PATH", help="recall and save turns in this local database")
    interactive = sub.add_parser("chat", parents=[common, generation], help="keep the model hot across turns")
    chat_memory = interactive.add_mutually_exclusive_group()
    chat_memory.add_argument("--memory", type=Path, metavar="PATH",
                             help="use a different database (default: ~/nanbeige-memory.sqlite3)")
    chat_memory.add_argument("--session-only", action="store_true", help="keep memory only for this session")
    interactive.add_argument("--compress-context", action="store_true",
                             help="let a separate model monitor chat and decide when to compress history")
    interactive.add_argument("--compression-provider", choices=("gemini", "openrouter"), default="gemini",
                             help="context watcher provider (default: gemini)")
    interactive.add_argument("--gemini-model", metavar="MODEL",
                             help=f"Gemini context watcher model (default: {DEFAULT_GEMINI_MODEL})")
    interactive.add_argument("--openrouter-model", metavar="MODEL",
                             help="OpenRouter model with structured outputs (OPENROUTER_API_KEY required)")
    memory = sub.add_parser("memory", help="inspect or manage a local memory database without loading the model")
    memory.add_argument("path", type=Path)
    actions = memory.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="list explicit notes")
    actions.add_parser("history", help="show saved excerpts of past turns")
    actions.add_parser("remember", help="add an explicit note").add_argument("value", metavar="TEXT")
    actions.add_parser("forget", help="remove an explicit note").add_argument("value", metavar="ID")
    actions.add_parser("clear", help="delete all notes and past turns")
    spec = sub.add_parser("spec", parents=[common, generation], help="try target-verified speculative decoding")
    spec.add_argument("prompt", nargs="?")
    spec.add_argument("--mode", choices=["lookup", "dspark"], default="lookup")
    spec.add_argument("--draft-cap", type=int, choices=range(1, 8), default=3)
    spec.add_argument("--draft-bits", type=int, choices=[2, 4], default=4)
    api = sub.add_parser("serve", parents=[common], help="serve local OpenAI-style chat requests")
    api.add_argument("--host", choices=["127.0.0.1", "::1"], default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)
    sub.add_parser("api-key", help="print the local API key, creating one on first use")
    benchmark = argparse.ArgumentParser(add_help=False)
    benchmark.add_argument("--workload", choices=list(PROMPTS), default="short")
    benchmark.add_argument("--max-tokens", type=int, default=48)
    benchmark.add_argument("--repeat", type=int, default=3)
    benchmark.add_argument("--no-warmup", action="store_true", help="include cold kernel start in timings")
    benchmark.add_argument("--json", type=Path)
    regular = sub.add_parser("bench", parents=[common, benchmark], help="measure latency, speed, and peak memory")
    regular.add_argument("--suite", choices=["compare", "single", "prefill", "kv", "cache", "lookup", "first"],
                         default="compare")
    speculative = sub.add_parser("bench-spec", parents=[common, benchmark], help="compare speculative decoding modes")
    speculative.add_argument("--modes", nargs="+", choices=["baseline", "lookup", "dspark"],
                             default=["baseline", "lookup", "dspark"])
    speculative.add_argument("--draft-cap", type=int, choices=range(1, 8), default=3)
    speculative.add_argument("--draft-bits", type=int, choices=[2, 4], default=4)
    args = parser.parse_args(sys.argv[1:] or ["chat"])
    try:
        if args.command == "api-key":
            from nanbeige_api_key import get_api_key

            print(get_api_key())
            return 0
        if args.command == "setup":
            print(model_path(download=True))
            if args.dspark:
                from nanbeige_dspark import fetch_drafter

                print(fetch_drafter())
            return 0
        if args.command == "memory":
            with MemoryStore(args.path) as store:
                memory_command(store, args.action, getattr(args, "value", None))
            return 0
        settings = settings_from(args)
        if args.command == "serve":
            from nanbeige_server import serve

            serve(lambda: load_engine(settings), host=args.host, port=args.port)
            return 0
        if args.command == "chat":
            client = compression_client(args)
            if args.memory is None and not args.session_only:
                args.memory = Path.home() / "nanbeige-memory.sqlite3"
            with MemoryStore(args.memory) as store:
                engine = load_engine(settings, download_missing=True)
                watcher = nullcontext()
                if client is not None:
                    from nanbeige_compression import CompressionWatcher

                    stderr(f"Context watcher enabled. Completed chat turns are sent to {client.provider_name} "
                           f"({client.model}).")
                    watcher = CompressionWatcher(client, engine, max_tokens=args.max_tokens, thinking=args.think,
                                                 report=stderr)
                with watcher as active_watcher:
                    chat(engine, args, store, active_watcher)
            return 0
        if args.command == "run":
            prompt = read_prompt(args)
            memory = MemoryStore(args.memory) if args.memory is not None else nullcontext()
            with memory as store:
                reply, _ = display(load_engine(settings), messages_for(prompt, args.system), args, store)
                if store is not None:
                    store.record(prompt, visible_reply(reply, args.think))
            return 0
        if args.command == "spec":
            messages = messages_for(read_prompt(args), args.system)
            drafter = preload_draft(args.draft_bits) if args.mode == "dspark" else None
            engine = load_engine(settings)
            ids = engine.prepare(messages, args.max_tokens, args.think)
            runner = spec_runner(engine, args.mode, args.draft_cap, args.draft_bits, drafter)
            with spec_wiring(engine):
                result = runner.generate(
                    ids, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                    on_text=lambda s: print(s, end="", flush=True), use_cache=False,
                )
            print()
            if args.stats:
                stderr(f"[{result.num_tokens} tokens | first text {result.ttft_seconds:.2f}s | "
                       f"{result.decode_tokens_per_sec:.2f} tok/s | {result.mean_accept_len:.2f} tokens/round]")
            return 0
        drafter = (
            preload_draft(args.draft_bits)
            if args.command == "bench-spec" and args.modes == ["dspark"]
            else None
        )
        engine = load_engine(settings)
        return bench_spec(engine, args, drafter) if args.command == "bench-spec" else bench(engine, args)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
        stderr(f"nanbeige: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
