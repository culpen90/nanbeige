"""A resident, single-stream runner for one pinned Nanbeige checkpoint."""

from __future__ import annotations

from contextlib import closing, nullcontext
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Iterator

MODEL_ID = "MercuriusDream/Nanbeige4.2-3B-mlx-6bit"
REVISION = "0ed232c61a153d4596bf3cf346f5a1a9b1acd870"
MODEL_FILES = (
    "config.json", "generation_config.json", "chat_template.jinja",
    "tokenizer.json", "tokenizer_config.json",
    "model.safetensors", "model.safetensors.index.json",
)


class CheckpointUnavailableError(ValueError):
    """The pinned checkpoint is missing or incomplete in the local cache."""


def model_path(*, download: bool = False) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(
            MODEL_ID, revision=REVISION, allow_patterns=MODEL_FILES,
            local_files_only=not download, max_workers=2,
        ))
    except LocalEntryNotFoundError as exc:
        if download:
            raise
        raise CheckpointUnavailableError("Checkpoint missing or incomplete. Run ./nanbeige setup.") from exc
    if missing := [name for name in MODEL_FILES if not (path / name).is_file()]:
        raise CheckpointUnavailableError(f"Checkpoint is incomplete: {', '.join(missing)}. Run ./nanbeige setup.")
    return path


@dataclass(frozen=True)
class Settings:
    context: int = 2048
    prefill: int = 256
    cache_mib: int = 128
    kv_bits: int | None = None
    impl: str = "fast"
    fused: bool = False
    disk_embedding: bool = False
    layer_chunk: int = 0
    stream_weights: int = 0
    resident_layers: int = 0
    lookup: int = 0
    last_row_prefill: bool = False

    def __post_init__(self) -> None:
        if self.context not in range(32, 16385):
            raise ValueError("Context must be between 32 and 16384 tokens.")
        if self.prefill not in range(1, 2049):
            raise ValueError("Prefill chunk must be between 1 and 2048 tokens.")
        if not 0 <= self.cache_mib <= 1024:
            raise ValueError("Allocator cache must be between 0 and 1024 MiB.")
        if self.kv_bits not in (None, 4, 8):
            raise ValueError("Supported KV cache sizes are 4, 8, or full precision.")
        if self.impl not in ("fast", "stock"):
            raise ValueError("Unknown generation implementation.")
        if not 0 <= self.layer_chunk <= 22:
            raise ValueError("Layer chunk must be between 0 and 22.")
        if not 0 <= self.stream_weights <= 4:
            raise ValueError("Weight streaming groups must have 1 to 4 layers.")
        if not 0 <= self.resident_layers <= 20 or (self.resident_layers and not self.stream_weights):
            raise ValueError("Keep up to 20 resident layers when using weight streaming.")
        if not 0 <= self.lookup <= 8:
            raise ValueError("Lookup verification can propose up to 8 tokens per round.")
        if self.stream_weights and (self.fused or self.layer_chunk):
            raise ValueError("Weight streaming already stages the original projections.")
        if self.impl == "stock" and any((
            self.fused, self.disk_embedding, self.layer_chunk, self.stream_weights, self.lookup,
        )):
            raise ValueError("Stock measurements require original weight placement and projections.")


@dataclass(frozen=True)
class Stats:
    prompt_tokens: int
    generated_tokens: int
    cached_tokens: int
    ttft_s: float
    decode_tps: float
    prefill_tps: float
    wall_s: float
    peak_mib: float


@dataclass(frozen=True)
class Event:
    text: str
    token: int | None = None
    finish_reason: str | None = None
    stats: Stats | None = None


def common_prefix(incoming: list[int], previous: list[int], offset: int) -> int:
    # The final prompt token must still be evaluated to produce logits.
    cap = min(max(0, len(incoming) - 1), len(previous), offset)
    for index in range(cap):
        if incoming[index] != previous[index]:
            return index
    return cap


def lookup_tokens(history: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    for width in range(min(8, (len(history) - 1) // 2), 2, -1):
        suffix = history[-width:]
        for start in range(len(history) - width - 1, -1, -1):
            if history[start:start + width] == suffix:
                return history[start + width:start + width + limit]
    return []


class Engine:
    model_id = MODEL_ID

    def __init__(self, settings: Settings = Settings()):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.nanbeige import Model
        from transformers import PreTrainedConfig

        self.settings = settings
        self.mx = mx
        self.upstream = import_module("mlx_lm.generate")
        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable; use native arm64 Python on Apple Silicon.")
        mx.set_cache_limit(settings.cache_mib * 2**20)
        # AutoTokenizer needs tokenizer.json and tokenizer_config.json, not the
        # checkpoint's Transformers model class. Supply metadata locally.
        tokenizer_model_config = PreTrainedConfig()
        tokenizer_model_config.model_type = "nanbeige"
        self.checkpoint = model_path()
        self.model, self.tokenizer, config = load(
            str(self.checkpoint),
            lazy=bool(settings.disk_embedding or settings.stream_weights),
            tokenizer_config={
                "config": tokenizer_model_config,
                "trust_remote_code": False,
                "local_files_only": True,
            },
            return_config=True,
            trust_remote_code=False,
        )
        if (
            not isinstance(self.model, Model)
            or self.model.args.effective_num_loops != 2
            or len(self.model.layers) != 22
            or config["quantization"]["bits"] != 6
        ):
            raise RuntimeError("Unexpected checkpoint architecture or quantization.")
        if settings.stream_weights:
            from nanbeige_stream_weights import StreamedWeights

            self.streaming_weights = StreamedWeights(
                self.model, self.checkpoint / "model.safetensors", layers_per_group=settings.stream_weights,
                resident_layers=settings.resident_layers,
            )
        elif settings.disk_embedding:
            self._map_embedding()
            mx.eval(self.model.parameters())
        if settings.fused:
            from nanbeige_fuse import fuse_projections

            fuse_projections(self.model)
        self.stream = mx.new_stream(mx.gpu)
        self.cache = None
        self.cache_tokens: list[int] = []
        self._busy = False

    def reset(self) -> None:
        self.cache = None
        self.cache_tokens = []

    def _map_embedding(self) -> None:
        from nanbeige_embedding import map_input_embedding

        map_input_embedding(self.model, self.checkpoint / "model.safetensors")

    def configure(self, settings: Settings) -> None:
        """Change benchmark settings that don't require transforming the weights."""
        if (
            (self.settings.fused and not settings.fused)
            or (self.settings.disk_embedding and not settings.disk_embedding)
            or self.settings.stream_weights != settings.stream_weights
            or self.settings.resident_layers != settings.resident_layers
            or self._busy
        ):
            raise ValueError("This configuration requires a new, idle engine.")
        self.reset()
        if settings.disk_embedding and not (self.settings.disk_embedding or settings.stream_weights):
            self._map_embedding()
        if settings.fused and not self.settings.fused:
            from nanbeige_fuse import fuse_projections

            fuse_projections(self.model)
        self.settings = settings
        self.mx.set_cache_limit(settings.cache_mib * 2**20)
        self.mx.clear_cache()

    def encode(self, messages: list[dict], thinking: bool) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
            preserve_thinking=False,
        )
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def prepare(self, messages: list[dict], max_tokens: int, thinking: bool) -> list[int]:
        ids = self.encode(messages, thinking)
        self._check_context(ids, max_tokens)
        return ids

    def _check_context(self, ids: list[int], max_tokens: int) -> None:
        if not ids:
            raise ValueError("The prompt contains no tokens.")
        if not 1 <= max_tokens <= self.settings.context - len(ids):
            raise ValueError(
                f"Prompt has {len(ids)} tokens; requested {max_tokens} output tokens "
                f"with a {self.settings.context}-token total context limit."
            )

    def _new_cache(self):
        if self.settings.kv_bits is None:
            return self.model.make_cache()
        from mlx_lm.models.cache import QuantizedKVCache

        return [QuantizedKVCache(group_size=64, bits=self.settings.kv_bits) for _ in range(44)]

    def _cache_for(self, ids: list[int], reuse: bool):
        cache = self.cache if reuse else None
        reused = common_prefix(ids, self.cache_tokens, cache[0].offset) if cache else 0
        self.reset()
        if not reused:
            return self._new_cache(), 0
        for slot in cache:
            slot.trim(slot.offset - reused)
        return cache, reused

    def _staged_forward(self, x, cache, *, logits):
        """Finish small groups of layers before submitting more work to Metal."""
        from mlx_lm.models.base import create_attention_mask

        inner = self.model.model
        hidden = inner.embed_tokens(x)
        mask = create_attention_mask(hidden, cache[0])
        count = len(inner.layers)
        chunk = self.settings.layer_chunk
        for loop in range(inner.num_loops):
            for i, layer in enumerate(inner.layers):
                hidden = layer(hidden, mask, cache[loop * count + i])
                if (i + 1) % chunk == 0 and (i + 1 < count or loop + 1 < inner.num_loops):
                    self.mx.eval(hidden)
            if not inner.skip_loop_final_norm:
                hidden = inner.norm(hidden)
        if logits:
            if inner.skip_loop_final_norm:
                hidden = inner.norm(hidden)
            return self.model.lm_head(hidden)

    def _fast_tokens(self, prompt, cache, max_tokens, temperature, sampler, history) -> Iterator[int]:
        mx = self.mx
        settings = self.settings
        model = self.model
        with mx.stream(self.stream):
            prefill = (
                min(settings.prefill, self.streaming_weights.max_prompt_chunk)
                if settings.stream_weights else settings.prefill
            )
            def forward(x, *, logits, last_only=False):
                if settings.stream_weights:
                    return self.streaming_weights.forward(x, cache, logits=logits, last_only=last_only)
                if settings.layer_chunk:
                    return self._staged_forward(x, cache, logits=logits)
                if logits and last_only and x.shape[1] > 1:
                    hidden = model.model(x, cache=cache)
                    return model.lm_head(hidden[:, -1:, :])
                return model(x, cache=cache) if logits else model.model(x, cache=cache)

            # Let the final prompt block produce the first token using only
            # its last logits. A separate last-token forward rereads weights.
            combined = settings.stream_weights or (settings.last_row_prefill and not settings.layer_chunk)
            prefill_length = prompt.size if combined else prompt.size - 1
            first_input = prompt[-1:]
            for start in range(0, prefill_length, prefill):
                block = prompt[start:min(start + prefill, prefill_length)]
                if combined and start + block.size == prefill_length:
                    first_input = block
                    break
                # For prefix tokens the only observable result is KV state.
                forward(block[None], logits=False)
                mx.eval([slot.state for slot in cache])

            def step(x):
                logits = forward(x[None], logits=True, last_only=True)[0, -1, :]
                if temperature == 0:
                    return mx.argmax(logits, axis=-1).reshape(1)
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                return sampler(logprobs).reshape(1)

            current = step(first_input)
            mx.async_eval(current)
            if settings.lookup and temperature == 0:
                tokens = list(history)
                token = current.item()
                tokens.append(token)
                yield token
                remaining = max_tokens - 1
                while remaining:
                    draft = lookup_tokens(tokens, min(remaining - 1, settings.lookup))
                    if not draft:
                        output = [step(mx.array([token], dtype=mx.uint32)).item()]
                    else:
                        inputs = mx.array([token, *draft], dtype=mx.uint32)
                        predictions = mx.argmax(forward(inputs[None], logits=True)[0], axis=-1).tolist()
                        accepted = 0
                        while accepted < len(draft) and draft[accepted] == predictions[accepted]:
                            accepted += 1
                        output = predictions[:accepted + 1]
                        for index, candidate in enumerate(output):
                            if candidate in self.tokenizer.eos_token_ids:
                                output = output[:index + 1]
                                break
                        for slot in cache:
                            slot.trim(len(draft) + 1 - len(output))
                    tokens.extend(output)
                    remaining -= len(output)
                    token = output[-1]
                    yield from output
                return
            if settings.disk_embedding or settings.layer_chunk or settings.stream_weights:
                # These modes synchronize during each step. Yield this token
                # before evaluating any of the next step.
                for index in range(max_tokens):
                    yield current.item()
                    if index + 1 < max_tokens:
                        current = step(current)
                        mx.async_eval(current)
                return
            for index in range(max_tokens):
                if index + 1 < max_tokens:
                    following = step(current)
                    mx.async_eval(following)
                yield current.item()
                if index + 1 < max_tokens:
                    current = following

    def generate(
        self, messages: list[dict], *, max_tokens: int, temperature: float,
        thinking: bool, top_p: float = 1.0,
    ) -> Iterator[Event]:
        started = perf_counter()
        ids = self.prepare(messages, max_tokens, thinking)
        return self.generate_ids(
            ids, max_tokens=max_tokens, temperature=temperature, top_p=top_p, started=started,
        )

    def generate_ids(
        self, ids: list[int], *, max_tokens: int, temperature: float = 0,
        top_p: float = 1.0, reuse: bool = True, started: float | None = None,
    ) -> Iterator[Event]:
        from mlx_lm.sample_utils import make_sampler

        started = perf_counter() if started is None else started
        self._check_context(ids, max_tokens)
        if not isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Temperature must be nonnegative and top_p must be in (0, 1].")
        if self._busy:
            raise RuntimeError("Generation is already in progress.")
        self._busy = True
        mx = self.mx
        stock = self.settings.impl == "stock"
        stream = self.upstream.generation_stream if stock else self.stream
        # Use the same per-request residency policy as MLX-LM for a resident
        # model. A streamed model must be able to release its detached groups.
        wired = not self.settings.stream_weights
        wiring = self.upstream.wired_limit(self.model, [stream]) if wired else nullcontext()
        generated: list[int] = []
        first = last = None
        finish = "length"
        eos = self.tokenizer.eos_token_ids

        try:
            cache, reused = self._cache_for(ids, reuse)
            mx.reset_peak_memory()
            prompt = mx.array(ids[reused:], dtype=mx.uint32)
            sampler = make_sampler(temp=temperature, top_p=top_p)
            detokenizer = self.tokenizer.detokenizer
            detokenizer.reset()

            with wiring:
                if stock:
                    source = self.upstream.generate_step(
                        prompt, self.model, max_tokens=max_tokens, sampler=sampler,
                        prompt_cache=cache, prefill_step_size=self.settings.prefill,
                    )
                else:
                    source = self._fast_tokens(prompt, cache, max_tokens, temperature, sampler, ids)
                with closing(source):
                    for value in source:
                        token = value[0] if stock else value
                        last = perf_counter()
                        if first is None:
                            first = last
                        generated.append(token)
                        if token in eos:
                            finish = "stop"
                            break
                        detokenizer.add_token(token)
                        yield Event(detokenizer.last_segment, token=token)

            detokenizer.finalize()
        finally:
            try:
                # A closed iterator can still have a prefetched GPU token in flight.
                if not wired:
                    mx.synchronize(stream)
            finally:
                self._busy = False

        if reuse:
            self.cache = cache
            self.cache_tokens = (ids + generated)[:cache[0].offset]
        if first is None:
            raise RuntimeError("The model did not return a token.")
        stats = Stats(
            prompt_tokens=len(ids),
            generated_tokens=len(generated),
            cached_tokens=reused,
            ttft_s=first - started,
            decode_tps=(len(generated) - 1) / max(last - first, 1e-9),
            prefill_tps=(len(ids) - reused) / max(first - started, 1e-9),
            wall_s=perf_counter() - started,
            peak_mib=mx.get_peak_memory() / 2**20,
        )
        yield Event(
            detokenizer.last_segment, token=generated[-1] if finish == "stop" else None,
            finish_reason=finish, stats=stats,
        )
