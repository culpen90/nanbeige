"""Optional speculation against the same loaded 6-bit Nanbeige target."""

from pathlib import Path
from threading import Lock
from time import perf_counter


DRAFTER_ID = "Nanbeige/Nanbeige4.2-3B-DSpark"
DRAFTER_REVISION = "5f12c792dabbfcdc4a0cf504e75f7216706d9590"
DRAFTER_FILES = ("config.json", "model.safetensors")
_GENERATION_LOCK = Lock()


def fetch_drafter(*, local_only=False):
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        DRAFTER_ID,
        revision=DRAFTER_REVISION,
        allow_patterns=DRAFTER_FILES,
        local_files_only=local_only,
    ))
    if any(not (snapshot / name).is_file() for name in DRAFTER_FILES):
        raise ValueError("The official DSpark snapshot is incomplete. Fetch it before running offline.")
    return str(snapshot)


def load_compact_drafter(bits, *, local_only=True):
    """Pack the drafter before the target is loaded to avoid overlapping peaks."""
    import mlx.core as mx
    from mlx_dspark.load import load_drafter

    previous = mx.set_cache_limit(32 * 2**20)
    try:
        drafter, _ = load_drafter(fetch_drafter(local_only=local_only), bits=bits)
        return drafter
    finally:
        mx.clear_cache()
        mx.set_cache_limit(previous)


class _DraftBudget:
    """Reserve one position for the target's bonus token."""

    probe_every = 1

    def __init__(self, cap, max_position):
        self.max_cap = self.cap = cap
        self.max_position = max_position

    def set_depth(self, depth):
        self.cap = max(0, min(self.max_cap, self.max_position - depth - 1))

    def update(self, *args, **kwargs):
        pass


def _streamed_target(target_type, streamed, tokenizer, *, kv_bits):
    class StreamedTarget(target_type):
        def prefill(self, ids, cache, tap=None, *, want_logits=False, head_last_row=True):
            return streamed.forward_tapped(ids, cache, logits=want_logits, last_only=head_last_row, tap=tap)

        def plain(self, ids, cache):
            return streamed.forward(ids, cache, logits=True)

        def run(self, ids, cache, tap):
            return streamed.forward_tapped(ids, cache, logits=True, tap=tap)

        def lm_head_proj(self, hidden):
            return streamed.project(hidden)

    return StreamedTarget(streamed.model, tokenizer, kv_bits=kv_bits)


class DSparkRunner:
    """Run one target, using the model's existing MLX weights and tokenizer."""

    def __init__(
        self, model, tokenizer, *, mode="dspark", context=4096, prefill=512,
        draft_cap=3, prefix_slots=1, kv_bits=None, local_only=False, streamed=None, draft_bits=4,
        preloaded_drafter=None,
    ):
        try:
            import mlx_dspark.generate as generation
            import mlx_dspark.lookup as lookup
            from mlx_dspark.prefix_cache import PrefixCache
            from mlx_dspark.target import Target
        except ModuleNotFoundError as exc:
            if exc.name != "mlx_dspark":
                raise
            raise RuntimeError("Use ./nanbeige spec or uv run --extra dspark.") from exc

        if getattr(model, "model_type", None) != "nanbeige":
            raise ValueError("This decoder requires the loaded Nanbeige4.2 model")
        if mode not in {"dspark", "lookup", "baseline"}:
            raise ValueError("Speculation mode must be dspark, lookup, or baseline")
        if min(context, prefill) < 1 or not 1 <= draft_cap <= 7 or prefix_slots < 0:
            raise ValueError("Invalid context, prefill, draft cap, or prefix slot count")
        if draft_bits not in (2, 4):
            raise ValueError("The draft model supports 2-bit or 4-bit weights")
        if streamed is not None and streamed.model is not model:
            raise ValueError("The streamed weights must belong to the loaded Nanbeige model")
        self.mode = mode
        self.context = context
        self.prefill = min(prefill, streamed.max_prompt_chunk) if streamed is not None else prefill
        self.draft_cap = draft_cap
        self.generation = generation
        self.lookup = lookup
        self.tokenizer = tokenizer
        self.target = (Target(model, tokenizer, kv_bits=kv_bits) if streamed is None
                       else _streamed_target(Target, streamed, tokenizer, kv_bits=kv_bits))
        self.drafter = None
        if mode == "dspark":
            # The upstream self-test directly invokes the resident model's layers.
            if streamed is None:
                self.target.verify_tap()
            self.drafter = (preloaded_drafter if preloaded_drafter is not None
                            else load_compact_drafter(draft_bits, local_only=local_only))
            self.drafter.ctx_window = min(context, 4096)
        self.prefix = None
        if prefix_slots:
            make_ctx = self.drafter.make_ctx_cache if self.drafter is not None else None
            self.prefix = PrefixCache(self.target.make_cache, make_ctx, slots=prefix_slots)

    def clear(self):
        with _GENERATION_LOCK:
            if self.prefix is not None:
                self.prefix.reset()

    def generate(
        self, prompt_ids, *, max_tokens, temperature=0.0, top_p=1.0,
        top_k=0, seed=None, stop=None, on_text=None, use_cache=True,
    ):
        if not prompt_ids or max_tokens < 1 or len(prompt_ids) + max_tokens > self.context:
            raise ValueError(f"Prompt and completion must fit the {self.context}-token context")

        with _GENERATION_LOCK:
            return self._generate(
                prompt_ids, max_tokens, temperature, top_p, top_k, seed, stop, on_text, use_cache,
            )

    def _generate(self, prompt_ids, max_tokens, temperature, top_p, top_k, seed, stop, on_text, use_cache):
        prefix = self.prefix if use_cache else None
        cache, ctx, reuse = prefix.acquire(prompt_ids) if prefix is not None else (None, None, 0)
        started = perf_counter()
        first_text = None

        def emit(chunk):
            nonlocal first_text
            if first_text is None:
                first_text = perf_counter() - started
            if on_text is not None:
                on_text(chunk)

        options = dict(
            prompt_ids=prompt_ids, cache=cache, reuse_len=reuse, max_new_tokens=max_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, seed=seed, stop=stop, on_text=emit,
        )
        old_prefill = self.generation.PREFILL_CHUNK
        original_index = self.lookup.NGramIndex
        max_position = len(prompt_ids) + max_tokens

        # mlx-dspark 0.20.4 can emit a bonus past max_new_tokens on a fully accepted
        # lookup or drafter round. Clamp before verification so streaming obeys the limit.
        class BoundedIndex(original_index):
            def propose(self, long_draft=None):
                remaining = max_position - len(self.tokens) - 1
                return super().propose(long_draft)[:remaining] if remaining > 0 else []

        self.generation.PREFILL_CHUNK = self.prefill
        self.lookup.NGramIndex = BoundedIndex
        try:
            if self.mode == "dspark":
                budget = _DraftBudget(self.draft_cap, max_position)
                result = self.generation.speculative_generate(
                    self.target, self.tokenizer, self.drafter, ctx_caches=ctx,
                    max_draft_tokens=self.draft_cap, cap_controller=budget,
                    lookup_max_draft=self.draft_cap, lookup_long_draft=self.draft_cap, **options,
                )
            elif self.mode == "lookup":
                result = self.lookup.lookup_generate(
                    self.target, self.tokenizer, max_draft_tokens=self.draft_cap,
                    long_draft_tokens=self.draft_cap, **options,
                )
            else:
                result = self.generation.greedy_generate(self.target, self.tokenizer, **options)
            if prefix is not None:
                prefix.store(cache, ctx, prompt_ids, result.token_ids)
        except BaseException:
            if prefix is not None:
                prefix.reset()
            raise
        finally:
            self.generation.PREFILL_CHUNK = old_prefill
            self.lookup.NGramIndex = original_index
        result.prompt_tokens = len(prompt_ids)
        result.reused_tokens = reuse
        result.ttft_seconds = first_text or 0.0
        return result
