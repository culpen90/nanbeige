"""A bounded weight working set for the pinned Nanbeige checkpoint."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.nanbeige import Attention, MLP, Model

from nanbeige_embedding import map_input_embedding


@dataclass(frozen=True)
class _ParameterSlot:
    owner: dict
    key: str
    checkpoint_name: str


def _slots(module, prefix, weights, expected_count):
    parameters = tree_flatten(module.parameters())
    if len(parameters) != expected_count:
        raise ValueError(f"Unexpected parameter count for {prefix}; streaming requires original weights.")
    slots = []
    for name, parameter in parameters:
        checkpoint_name = f"{prefix}.{name}"
        source = weights.get(checkpoint_name)
        if source is None or source.shape != parameter.shape or source.dtype != parameter.dtype:
            raise ValueError(f"Unexpected checkpoint layout for {checkpoint_name}.")
        owner = module
        *parents, key = name.split(".")
        for parent in parents:
            owner = owner[parent]
        slots.append(_ParameterSlot(owner, key, checkpoint_name))
    return slots


@contextmanager
def _attach(slots, weights):
    attached = []
    try:
        for slot in slots:
            # Pop the source so the lazy dictionary cannot keep a materialized
            # weight alive after the group leaves the model.
            slot.owner[slot.key] = weights.pop(slot.checkpoint_name)
            attached.append(slot)
        yield
    finally:
        for slot in attached:
            del slot.owner[slot.key]


class StreamedWeights:
    """Run the original layers in groups, reloading each pass from safetensors.

    Construct this from a fresh lazy model. It takes ownership of the model's
    streamed layer and head parameters, so subsequent forwards must go
    through this object. The full vocabulary projection is evaluated as a
    separate group. The first ``resident_layers`` layers can stay resident
    across both weight-sharing passes and subsequent calls.
    """

    max_prompt_chunk = 256

    def __init__(self, model, safetensors_path, *, layers_per_group=2, resident_layers=0):
        if not isinstance(model, Model) or model.args.tie_word_embeddings:
            raise ValueError("Streaming requires the untied Nanbeige MLX model.")
        args = model.args
        inner = model.model
        dimensions = (
            args.hidden_size, args.intermediate_size, args.num_attention_heads,
            args.num_key_value_heads, args.head_dim, args.vocab_size,
            len(inner.layers), inner.num_loops,
        )
        if dimensions != (3072, 10752, 48, 8, 128, 166144, 22, 2):
            raise ValueError("Streaming requires the Nanbeige4.2-3B architecture.")
        if type(layers_per_group) is not int or not 1 <= layers_per_group <= 4:
            raise ValueError("Use between one and four active layers for weight streaming.")
        if type(resident_layers) is not int or not 0 <= resident_layers <= 20:
            raise ValueError("Retain between zero and twenty layers for weight streaming.")
        if any(type(layer.self_attn) is not Attention or type(layer.mlp) is not MLP
               for layer in inner.layers):
            raise ValueError("Streaming requires the original Nanbeige projections.")

        self.model = model
        self.path = Path(safetensors_path)
        self.layers_per_group = layers_per_group
        self.resident_layers = resident_layers
        self._lock = Lock()

        # MLX validates the safetensors header without reading the tensor data.
        weights = self._source()
        self._layer_slots = [
            _slots(layer, f"model.layers.{index}", weights, 23)
            for index, layer in enumerate(inner.layers)
        ]
        self._head_slots = _slots(model.lm_head, "lm_head", weights, 3)
        map_input_embedding(model, self.path)
        for group in (*self._layer_slots[resident_layers:], self._head_slots):
            for slot in group:
                del slot.owner[slot.key]

    def _source(self):
        # The native loader creates a fresh lazy tensor per name. Reuse one
        # source within a pass and reopen for the second pass through the stack.
        return mx.load(self.path)

    def forward(self, x, cache, *, logits, last_only=False):
        """Update the two KV stacks and return logits when requested."""
        result, _ = self.forward_tapped(x, cache, logits=logits, last_only=last_only)
        return result

    def forward_tapped(self, x, cache, *, logits, last_only=False, tap=None):
        """Return logits and the concatenated residual states, when requested."""
        if x.ndim != 2 or x.shape[0] != 1 or not 1 <= x.shape[1] <= self.max_prompt_chunk:
            raise ValueError(f"Streaming expects one batch of at most {self.max_prompt_chunk} tokens.")
        with self._exclusive():
            return self._forward(x, cache, logits=logits, last_only=last_only, tap=tap)

    def project(self, hidden):
        """Project drafter states with the untied target vocabulary weights."""
        with self._exclusive():
            return self._project(hidden, self._source())

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Weight streaming already has a forward in progress.")
        try:
            yield
        finally:
            self._lock.release()

    def _forward(self, x, cache, *, logits, last_only, tap):
        inner = self.model.model
        count = len(inner.layers)
        depth = count * inner.num_loops
        if len(cache) != depth or any(c.offset != cache[0].offset for c in cache):
            raise ValueError("Weight streaming needs aligned KV offsets for all 44 layers.")
        # The drafter tap convention names layer k but captures the input to step k+1.
        # This includes the inter-loop norm before step 22 for the tap at layer 21.
        wanted = {index + 1 for index in tap} if tap else set()
        if any(not 1 <= index < depth for index in wanted):
            raise ValueError("A hidden-state tap must precede another Nanbeige layer.")
        captured = []
        hidden = inner.embed_tokens(x)
        mask = create_attention_mask(hidden, cache[0])

        for loop in range(inner.num_loops):
            weights = self._source()
            for start in range(0, count, self.layers_per_group):
                end = min(start + self.layers_per_group, count)
                stream_start = max(start, self.resident_layers)
                slots = [slot for group in self._layer_slots[stream_start:end] for slot in group]
                layer_caches = cache[loop * count + start:loop * count + end]
                with _attach(slots, weights):
                    for index, layer_cache in enumerate(layer_caches, start=start):
                        if loop * count + index in wanted:
                            captured.append(hidden)
                        hidden = inner.layers[index](hidden, mask, layer_cache)
                    states = [layer_cache.state for layer_cache in layer_caches]
                    if not logits and end == count and loop + 1 == inner.num_loops:
                        # The final prefill group only needs its KV updates.
                        del hidden
                        mx.eval(states, captured)
                        fused = mx.concatenate(captured, axis=-1) if captured else None
                        return None, fused
                    mx.eval(hidden, states)
            if not inner.skip_loop_final_norm:
                hidden = inner.norm(hidden)

        if inner.skip_loop_final_norm:
            hidden = inner.norm(hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        fused = mx.concatenate(captured, axis=-1) if captured else None
        return self._project(hidden, weights), fused

    def _project(self, hidden, weights):
        with _attach(self._head_slots, weights):
            result = self.model.lm_head(hidden)
            mx.eval(result)
        return result
