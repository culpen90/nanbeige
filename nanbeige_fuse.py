"""Pack Nanbeige's independent projections without requantizing weights."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.nanbeige import Attention, MLP, Model


class _PackedLinear(nn.QuantizedLinear):
    def __init__(self, *projections):
        # The source weights are already packed. Skip random initialization.
        nn.Module.__init__(self)
        self.group_size, self.bits, self.mode = 64, 6, "affine"
        self.weight = mx.concatenate([p.weight for p in projections], axis=0)
        self.scales = mx.concatenate([p.scales for p in projections], axis=0)
        self.biases = mx.concatenate([p.biases for p in projections], axis=0)
        # Evaluate one group before replacing its source modules so the lazy
        # concatenation cannot retain source matrices across all 22 layers.
        mx.eval(self.weight, self.scales, self.biases)
        self.freeze()


class _PackedAttention(Attention):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.n_heads = original.n_heads
        self.n_kv_heads = original.n_kv_heads
        self.head_dim = original.head_dim
        self.scale = original.scale
        self.rope = original.rope
        self.o_proj = original.o_proj
        self.qkv_proj = _PackedLinear(original.q_proj, original.k_proj, original.v_proj)
        query_width = self.n_heads * self.head_dim
        self.split_indices = (query_width, query_width + self.n_kv_heads * self.head_dim)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        queries, keys, values = mx.split(self.qkv_proj(x), self.split_indices, axis=-1)
        queries = queries.reshape(batch, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class _PackedMLP(MLP):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.down_proj = original.down_proj
        self.gate_up_proj = _PackedLinear(original.gate_proj, original.up_proj)

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


def _validate_projection(projection, output_width, input_width):
    if (
        not isinstance(projection, nn.QuantizedLinear)
        or (projection.group_size, projection.bits, projection.mode) != (64, 6, "affine")
        or "bias" in projection
        or projection.weight.dtype != mx.uint32
        or projection.weight.shape != (output_width, input_width * 6 // 32)
        or projection.scales.dtype != mx.bfloat16
        or projection.scales.shape != (output_width, input_width // 64)
        or projection.biases is None
        or projection.biases.dtype != projection.scales.dtype
        or projection.biases.shape != projection.scales.shape
    ):
        raise ValueError("Packed projections require Nanbeige's affine 6-bit BF16 weights")


def fuse_projections(model):
    """Replace the five input projections per layer with two packed matmuls."""
    if not isinstance(model, Model):
        raise ValueError("Packed projections require the Nanbeige MLX model")
    args = model.args
    dimensions = (
        args.hidden_size, args.intermediate_size, args.num_hidden_layers,
        args.num_attention_heads, args.num_key_value_heads, args.head_dim,
        args.effective_num_loops,
    )
    if dimensions != (3072, 10752, 22, 48, 8, 128, 2):
        raise ValueError("Packed projections require the Nanbeige4.2-3B dimensions")

    # Validate every source before mutating the first layer.
    query_width = args.num_attention_heads * args.head_dim
    kv_width = args.num_key_value_heads * args.head_dim
    for layer in model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if not isinstance(attn, Attention) or not isinstance(mlp, MLP):
            raise ValueError("Packed projections require the Nanbeige attention and MLP modules")
        if not isinstance(attn, _PackedAttention):
            _validate_projection(attn.q_proj, query_width, args.hidden_size)
            _validate_projection(attn.k_proj, kv_width, args.hidden_size)
            _validate_projection(attn.v_proj, kv_width, args.hidden_size)
        if not isinstance(mlp, _PackedMLP):
            _validate_projection(mlp.gate_proj, args.intermediate_size, args.hidden_size)
            _validate_projection(mlp.up_proj, args.intermediate_size, args.hidden_size)

    for layer in model.layers:
        if not isinstance(layer.self_attn, _PackedAttention):
            layer.self_attn = _PackedAttention(layer.self_attn)
        if not isinstance(layer.mlp, _PackedMLP):
            layer.mlp = _PackedMLP(layer.mlp)
    model.eval()
