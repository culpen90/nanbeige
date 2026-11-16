"""Read Nanbeige's untied input embedding from the pinned safetensors file."""

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.nanbeige import Model


_VOCAB_SIZE = 166144
_HIDDEN_SIZE = 3072
_GROUP_SIZE = 64
_QUANT_BITS = 6
_QUANT_COLS = _HIDDEN_SIZE * _QUANT_BITS // 32
_GROUP_COLS = _HIDDEN_SIZE // _GROUP_SIZE
_EMBEDDING = "model.embed_tokens"


def _map_rows(path: Path):
    with path.open("rb") as file:
        size = os.fstat(file.fileno()).st_size
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors file.")
        header_size, = struct.unpack("<Q", prefix)
        if not 2 <= header_size <= min(1024 * 1024, size - 8):
            raise ValueError("Unexpected safetensors header size.")
        data_start = 8 + header_size
        header = json.loads(file.read(header_size))
        if not isinstance(header, dict):
            raise ValueError("Unexpected safetensors header.")

        maps = []
        spans = []
        for name, format_name, dtype, columns in (
            ("weight", "U32", np.dtype("<u4"), _QUANT_COLS),
            ("scales", "BF16", np.dtype("<u2"), _GROUP_COLS),
            ("biases", "BF16", np.dtype("<u2"), _GROUP_COLS),
        ):
            entry = header.get(f"{_EMBEDDING}.{name}")
            shape = (_VOCAB_SIZE, columns)
            if (not isinstance(entry, dict) or entry.get("dtype") != format_name
                    or entry.get("shape") != list(shape)):
                raise ValueError(f"Unexpected safetensors input embedding {name}.")
            offsets = entry.get("data_offsets")
            nbytes = _VOCAB_SIZE * columns * dtype.itemsize
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)
                    or not 0 <= offsets[0] < offsets[1] <= size - data_start
                    or offsets[1] - offsets[0] != nbytes):
                raise ValueError(f"Invalid safetensors offsets for input embedding {name}.")
            start, end = offsets
            if any(start < previous_end and end > previous_start
                   for previous_start, previous_end in spans):
                raise ValueError("Overlapping safetensors input embedding tensors.")
            spans.append((start, end))
            maps.append(np.memmap(file, dtype=dtype, mode="r", offset=data_start + start, shape=shape))
        return maps


class _MappedInputEmbedding(nn.Module):
    def __init__(self, maps):
        super().__init__()
        # NumPy arrays stay out of the MLX parameter tree and its eager eval.
        self._weight, self._scales, self._biases = maps

    def __call__(self, x):
        # Reading generated IDs synchronizes their producer before gathering
        # the rows. The page cache can reclaim the file backing under pressure.
        indices = np.asarray(x)
        return mx.dequantize(
            mx.array(self._weight[indices]),
            scales=mx.array(self._scales[indices]).view(mx.bfloat16),
            biases=mx.array(self._biases[indices]).view(mx.bfloat16),
            group_size=_GROUP_SIZE,
            bits=_QUANT_BITS,
            mode="affine",
        )


def map_input_embedding(model, safetensors_path):
    """Replace just the checkpoint's input embedding before evaluating the model."""
    if not isinstance(model, Model) or model.args.tie_word_embeddings:
        raise ValueError("A mapped input embedding requires the untied Nanbeige MLX model.")

    embedding = model.model.embed_tokens
    if isinstance(embedding, _MappedInputEmbedding):
        return
    if (
        not isinstance(embedding, nn.QuantizedEmbedding)
        or (model.args.vocab_size, model.args.hidden_size, model.args.effective_num_loops,
            model.args.num_hidden_layers) != (_VOCAB_SIZE, _HIDDEN_SIZE, 2, 22)
        or (embedding.group_size, embedding.bits, embedding.mode) != (_GROUP_SIZE, _QUANT_BITS, "affine")
        or embedding.weight.shape != (_VOCAB_SIZE, _QUANT_COLS)
        or embedding.weight.dtype != mx.uint32
        or embedding.scales.shape != (_VOCAB_SIZE, _GROUP_COLS)
        or embedding.scales.dtype != mx.bfloat16
        or embedding.biases is None
        or embedding.biases.shape != embedding.scales.shape
        or embedding.biases.dtype != mx.bfloat16
    ):
        raise ValueError("A mapped input embedding requires Nanbeige4.2-3B's affine 6-bit layout.")

    mapped = _MappedInputEmbedding(_map_rows(Path(safetensors_path)))
    mapped.eval()
    model.model.embed_tokens = mapped
