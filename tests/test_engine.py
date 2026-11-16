"""Check token and cache behavior on a small two-pass Nanbeige model."""

from importlib import import_module
from types import SimpleNamespace
import unittest

import mlx.core as mx
from mlx_lm.models.nanbeige import Model, ModelArgs

from nanbeige_engine import Engine, Settings


class _Detokenizer:
    last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f" {token}"

    def finalize(self):
        self.last_segment = ""


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.runner = Engine.__new__(Engine)
        self.runner.mx = mx
        self.runner.upstream = import_module("mlx_lm.generate")
        self.runner.stream = mx.new_stream(mx.cpu)
        self.runner.settings = Settings(context=64, prefill=4)
        self.runner._busy = False
        self.runner.reset()
        self.runner.tokenizer = SimpleNamespace(eos_token_ids=set(), detokenizer=_Detokenizer())
        with mx.stream(self.runner.stream):
            mx.random.seed(42)
            self.runner.model = Model(ModelArgs(
                model_type="nanbeige", hidden_size=64, num_hidden_layers=2,
                intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
                rms_norm_eps=1e-5, vocab_size=128, head_dim=32, num_loops=2,
            ))
            mx.eval(self.runner.model.parameters())

    def output(self, ids, reuse=True):
        events = list(self.runner.generate_ids(ids, max_tokens=5, reuse=reuse))
        return [event.token for event in events[:-1]], events[-1].stats

    def test_prefix_is_trimmed_across_both_loops(self):
        self.output([15, 18, 27, 11, 51, 99, 20])
        updated = [15, 18, 27, 11, 62, 101, 90]
        warm, stats = self.output(updated)
        self.assertEqual(stats.cached_tokens, 4)
        self.assertEqual([slot.offset for slot in self.runner.cache], [11] * 4)
        cold, stats = self.output(updated, reuse=False)
        self.assertEqual(stats.cached_tokens, 0)
        self.assertEqual(warm, cold)

    def test_layer_staging_keeps_greedy_tokens(self):
        ids = [62, 27, 18, 10, 53, 23, 77, 40]
        reference, _ = self.output(ids, reuse=False)
        self.runner.configure(Settings(context=64, prefill=4, layer_chunk=1))
        staged, _ = self.output(ids, reuse=False)
        self.assertEqual(staged, reference)

    def test_cancelled_iterator_does_not_leave_a_live_cache(self):
        self.output([5, 6, 7, 8])
        stream = self.runner.generate_ids([5, 6, 7, 9], max_tokens=5)
        self.assertIsNotNone(next(stream).token)
        stream.close()
        self.assertIsNone(self.runner.cache)
        self.assertEqual(self.output([5, 6, 7, 9])[1].cached_tokens, 0)

    def test_lookup_trims_verified_cache_at_eos(self):
        class Cache:
            def __init__(self):
                self.offset = 0

            def trim(self, count):
                self.offset -= count

            @property
            def state(self):
                return self.offset

        class PredictNext:
            def __init__(self):
                self.model = self
                self.widths = []

            def make_cache(self):
                return [Cache(), Cache()]

            def __call__(self, x, *, cache):
                self.widths.append(x.shape[-1])
                for slot in cache:
                    slot.offset += x.shape[-1]
                return (mx.arange(16) == (x[..., None] + 1) % 16).astype(mx.float32)

        self.runner.model = PredictNext()
        self.runner.tokenizer.eos_token_ids.add(6)
        self.runner.configure(Settings(context=64, prefill=4, lookup=4))
        prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        events = list(self.runner.generate_ids(prompt, max_tokens=8))
        self.assertEqual([event.token for event in events], [4, 5, 6])
        self.assertEqual(events[-1].finish_reason, "stop")
        self.assertGreater(self.runner.model.widths[-1], 3)
        self.assertEqual([slot.offset for slot in self.runner.cache], [len(prompt) + 2] * 2)


if __name__ == "__main__":
    unittest.main()
