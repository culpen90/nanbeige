import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from nanbeige_api_key import get_api_key


class APIKeyTests(unittest.TestCase):
    def setUp(self):
        self.config = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.config)}))
        self.key_file = self.config / "nanbeige" / "api-key"

    def test_cli_creates_and_prints_the_persisted_key(self):
        cli = Path(__file__).resolve().parents[1] / "nanbeige_cli.py"
        result = subprocess.run(
            [sys.executable, str(cli), "api-key"], capture_output=True, text=True, check=True,
        )
        key = get_api_key()
        self.assertRegex(key, r"\Ask-[0-9a-f]{64}\Z")
        self.assertEqual(result.stdout, key + "\n")
        self.assertEqual(self.key_file.read_text().strip(), key)
        self.assertEqual(stat.S_IMODE(self.key_file.stat().st_mode), 0o600)
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.config / "other")}):
            self.assertNotEqual(get_api_key(), key)

    def test_concurrent_first_use_returns_one_key(self):
        start = Barrier(8, timeout=5)

        def create_key():
            start.wait()
            return get_api_key()

        with ThreadPoolExecutor(max_workers=start.parties) as workers:
            requests = [workers.submit(create_key) for _ in range(start.parties)]
        self.assertEqual({request.result() for request in requests}, {get_api_key()})

    def test_malformed_key_is_rejected_and_preserved(self):
        get_api_key()
        self.key_file.write_text("sk-truncated")
        with self.assertRaises(ValueError):
            get_api_key()
        self.assertEqual(self.key_file.read_text(), "sk-truncated")
