"""Threshold-loader fallback tests for backend.config (stdlib unittest).

backend/config.py loads <DATA_DIR>/stage1_threshold.json at import time and
must NEVER raise: a missing file, corrupt JSON, or a non-numeric "threshold"
all fall back to 0.35 (the documented default). The loader is exercised
directly with config.DATA_DIR patched to a throwaway temp directory, so the
repo's data/ and training/ directories are never read or written.

Also guards item 4 (the dead duplicate DEFAULT_ANOMALY_THRESHOLD assignment
was removed): consumers must observe the single, final loader-backed value.

Run from the repo root:
    python -m unittest tests.test_config_paths -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend import config  # noqa: E402

FALLBACK = 0.35


class ThresholdLoaderFallbackTests(unittest.TestCase):
    """_load_calibrated_threshold: every failure mode -> the 0.35 fallback."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lg_threshold_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "stage1_threshold.json")

    def _load(self):
        """Call the loader with DATA_DIR pinned at this test's temp dir."""
        with mock.patch.object(config, "DATA_DIR", Path(self.dir)):
            return config._load_calibrated_threshold()

    def _write(self, text: str) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _load_quietly(self):
        """Like _load(), silencing the loader's [WARN] line on stderr."""
        with contextlib.redirect_stderr(io.StringIO()):
            return self._load()

    # -- failure modes ------------------------------------------------
    def test_missing_file_falls_back(self):
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._load(), FALLBACK)

    def test_corrupt_json_falls_back(self):
        self._write("{this is not json")
        self.assertEqual(self._load_quietly(), FALLBACK)

    def test_non_numeric_threshold_falls_back(self):
        self._write(json.dumps({"threshold": "very high"}))
        self.assertEqual(self._load_quietly(), FALLBACK)

    def test_missing_threshold_key_falls_back(self):
        self._write(json.dumps({"value": 0.42}))
        self.assertEqual(self._load_quietly(), FALLBACK)

    def test_unreadable_file_falls_back(self):
        # A directory where the JSON file should be -> is_file() is False.
        os.mkdir(self.path)
        self.assertEqual(self._load(), FALLBACK)

    # -- happy path ---------------------------------------------------
    def test_valid_file_returns_numeric_threshold(self):
        self._write(json.dumps({"threshold": 0.42}))
        self.assertEqual(self._load(), 0.42)

    def test_numeric_string_threshold_is_coerced(self):
        self._write(json.dumps({"threshold": "0.42"}))
        self.assertEqual(self._load(), 0.42)

    # -- module-level wiring (guards the deleted duplicate constant) --
    def test_fallback_default_parameter_is_0_35(self):
        self.assertEqual(config._load_calibrated_threshold.__defaults__, (FALLBACK,))

    def test_module_constant_matches_loader_result(self):
        # DEFAULT_ANOMALY_THRESHOLD must be the FINAL loader-backed value
        # (backend/edge_infer.py and backend/run_edge.py both read it).
        self.assertIsInstance(config.DEFAULT_ANOMALY_THRESHOLD, float)
        self.assertEqual(
            config.DEFAULT_ANOMALY_THRESHOLD, config._load_calibrated_threshold()
        )


if __name__ == "__main__":
    unittest.main()
