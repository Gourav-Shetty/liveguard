"""C. Pipeline regression (test 11): run_edge MOCK / --no-ws keeps streaming.

The child is waited on until its FIRST `Beat #` appears (up to
FIRST_BEAT_DEADLINE seconds of startup allowance), then given a full
RUN_SECONDS of streaming time before the test asserts it is still alive
and that stdout contains `Beat #` lines with `[NORMAL BEAT]`. Gating the
window on the first beat (instead of wall-clock from spawn) keeps numpy/
scipy import time under load from starving the window. LIVEGUARD_DATA_DIR
points the child at a temp directory so the repo's data/ directory is
never touched.

Run from the repo root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SECONDS = 6.0  # streaming time required AFTER the first beat
FIRST_BEAT_DEADLINE = 20.0  # startup allowance for imports under load


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp(prefix="liveguard_pipe_")

    def tearDown(self):
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_11_run_edge_mock_still_streams_beats(self):
        env = dict(os.environ)
        env["LIVEGUARD_DATA_DIR"] = self.data_dir
        proc = subprocess.Popen(
            [sys.executable, "-m", "backend.run_edge",
             "--source", "MOCK", "--no-ws"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: list[str] = []

        def reader():
            for line in proc.stdout:
                lines.append(line)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        start = time.monotonic()
        # Phase 1: wait for the first beat (bounded by the startup
        # deadline), breaking early if the child dies on its own.
        first_beat_at = None
        while time.monotonic() - start < FIRST_BEAT_DEADLINE:
            if proc.poll() is not None:
                break
            if any("Beat #" in line for line in list(lines)):
                first_beat_at = time.monotonic()
                break
            time.sleep(0.05)
        # Phase 2: stream for a full RUN_SECONDS from the first beat.
        if first_beat_at is not None:
            while time.monotonic() - first_beat_at < RUN_SECONDS:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
        alive_secs = time.monotonic() - start
        still_running = proc.poll() is None

        # stop the child (never leave it running)
        if still_running:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        thread.join(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        output = "".join(lines)

        self.assertTrue(
            still_running,
            "pipeline exited on its own (rc=%s) before %.1fs of streaming; "
            "first beat after %ss; output:\n%s"
            % (proc.returncode, RUN_SECONDS,
               "never" if first_beat_at is None
               else "%.1f" % (first_beat_at - start), output),
        )
        self.assertGreaterEqual(alive_secs, RUN_SECONDS)
        self.assertNotIn("[ERROR]", output, "pipeline logged errors:\n%s" % output)
        self.assertIn("Beat #", output, "no beats printed:\n%s" % output)
        self.assertIn("[NORMAL BEAT]", output, "no normal beats:\n%s" % output)
        # at least a couple of beats so we know it kept streaming
        self.assertGreaterEqual(output.count("Beat #"), 2, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
