#!/usr/bin/env python
"""
Integration tests for concat_clips.py

Tests motion-aware pair matching concepts, clip ordering, and end-to-end
concatenation integrity using real test footage.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Try to import OpenCV
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Import functions from concat_clips module
if HAS_OPENCV:
    from concat_clips.concat_clips import (
        compute_frame_difference,
        find_best_seam,
    )


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestMotionAwarePairMatching(unittest.TestCase):
    """Test the concept of motion-aware pair matching."""

    def test_pair_matching_concept(self):
        """
        Test that pair matching can distinguish motion direction.

        Scenario:
        - Needle pair: frames showing object moving right (left position, then right position)
        - Haystack pair A: object moving right (same as needle)
        - Haystack pair B: object moving left (opposite of needle)

        Pair matching should prefer A over B because it matches both frames,
        whereas single-frame matching might incorrectly prefer B if the final
        positions happen to match better.
        """
        # Create frames with a "moving object" (a white square)
        def create_frame(object_x):
            """Create a frame with a white square at position object_x."""
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, object_x:object_x+20] = 255
            return frame

        # Needle pair: object at x=10, then x=30 (moving right)
        needle1 = create_frame(10)
        needle2 = create_frame(30)

        # Haystack pair A: object at x=10, then x=30 (moving right - matches needle)
        haystack_a1 = create_frame(10)
        haystack_a2 = create_frame(30)

        # Haystack pair B: object at x=50, then x=30 (moving left - opposite direction)
        # Note: haystack_b2 matches needle2 position exactly
        haystack_b1 = create_frame(50)
        haystack_b2 = create_frame(30)

        # Compute combined MSE for pair A
        mse_a1 = compute_frame_difference(needle1, haystack_a1)
        mse_a2 = compute_frame_difference(needle2, haystack_a2)
        combined_mse_a = mse_a1 + mse_a2

        # Compute combined MSE for pair B
        mse_b1 = compute_frame_difference(needle1, haystack_b1)
        mse_b2 = compute_frame_difference(needle2, haystack_b2)
        combined_mse_b = mse_b1 + mse_b2

        # Pair A should have lower combined MSE (perfect match)
        self.assertEqual(combined_mse_a, 0.0)  # Exact match
        self.assertGreater(combined_mse_b, 0.0)  # Not a match for first frame
        self.assertLess(combined_mse_a, combined_mse_b)

        # Note: If we only compared needle2 to haystack frames:
        # - Single frame matching would give mse_a2 = 0 and mse_b2 = 0
        # - It couldn't distinguish the motion direction!
        # But pair matching correctly identifies A as the better match.

    def test_pair_matching_prefers_direction(self):
        """
        More realistic test: pair matching should prefer matching motion direction
        even when both final frames are imperfect matches.
        """
        def create_frame(object_x):
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, object_x:object_x+20] = 255
            return frame

        # Needle pair: moving right
        needle1 = create_frame(20)  # Object at x=20
        needle2 = create_frame(40)  # Object at x=40

        # Haystack pair A: moving right (same direction, different speed)
        haystack_a1 = create_frame(22)  # Close to needle1
        haystack_a2 = create_frame(42)  # Close to needle2

        # Haystack pair B: moving left (opposite direction)
        # Final frame matches well, but first frame is far off
        haystack_b1 = create_frame(60)  # Far from needle1
        haystack_b2 = create_frame(41)  # Very close to needle2

        # Pair A: small difference on both frames
        mse_a = compute_frame_difference(needle1, haystack_a1) + compute_frame_difference(needle2, haystack_a2)

        # Pair B: large difference on first frame, small on second
        mse_b = compute_frame_difference(needle1, haystack_b1) + compute_frame_difference(needle2, haystack_b2)

        # Pair A should win because it matches the motion direction
        self.assertLess(mse_a, mse_b)


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestFindBestSeamScoring(unittest.TestCase):
    """Test that find_best_seam correctly weights similarity, direction, and velocity."""

    def _make_frames(self, positions):
        """Create (timestamp, grayscale_frame) pairs with a white square at each x position."""
        frames = []
        for t, x in positions:
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, x:x + 20] = 255
            frames.append((t, frame))
        return frames

    def test_prefers_same_direction_over_opposite(self):
        """
        find_best_seam should prefer a seam where motion continues in the same direction.

        Both candidates have the same similarity score (identical junction frames),
        but candidate A continues moving right while candidate B reverses to the left.
        The seam with matching direction (A) should have a better (lower) score.
        """
        # Preceding clip tail: object moving right, ending at x=40.
        # Pair: (x=20 → x=40) — rightward motion ending at the seam.
        preceding = self._make_frames([(8.0, 20), (8.033, 40)])

        # Successor candidate A: continues rightward (x=40 → x=60).
        # b_curr (x=40) matches a_curr (x=40) perfectly.
        successor_same_dir = self._make_frames([(0.0, 40), (0.033, 60)])

        # Successor candidate B: reverses leftward (x=40 → x=20).
        # b_curr (x=40) also matches a_curr (x=40) perfectly — same similarity.
        successor_opp_dir = self._make_frames([(0.0, 40), (0.033, 20)])

        _, _, score_same = find_best_seam(preceding, successor_same_dir)
        _, _, score_opp = find_best_seam(preceding, successor_opp_dir)

        self.assertLess(score_same, score_opp,
            "Seam with same motion direction should have a better (lower) score")

    def test_prefers_high_velocity_when_similarity_is_equal(self):
        """
        find_best_seam should prefer positions with faster motion when frames are equally similar.

        Both candidates have the same similarity (identical junction frame content),
        but candidate A has a high-velocity preceding pair while candidate B is nearly static.
        The high-velocity seam (A) should have a better (lower) score.
        """
        # Two separate preceding clips: one with fast motion, one nearly still.
        # Both end on the same frame (x=40), giving equal similarity.
        preceding_fast = self._make_frames([(8.0, 10), (8.033, 40)])   # large displacement
        preceding_slow = self._make_frames([(8.0, 39), (8.033, 40)])   # tiny displacement

        # A shared successor: also moves rightward from x=40.
        successor = self._make_frames([(0.0, 40), (0.033, 60)])

        _, _, score_fast = find_best_seam(preceding_fast, successor)
        _, _, score_slow = find_best_seam(preceding_slow, successor)

        self.assertLess(score_fast, score_slow,
            "Seam during fast motion should have a better (lower) score than a near-static seam")


class FootageTestMixin:
    """Shared setup for tests that run the real script against real footage."""

    SCRIPT_DIR = Path(__file__).resolve().parent
    TEST_FOOTAGE_DIR = SCRIPT_DIR / "test_footage"

    @classmethod
    def _has_ffprobe(cls) -> bool:
        return shutil.which("ffprobe") is not None

    @classmethod
    def _has_ffmpeg(cls) -> bool:
        return shutil.which("ffmpeg") is not None

    def _skip_if_missing(self):
        if not self.TEST_FOOTAGE_DIR.exists():
            self.skipTest(f"Test footage not found at {self.TEST_FOOTAGE_DIR}")
        mp4s = list(self.TEST_FOOTAGE_DIR.glob("*.mp4"))
        if len(mp4s) < 2:
            self.skipTest("Need at least 2 mp4 files in test_footage/")

        # Verify files are real video (not LFS pointers)
        first = mp4s[0]
        with open(first, "rb") as f:
            header = f.read(16)
        if b"git-lfs" in header or b"version https://git-lfs" in header:
            self.skipTest("Test footage files are LFS pointers (not checked out)")

        if not self._has_ffmpeg():
            self.skipTest("ffmpeg not found on PATH")
        if not self._has_ffprobe():
            self.skipTest("ffprobe not found on PATH")

    # Entries in the chosen-order block look like:
    #   "1. name.mp4"
    #   "2. name.mp4 (score=1.23)"
    #   "1. name.mp4 intensity=4.56"
    #   "2. name.mp4 (score=1.23, intensity=4.56)"
    _ORDER_ENTRY = re.compile(
        r"^(?P<position>\d+)\. (?P<name>.+?)"
        r"(?: \(score=[^)]*\))?"
        r"(?: intensity=(?P<intensity>[\d.]+))?$"
    )

    @classmethod
    def _parse_chosen_order(cls, stdout, with_intensity=False):
        """Pull the clips out of the '[sort] Order chosen' block.

        The expected position is tracked so that any following numbered list in
        the log cannot bleed into the parsed order.

        Returns a list of names, or of (name, intensity) pairs when
        with_intensity is set (intensity is None if the line carried none).
        """
        entries = []
        collecting = False
        for line in stdout.splitlines():
            if line.startswith("[sort] Order chosen"):
                collecting = True
                continue
            if not collecting:
                continue
            match = cls._ORDER_ENTRY.match(line.strip())
            if not match or int(match.group("position")) != len(entries) + 1:
                break
            raw = match.group("intensity")
            intensity = float(raw) if raw is not None else None
            entries.append((match.group("name"), intensity))

        if with_intensity:
            return entries
        return [name for name, _ in entries]


class TestConcatenationIntegrity(FootageTestMixin, unittest.TestCase):
    """Integration test: concatenate test footage and verify stream integrity with ffprobe."""

    def test_concatenated_output_has_no_stream_errors(self):
        """Run concat_clips on test footage and verify ffprobe reports no warnings."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            # Run the script
            env = {**subprocess.os.environ, "PYTHONIOENCODING": "utf-8"}
            cmd = [
                sys.executable,
                str(self.SCRIPT_DIR / "concat_clips.py"),
                str(self.TEST_FOOTAGE_DIR),
                str(output_path),
                "--shuffle",
                "--match-seams",
                "--seed", "1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}")
            self.assertTrue(output_path.exists(), "Output file was not created")
            self.assertGreater(output_path.stat().st_size, 1000, "Output file is suspiciously small")

            # Probe the output for warnings (SEI messages, corrupt frames, etc.)
            probe_cmd = [
                "ffprobe",
                "-v", "warning",
                "-select_streams", "v:0",
                "-show_frames",
                "-show_entries", "frame=pkt_pts_time,pict_type,key_frame",
                "-of", "csv",
                str(output_path),
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=60)

            # Collect any non-frame lines from combined stdout+stderr (ffprobe prints
            # warnings to stderr and frame data to stdout).
            # Filter out cosmetic SEI warnings — libx264 embeds encoder-metadata
            # SEI NAL units ("User Data Unregistered") that ffprobe may report at
            # clip boundaries.  These do not affect playability.
            warnings = []
            for line in probe.stderr.splitlines():
                line_stripped = line.strip()
                if line_stripped and "SEI" not in line_stripped:
                    warnings.append(line_stripped)

            self.assertEqual(
                len(warnings), 0,
                f"ffprobe reported {len(warnings)} warning(s) on concatenated output:\n"
                + "\n".join(warnings[:20]),
            )

            # Verify we actually got frame data
            frame_lines = [l for l in probe.stdout.splitlines() if l.startswith("frame,")]
            self.assertGreater(len(frame_lines), 0, "ffprobe returned no frame data")


class TestSortByMatchingEndsIntegration(FootageTestMixin, unittest.TestCase):
    """Integration tests for --sort-by-matching-ends against real footage."""

    def _run(self, extra_args, output_path, expect_success=True):
        """Run concat_clips.py with the given extra arguments."""
        env = {**subprocess.os.environ, "PYTHONIOENCODING": "utf-8"}
        cmd = [
            sys.executable,
            str(self.SCRIPT_DIR / "concat_clips.py"),
            str(self.TEST_FOOTAGE_DIR),
            str(output_path),
        ] + extra_args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
        if expect_success:
            self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}\n{result.stdout}")
        return result

    def test_sorted_output_has_no_stream_errors(self):
        """--sort-by-matching-ends produces a clean, playable concatenation."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(["--sort-by-matching-ends"], output_path)

            self.assertTrue(output_path.exists(), "Output file was not created")
            self.assertGreater(output_path.stat().st_size, 1000, "Output file is suspiciously small")

            # Every input clip should appear exactly once in the chosen order.
            chosen = self._parse_chosen_order(result.stdout)
            expected = sorted(p.name for p in self.TEST_FOOTAGE_DIR.glob("*.mp4"))
            self.assertEqual(sorted(chosen), expected,
                "Ordering must be a permutation of the input clips")

            probe = subprocess.run(
                ["ffprobe", "-v", "warning", "-select_streams", "v:0", "-show_frames",
                 "-show_entries", "frame=pkt_pts_time,pict_type,key_frame", "-of", "csv",
                 str(output_path)],
                capture_output=True, text=True, timeout=60,
            )
            warnings = [l.strip() for l in probe.stderr.splitlines()
                        if l.strip() and "SEI" not in l]
            self.assertEqual(len(warnings), 0,
                f"ffprobe reported warning(s) on sorted output:\n" + "\n".join(warnings[:20]))

    def test_first_clip_substring_leads_the_output(self):
        """--first-clip picks the opening clip by substring, as documented."""
        self._skip_if_missing()

        mp4s = sorted(p.name for p in self.TEST_FOOTAGE_DIR.glob("*.mp4"))
        # Choose a clip that is NOT alphabetically first, so the flag has to work.
        target = mp4s[-1]
        needle = Path(target).stem

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(["--sort-by-matching-ends", "--first-clip", needle], output_path)

            chosen = self._parse_chosen_order(result.stdout)
            self.assertTrue(chosen, "Could not parse the chosen order from stdout")
            self.assertEqual(chosen[0], target,
                f"--first-clip {needle} should place {target} first, got {chosen[0]}")

    def test_shuffle_and_sort_together_is_rejected(self):
        """The two ordering modes cannot be combined."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(["--shuffle", "--sort-by-matching-ends"], output_path,
                               expect_success=False)

            self.assertNotEqual(result.returncode, 0,
                "Combining --shuffle and --sort-by-matching-ends must fail")
            self.assertIn("cannot be combined", result.stdout + result.stderr)
            self.assertFalse(output_path.exists(), "No output should be written on error")

    def test_unmatched_first_clip_is_rejected(self):
        """An unmatched --first-clip pattern fails loudly instead of guessing."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(
                ["--sort-by-matching-ends", "--first-clip", "no-such-clip-xyz"],
                output_path, expect_success=False,
            )

            self.assertNotEqual(result.returncode, 0,
                "An unmatched --first-clip pattern must fail")
            self.assertIn("did not match", result.stdout + result.stderr)


class TestSortByIntensityIntegration(FootageTestMixin, unittest.TestCase):
    """Integration tests for --sort-by-intensity against real footage."""

    def _run(self, extra_args, output_path, expect_success=True):
        env = {**subprocess.os.environ, "PYTHONIOENCODING": "utf-8"}
        cmd = [
            sys.executable,
            str(self.SCRIPT_DIR / "concat_clips.py"),
            str(self.TEST_FOOTAGE_DIR),
            str(output_path),
        ] + extra_args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
        if expect_success:
            self.assertEqual(result.returncode, 0, f"Script failed:\n{result.stderr}\n{result.stdout}")
        return result

    def test_ascending_orders_quietest_to_busiest(self):
        """--sort-by-intensity puts the least motion first."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(["--sort-by-intensity"], output_path)

            entries = self._parse_chosen_order(result.stdout, with_intensity=True)
            self.assertTrue(entries, "Could not parse the chosen order from stdout")

            values = [intensity for _, intensity in entries]
            self.assertTrue(all(v is not None for v in values),
                "Every clip in a pure intensity sort should report a measurement")
            self.assertEqual(values, sorted(values),
                f"Ascending order must be non-decreasing, got {values}")

            self.assertTrue(output_path.exists(), "Output file was not created")
            self.assertGreater(output_path.stat().st_size, 1000, "Output file is suspiciously small")

    def test_descending_reverses_ascending(self):
        """--sort-by-intensity desc is the exact mirror of the default."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            asc = self._parse_chosen_order(
                self._run(["--sort-by-intensity"], Path(tmpdir) / "asc.mp4").stdout)
            desc = self._parse_chosen_order(
                self._run(["--sort-by-intensity", "desc"], Path(tmpdir) / "desc.mp4").stdout)

            self.assertTrue(asc and desc, "Could not parse both orders")
            self.assertEqual(asc, list(reversed(desc)),
                f"desc should reverse asc:\n  asc={asc}\n  desc={desc}")

    def test_combined_with_matching_ends_opens_on_the_quietest(self):
        """Layered onto matching-ends, an ascending arc still starts at the bottom."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            pure = self._parse_chosen_order(
                self._run(["--sort-by-intensity"], Path(tmpdir) / "pure.mp4").stdout)
            combined = self._parse_chosen_order(
                self._run(["--sort-by-matching-ends", "--sort-by-intensity"],
                          Path(tmpdir) / "combined.mp4").stdout)

            self.assertTrue(pure and combined, "Could not parse both orders")
            self.assertEqual(combined[0], pure[0],
                "The combined ordering should still open on the quietest clip")
            self.assertEqual(sorted(combined), sorted(pure),
                "The combined ordering must still be a permutation of the clips")

    def test_explicit_first_clip_overrides_the_intensity_opening(self):
        """--first-clip wins over the automatic quietest/busiest choice."""
        self._skip_if_missing()

        mp4s = sorted(p.name for p in self.TEST_FOOTAGE_DIR.glob("*.mp4"))
        target = mp4s[-1]
        needle = Path(target).stem

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(
                ["--sort-by-matching-ends", "--sort-by-intensity", "--first-clip", needle],
                Path(tmpdir) / "output.mp4",
            )
            chosen = self._parse_chosen_order(result.stdout)
            self.assertEqual(chosen[0], target,
                "An explicit --first-clip must beat the intensity-derived opening")

    def test_combined_output_has_no_stream_errors(self):
        """Both ordering flags together still produce a clean concatenation."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            self._run(["--sort-by-matching-ends", "--sort-by-intensity", "desc", "--match-seams"],
                      output_path)

            probe = subprocess.run(
                ["ffprobe", "-v", "warning", "-select_streams", "v:0", "-show_frames",
                 "-show_entries", "frame=pkt_pts_time,pict_type,key_frame", "-of", "csv",
                 str(output_path)],
                capture_output=True, text=True, timeout=60,
            )
            warnings = [l.strip() for l in probe.stderr.splitlines()
                        if l.strip() and "SEI" not in l]
            self.assertEqual(len(warnings), 0,
                "ffprobe reported warning(s) on combined output:\n" + "\n".join(warnings[:20]))

    def test_shuffle_and_intensity_together_is_rejected(self):
        """Randomising and measuring are contradictory ways to pick an order."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(["--shuffle", "--sort-by-intensity"], output_path,
                               expect_success=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot be combined", result.stdout + result.stderr)
            self.assertFalse(output_path.exists(), "No output should be written on error")

    def test_invalid_direction_is_rejected(self):
        """Anything other than asc/desc fails at argument parsing."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = self._run(["--sort-by-intensity", "sideways"], output_path,
                               expect_success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid choice", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
