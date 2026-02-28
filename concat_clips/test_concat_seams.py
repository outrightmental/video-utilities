#!/usr/bin/env python
"""
Integration tests for concat_clips.py

Tests motion-aware pair matching concepts and end-to-end concatenation integrity
using real test footage.
"""

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


class TestConcatenationIntegrity(unittest.TestCase):
    """Integration test: concatenate test footage and verify stream integrity with ffprobe."""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
