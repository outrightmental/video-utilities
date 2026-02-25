#!/usr/bin/env python3
"""
Unit tests for shuffle_concat_seam.py

Tests the core frame matching functions to ensure motion-aware seam matching works correctly.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Try to import the module
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Import functions from shuffle_concat_seam module
if HAS_OPENCV:
    from shuffle_concat_seam.shuffle_concat_seam import (
        preprocess_frame_for_comparison,
        compute_frame_difference,
    )


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestFrameComparison(unittest.TestCase):
    """Test the frame preprocessing and comparison functions."""
    
    def test_preprocess_frame_for_comparison(self):
        """Test that frame preprocessing produces a grayscale blurred image."""
        # Create a simple test image (100x100 BGR)
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :, 2] = 255  # Red channel
        
        result = preprocess_frame_for_comparison(test_frame)
        
        # Result should be 2D (grayscale)
        self.assertEqual(len(result.shape), 2)
        # Same dimensions
        self.assertEqual(result.shape, (100, 100))
    
    def test_compute_frame_difference_identical(self):
        """Test that identical frames have zero difference."""
        frame = np.ones((100, 100), dtype=np.uint8) * 128
        
        mse = compute_frame_difference(frame, frame)
        
        self.assertEqual(mse, 0.0)
    
    def test_compute_frame_difference_different(self):
        """Test that different frames have non-zero difference."""
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.ones((100, 100), dtype=np.uint8) * 255
        
        mse = compute_frame_difference(frame1, frame2)
        
        # MSE should be 255^2 = 65025 for all white vs all black
        self.assertEqual(mse, 65025.0)
    
    def test_compute_frame_difference_partial(self):
        """Test MSE for partially different frames."""
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        # Make half the second frame white
        frame2[:, 50:] = 255
        
        mse = compute_frame_difference(frame1, frame2)
        
        # Half the pixels differ by 255, so MSE = 0.5 * (255**2) = 32512.5
        self.assertEqual(mse, 32512.5)
    
    def test_compute_frame_difference_none(self):
        """Test that None frames return infinity."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        
        mse = compute_frame_difference(None, frame)
        self.assertEqual(mse, float('inf'))
        
        mse = compute_frame_difference(frame, None)
        self.assertEqual(mse, float('inf'))
    
    def test_compute_frame_difference_resize(self):
        """Test that frames of different sizes are resized."""
        frame1 = np.ones((100, 100), dtype=np.uint8) * 128
        frame2 = np.ones((50, 50), dtype=np.uint8) * 128
        
        # Should not raise, should resize and compare
        mse = compute_frame_difference(frame1, frame2)
        
        # After resize, frames should match (both are uniform gray)
        self.assertEqual(mse, 0.0)


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


class TestNoTrimMode(unittest.TestCase):
    """Test the --no-trim mode functionality."""
    
    @patch('shuffle_concat_seam.shuffle_concat_seam.HAS_OPENCV', False)
    def test_no_trim_works_without_opencv(self):
        """Verify that --no-trim mode works even without OpenCV installed."""
        from shuffle_concat_seam.shuffle_concat_seam import shuffle_and_concatenate_videos
        import tempfile
        
        # Create a mock setup
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create a fake video file
            fake_video = tmpdir_path / "test.mp4"
            fake_video.write_text("fake video content")
            
            output_path = tmpdir_path / "output.mp4"
            
            # This should not raise an error about OpenCV when no_trim=True
            try:
                shuffle_and_concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video],
                    output_path=output_path,
                    haystack_duration=1.0,
                    seed=42,
                    output_fps=None,
                    no_trim=True
                )
            except RuntimeError as e:
                if "OpenCV is required" in str(e):
                    self.fail("shuffle_and_concatenate_videos raised OpenCV error with no_trim=True")
                # Other errors are okay (since we're not running real ffmpeg)
            except Exception:
                # Other exceptions are expected since we're not running real ffmpeg
                pass
    
    @patch('shuffle_concat_seam.shuffle_concat_seam.HAS_OPENCV', True)
    @patch('shuffle_concat_seam.shuffle_concat_seam.find_best_matching_frame_pair')
    @patch('shuffle_concat_seam.shuffle_concat_seam.get_last_two_frames')
    def test_no_trim_skips_frame_matching(self, mock_get_last_frames, mock_find_best):
        """Verify that no_trim=True skips the frame matching function and doesn't extract last frames."""
        from shuffle_concat_seam.shuffle_concat_seam import shuffle_and_concatenate_videos
        import tempfile
        
        # Create a mock setup
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create fake video files
            fake_video1 = tmpdir_path / "test1.mp4"
            fake_video1.write_text("fake video content 1")
            fake_video2 = tmpdir_path / "test2.mp4"
            fake_video2.write_text("fake video content 2")
            
            output_path = tmpdir_path / "output.mp4"
            
            # Try to call with no_trim=True
            try:
                shuffle_and_concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video1, fake_video2],
                    output_path=output_path,
                    haystack_duration=1.0,
                    seed=42,
                    output_fps=None,
                    no_trim=True
                )
            except Exception:
                # Expected to fail due to missing ffmpeg, but we can still check
                # that find_best_matching_frame_pair and get_last_two_frames were not called
                pass
            
            # Verify that frame matching was not called when no_trim=True
            mock_find_best.assert_not_called()
            # Verify that get_last_two_frames was also not called (not needed for no_trim mode)
            mock_get_last_frames.assert_not_called()
    
    def test_no_trim_flag_default(self):
        """Verify that no_trim defaults to False in function signature."""
        from shuffle_concat_seam.shuffle_concat_seam import shuffle_and_concatenate_videos
        import inspect
        
        # Get the function signature
        sig = inspect.signature(shuffle_and_concatenate_videos)
        no_trim_param = sig.parameters['no_trim']
        
        # Verify default value is False
        self.assertEqual(no_trim_param.default, False)
    
    @patch('shuffle_concat_seam.shuffle_concat_seam.HAS_OPENCV', True)
    @patch('shuffle_concat_seam.shuffle_concat_seam.trim_video_reencode')
    @patch('shuffle_concat_seam.shuffle_concat_seam.find_best_matching_frame_pair')
    @patch('shuffle_concat_seam.shuffle_concat_seam.get_last_two_frames')
    @patch('shuffle_concat_seam.shuffle_concat_seam.get_video_specs')
    def test_trim_mode_calls_frame_matching(self, mock_get_specs, mock_get_last_frames,
                                            mock_find_best, mock_reencode):
        """Verify that no_trim=False (default) DOES call frame matching for successive clips."""
        from shuffle_concat_seam.shuffle_concat_seam import shuffle_and_concatenate_videos
        import tempfile
        
        # Mock get_video_specs to return valid specs
        mock_get_specs.return_value = {
            'codec': 'h264',
            'width': 1920,
            'height': 1080,
            'fps': 30.0,
            'duration': 10.0
        }
        
        # Mock get_last_two_frames to return a valid tuple of fake frames
        mock_get_last_frames.return_value = (MagicMock(), MagicMock())
        # Mock find_best_matching_frame_pair to return a trim time
        mock_find_best.return_value = (0.5, 100.0)
        # Mock trim_video_reencode to succeed (needed for codec consistency re-encode)
        mock_reencode.return_value = True
        
        # Create a mock setup
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create placeholder video files (content doesn't matter since get_video_specs is mocked)
            # Using bytes to be more representative of actual video files
            fake_video1 = tmpdir_path / "test1.mp4"
            fake_video1.write_bytes(b'\x00\x00\x00\x00')
            fake_video2 = tmpdir_path / "test2.mp4"
            fake_video2.write_bytes(b'\x00\x00\x00\x00')
            
            output_path = tmpdir_path / "output.mp4"
            
            # Try to call with no_trim=False (default)
            # This will fail when trying to actually process the fake video files with ffmpeg,
            # but we only care about verifying that the frame matching functions are called
            try:
                shuffle_and_concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video1, fake_video2],
                    output_path=output_path,
                    haystack_duration=1.0,
                    seed=42,
                    output_fps=None,
                    no_trim=False  # Explicitly set to False (the default)
                )
            except (RuntimeError, subprocess.CalledProcessError, OSError):
                # Expected failures when ffmpeg tries to process placeholder files
                pass
            
            # Verify that get_last_two_frames was called (to extract frames for matching)
            # It should be called at least once for the first clip
            self.assertTrue(mock_get_last_frames.called, 
                "get_last_two_frames should be called when no_trim=False")
            
            # Verify that find_best_matching_frame_pair was called for the second clip
            self.assertTrue(mock_find_best.called,
                "find_best_matching_frame_pair should be called for successive clips when no_trim=False")


    @patch('shuffle_concat_seam.shuffle_concat_seam.HAS_OPENCV', True)
    @patch('shuffle_concat_seam.shuffle_concat_seam.trim_video_reencode')
    @patch('shuffle_concat_seam.shuffle_concat_seam.find_best_matching_frame_pair')
    @patch('shuffle_concat_seam.shuffle_concat_seam.get_last_two_frames')
    @patch('shuffle_concat_seam.shuffle_concat_seam.get_video_specs')
    def test_first_clip_reencoded_for_consistency(self, mock_get_specs, mock_get_last_frames,
                                                   mock_find_best, mock_reencode):
        """Verify that the first clip is re-encoded for codec consistency when no_trim=False.

        Without this, the concat demuxer mixes the original encoding (possibly with
        B-frames, different SPS/PPS) with re-encoded trimmed clips, producing
        broken output.
        """
        from shuffle_concat_seam.shuffle_concat_seam import shuffle_and_concatenate_videos

        mock_get_specs.return_value = {
            'codec': 'h264',
            'width': 1920,
            'height': 1080,
            'fps': 30.0,
            'duration': 10.0
        }
        mock_get_last_frames.return_value = (MagicMock(), MagicMock())
        mock_find_best.return_value = (0.5, 100.0)
        # Let re-encode succeed and produce a temp file
        mock_reencode.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            fake_video1 = tmpdir_path / "test1.mp4"
            fake_video1.write_bytes(b'\x00\x00\x00\x00')
            fake_video2 = tmpdir_path / "test2.mp4"
            fake_video2.write_bytes(b'\x00\x00\x00\x00')

            output_path = tmpdir_path / "output.mp4"

            try:
                shuffle_and_concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video1, fake_video2],
                    output_path=output_path,
                    haystack_duration=1.0,
                    seed=42,
                    output_fps=None,
                    no_trim=False
                )
            except (RuntimeError, subprocess.CalledProcessError, OSError):
                pass

            # The first clip has trim_start=0 and matching specs, so it hits the
            # else branch.  With the fix, trim_video_reencode should be called
            # with start_time=0.0 for that first clip to ensure codec consistency.
            first_call_args = mock_reencode.call_args_list[0]
            self.assertAlmostEqual(first_call_args[0][3], 0.0,
                msg="First clip should be re-encoded with start_time=0.0 for consistency")


class TestDocumentation(unittest.TestCase):
    """Test that the module docstring is accurate."""
    
    def test_module_mentions_motion_aware(self):
        """Verify the docstring mentions motion-aware matching."""
        from shuffle_concat_seam.shuffle_concat_seam import __doc__ as module_doc
        
        self.assertIn("motion", module_doc.lower())
        self.assertIn("2 consecutive frames", module_doc.lower())
    
    def test_module_mentions_needle_pair(self):
        """Verify the algorithm description mentions needle pair."""
        from shuffle_concat_seam.shuffle_concat_seam import __doc__ as module_doc
        
        self.assertIn("needle pair", module_doc.lower())
    
    def test_module_mentions_folder_mode(self):
        """Verify the docstring mentions the folder mode usage."""
        from shuffle_concat_seam.shuffle_concat_seam import __doc__ as module_doc
        
        self.assertIn("--folder", module_doc.lower())


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
        """Run shuffle_concat_seam on test footage and verify ffprobe reports no warnings."""
        self._skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            # Run the script
            env = {**subprocess.os.environ, "PYTHONIOENCODING": "utf-8"}
            cmd = [
                sys.executable,
                str(self.SCRIPT_DIR / "shuffle_concat_seam.py"),
                str(self.TEST_FOOTAGE_DIR),
                str(output_path),
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
            # warnings to stderr and frame data to stdout)
            warnings = []
            for line in probe.stderr.splitlines():
                line_stripped = line.strip()
                if line_stripped:
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
    # Run with verbosity
    unittest.main(verbosity=2)
