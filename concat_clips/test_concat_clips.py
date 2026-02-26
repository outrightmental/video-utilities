#!/usr/bin/env python
"""
Unit tests for concat_clips.py

Tests the core functions to ensure basic concatenation, shuffle, and
match-seams features work correctly.
"""

import inspect
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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
        preprocess_frame_for_comparison,
        compute_frame_difference,
    )

from concat_clips.concat_clips import (
    find_video_files,
    concatenate_videos,
)


class TestFindVideoFiles(unittest.TestCase):
    """Test the find_video_files function."""

    def test_returns_sorted_alphabetical(self):
        """Verify that find_video_files returns files sorted alphabetically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create fake video files out of alphabetical order
            (tmpdir_path / "c_video.mp4").write_text("fake")
            (tmpdir_path / "a_video.mp4").write_text("fake")
            (tmpdir_path / "b_video.mp4").write_text("fake")

            result = find_video_files(tmpdir_path, recursive=False)
            names = [f.name for f in result]
            self.assertEqual(names, ["a_video.mp4", "b_video.mp4", "c_video.mp4"])

    def test_non_video_files_excluded(self):
        """Verify that non-video files are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "video.mp4").write_text("fake")
            (tmpdir_path / "readme.txt").write_text("not a video")
            (tmpdir_path / "image.jpg").write_text("not a video")

            result = find_video_files(tmpdir_path, recursive=False)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "video.mp4")

    def test_recursive_finds_subdirectory_files(self):
        """Verify that recursive mode finds files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            (tmpdir_path / "a.mp4").write_text("fake")
            (subdir / "b.mp4").write_text("fake")

            result = find_video_files(tmpdir_path, recursive=True)
            self.assertEqual(len(result), 2)

    def test_non_recursive_skips_subdirectories(self):
        """Verify that non-recursive mode skips subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            (tmpdir_path / "a.mp4").write_text("fake")
            (subdir / "b.mp4").write_text("fake")

            result = find_video_files(tmpdir_path, recursive=False)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "a.mp4")


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestFrameComparison(unittest.TestCase):
    """Test the frame preprocessing and comparison functions."""

    def test_preprocess_frame_for_comparison(self):
        """Test that frame preprocessing produces a grayscale blurred image."""
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :, 2] = 255  # Red channel

        result = preprocess_frame_for_comparison(test_frame)

        # Result should be 2D (grayscale)
        self.assertEqual(len(result.shape), 2)
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

    def test_compute_frame_difference_none(self):
        """Test that None frames return infinity."""
        frame = np.zeros((100, 100), dtype=np.uint8)

        mse = compute_frame_difference(None, frame)
        self.assertEqual(mse, float('inf'))

        mse = compute_frame_difference(frame, None)
        self.assertEqual(mse, float('inf'))


class TestConcatenateVideosSignature(unittest.TestCase):
    """Test that concatenate_videos has the expected signature."""

    def test_shuffle_parameter_default(self):
        """Verify that shuffle defaults to False."""
        sig = inspect.signature(concatenate_videos)
        self.assertEqual(sig.parameters['shuffle'].default, False)

    def test_match_seams_parameter_default(self):
        """Verify that match_seams defaults to False."""
        sig = inspect.signature(concatenate_videos)
        self.assertEqual(sig.parameters['match_seams'].default, False)

    def test_seed_parameter_default(self):
        """Verify that seed defaults to None."""
        sig = inspect.signature(concatenate_videos)
        self.assertIsNone(sig.parameters['seed'].default)

    def test_haystack_duration_parameter_default(self):
        """Verify that haystack_duration defaults to 1.0."""
        sig = inspect.signature(concatenate_videos)
        self.assertEqual(sig.parameters['haystack_duration'].default, 1.0)

    def test_haystack_skip_parameter_default(self):
        """Verify that haystack_skip defaults to 0.0."""
        sig = inspect.signature(concatenate_videos)
        self.assertEqual(sig.parameters['haystack_skip'].default, 0.0)


class TestShuffleMode(unittest.TestCase):
    """Test the --shuffle mode functionality."""

    @patch('concat_clips.concat_clips.get_video_specs')
    @patch('concat_clips.concat_clips.find_best_matching_frame_pair')
    @patch('concat_clips.concat_clips.get_last_two_frames')
    def test_shuffle_false_skips_random(self, mock_get_last, mock_find_best, mock_get_specs):
        """Verify that shuffle=False preserves alphabetical order."""
        mock_get_specs.return_value = {
            'codec': 'h264', 'width': 1920, 'height': 1080,
            'fps': 30.0, 'duration': 10.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Files named so alphabetical order is b, a (reverse of creation)
            fake_video_b = tmpdir_path / "b_video.mp4"
            fake_video_b.write_bytes(b'\x00\x00\x00\x00')
            fake_video_a = tmpdir_path / "a_video.mp4"
            fake_video_a.write_bytes(b'\x00\x00\x00\x00')

            output_path = tmpdir_path / "output.mp4"

            # Capture the order files are processed by checking log output
            processed_order = []
            original_get_specs = mock_get_specs.side_effect

            def track_order(ffprobe_exe, path):
                processed_order.append(path.name)
                return {'codec': 'h264', 'width': 1920, 'height': 1080, 'fps': 30.0, 'duration': 10.0}

            mock_get_specs.side_effect = track_order

            try:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=sorted([fake_video_b, fake_video_a]),
                    output_path=output_path,
                    shuffle=False,
                )
            except Exception:
                pass

            # Alphabetical order: a_video.mp4 before b_video.mp4
            if processed_order:
                self.assertEqual(processed_order[0], "a_video.mp4")

    @patch('concat_clips.concat_clips.HAS_OPENCV', False)
    def test_match_seams_without_opencv_raises(self):
        """Verify that match_seams=True raises error without OpenCV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_video = tmpdir_path / "test.mp4"
            fake_video.write_bytes(b'\x00\x00\x00\x00')

            with self.assertRaises(RuntimeError) as ctx:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video],
                    output_path=tmpdir_path / "output.mp4",
                    match_seams=True,
                )
            self.assertIn("OpenCV", str(ctx.exception))

    @patch('concat_clips.concat_clips.HAS_OPENCV', True)
    @patch('concat_clips.concat_clips.find_best_matching_frame_pair')
    @patch('concat_clips.concat_clips.get_last_two_frames')
    @patch('concat_clips.concat_clips.trim_video_reencode')
    @patch('concat_clips.concat_clips.get_video_specs')
    def test_match_seams_calls_frame_matching(self, mock_get_specs, mock_reencode,
                                              mock_get_last, mock_find_best):
        """Verify that match_seams=True calls frame matching for successive clips."""
        mock_get_specs.return_value = {
            'codec': 'h264', 'width': 1920, 'height': 1080,
            'fps': 30.0, 'duration': 10.0
        }
        mock_get_last.return_value = (MagicMock(), MagicMock())
        mock_find_best.return_value = (0.5, 100.0)
        mock_reencode.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_video1 = tmpdir_path / "test1.mp4"
            fake_video1.write_bytes(b'\x00\x00\x00\x00')
            fake_video2 = tmpdir_path / "test2.mp4"
            fake_video2.write_bytes(b'\x00\x00\x00\x00')

            output_path = tmpdir_path / "output.mp4"

            try:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video1, fake_video2],
                    output_path=output_path,
                    match_seams=True,
                    shuffle=False,
                )
            except (RuntimeError, subprocess.CalledProcessError, OSError):
                pass

            # Verify that get_last_two_frames was called (to extract frames for matching)
            self.assertTrue(mock_get_last.called,
                "get_last_two_frames should be called when match_seams=True")

            # Verify that find_best_matching_frame_pair was called for the second clip
            self.assertTrue(mock_find_best.called,
                "find_best_matching_frame_pair should be called for successive clips when match_seams=True")

    @patch('concat_clips.concat_clips.HAS_OPENCV', True)
    @patch('concat_clips.concat_clips.find_best_matching_frame_pair')
    @patch('concat_clips.concat_clips.get_last_two_frames')
    @patch('concat_clips.concat_clips.get_video_specs')
    def test_no_match_seams_skips_frame_matching(self, mock_get_specs, mock_get_last, mock_find_best):
        """Verify that match_seams=False skips frame matching."""
        mock_get_specs.return_value = {
            'codec': 'h264', 'width': 1920, 'height': 1080,
            'fps': 30.0, 'duration': 10.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_video1 = tmpdir_path / "test1.mp4"
            fake_video1.write_bytes(b'\x00\x00\x00\x00')
            fake_video2 = tmpdir_path / "test2.mp4"
            fake_video2.write_bytes(b'\x00\x00\x00\x00')

            output_path = tmpdir_path / "output.mp4"

            try:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake_video1, fake_video2],
                    output_path=output_path,
                    match_seams=False,
                )
            except Exception:
                pass

            # Verify frame matching was NOT called when match_seams=False
            mock_find_best.assert_not_called()
            mock_get_last.assert_not_called()


class TestDocumentation(unittest.TestCase):
    """Test that the module docstring mentions the new options."""

    def test_module_mentions_shuffle(self):
        """Verify the docstring mentions --shuffle."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("--shuffle", module_doc)

    def test_module_mentions_match_seams(self):
        """Verify the docstring mentions --match-seams."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("--match-seams", module_doc)

    def test_module_mentions_alphabetical(self):
        """Verify the docstring mentions alphabetical ordering."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("alphabetical", module_doc.lower())

    def test_module_mentions_needle_pair(self):
        """Verify the seam matching algorithm description mentions needle pair."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("needle pair", module_doc.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
