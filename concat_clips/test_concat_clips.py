#!/usr/bin/env python
"""
Unit tests for concat_clips.py

Tests the core functions to ensure basic concatenation, shuffle,
sort-by-matching-ends, and match-seams features work correctly.
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
        find_best_seam,
        summarize_boundary,
        score_clip_transition,
        BoundarySignature,
    )

from concat_clips.concat_clips import (
    find_video_files,
    concatenate_videos,
    select_first_clip_index,
    order_clips_by_matching_ends,
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

    def test_sort_by_matching_ends_parameter_default(self):
        """Verify that sort_by_matching_ends defaults to False."""
        sig = inspect.signature(concatenate_videos)
        self.assertEqual(sig.parameters['sort_by_matching_ends'].default, False)

    def test_first_clip_parameter_default(self):
        """Verify that first_clip defaults to None."""
        sig = inspect.signature(concatenate_videos)
        self.assertIsNone(sig.parameters['first_clip'].default)

    def test_sort_window_parameter_default(self):
        """Verify that sort_window defaults to 0.25."""
        sig = inspect.signature(concatenate_videos)
        self.assertEqual(sig.parameters['sort_window'].default, 0.25)


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
    @patch('concat_clips.concat_clips.find_best_seam')
    @patch('concat_clips.concat_clips.extract_haystack_frames')
    @patch('concat_clips.concat_clips.trim_video_reencode')
    @patch('concat_clips.concat_clips.get_video_specs')
    def test_match_seams_calls_frame_matching(self, mock_get_specs, mock_reencode,
                                              mock_extract, mock_find_best):
        """Verify that match_seams=True calls frame matching for successive clips."""
        mock_get_specs.return_value = {
            'codec': 'h264', 'width': 1920, 'height': 1080,
            'fps': 30.0, 'duration': 10.0
        }
        mock_extract.return_value = [(0.0, MagicMock()), (0.033, MagicMock())]
        mock_find_best.return_value = (9.0, 0.5, 100.0)  # (trim_end_preceding, trim_start_successor, score)
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

            # Verify that extract_haystack_frames was called (to extract frames for matching)
            self.assertTrue(mock_extract.called,
                "extract_haystack_frames should be called when match_seams=True")

            # Verify that find_best_seam was called for the adjacent clip pair
            self.assertTrue(mock_find_best.called,
                "find_best_seam should be called for adjacent clip pairs when match_seams=True")

    @patch('concat_clips.concat_clips.HAS_OPENCV', True)
    @patch('concat_clips.concat_clips.find_best_seam')
    @patch('concat_clips.concat_clips.extract_haystack_frames')
    @patch('concat_clips.concat_clips.get_video_specs')
    def test_no_match_seams_skips_frame_matching(self, mock_get_specs, mock_extract, mock_find_best):
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
            mock_extract.assert_not_called()


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestFindBestSeam(unittest.TestCase):
    """Test the find_best_seam function."""

    def _make_frames(self, positions):
        """Create a list of (timestamp, grayscale_frame) tuples for testing.

        Each 'position' is an x-offset for a white square in a black 100×100 frame.
        """
        frames = []
        for t, x in positions:
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, x:x + 20] = 255
            frames.append((t, frame))
        return frames

    def test_returns_none_for_short_preceding_list(self):
        """Returns (None, None, inf) when preceding list has fewer than 2 frames."""
        preceding = self._make_frames([(0.0, 10)])  # Only 1 frame
        successor = self._make_frames([(0.0, 10), (0.033, 30)])

        pre_end, suc_start, score = find_best_seam(preceding, successor)

        self.assertIsNone(pre_end)
        self.assertIsNone(suc_start)
        self.assertEqual(score, float('inf'))

    def test_returns_none_for_short_successor_list(self):
        """Returns (None, None, inf) when successor list has fewer than 2 frames."""
        preceding = self._make_frames([(0.0, 10), (0.033, 30)])
        successor = self._make_frames([(0.0, 10)])  # Only 1 frame

        pre_end, suc_start, score = find_best_seam(preceding, successor)

        self.assertIsNone(pre_end)
        self.assertIsNone(suc_start)
        self.assertEqual(score, float('inf'))

    def test_prefers_similar_junction_frames(self):
        """The seam with the most similar junction frames wins."""
        # Preceding has two candidate ending frames at x=10 and x=50.
        # Successor head starts at x=10 (matches first ending candidate).
        preceding = self._make_frames([(8.0, 30), (8.033, 10)])   # pair: ending at x=10
        successor = self._make_frames([(0.0, 10), (0.033, 30)])   # starts at x=10

        pre_end, suc_start, score = find_best_seam(preceding, successor)

        self.assertIsNotNone(pre_end)
        self.assertIsNotNone(suc_start)
        # The best junction is where a_curr (x=10) matches b_curr (x=10), so pre_end = 8.033
        self.assertAlmostEqual(pre_end, 8.033, places=2)

    def test_trim_end_is_a_curr_and_start_is_b_next(self):
        """trim_end is the time of a_curr; trim_start is the time of b_next (not b_curr)."""
        preceding = self._make_frames([(8.0, 30), (8.033, 10)])
        successor = self._make_frames([(0.0, 10), (0.033, 30)])

        pre_end, suc_start, score = find_best_seam(preceding, successor)

        # The junction is a_curr (t=8.033) ≈ b_curr (t=0.0).
        # The first included successor frame is b_next (t=0.033).
        self.assertAlmostEqual(pre_end, 8.033, places=2)
        self.assertAlmostEqual(suc_start, 0.033, places=2)


class TestSelectFirstClipIndex(unittest.TestCase):
    """Test --first-clip substring selection."""

    def _files(self, *names):
        return [Path("/videos") / n for n in names]

    def test_none_pattern_selects_alphabetically_first(self):
        """With no pattern, the first (alphabetically sorted) clip is chosen."""
        files = self._files("a.mp4", "b.mp4", "c.mp4")
        self.assertEqual(select_first_clip_index(files, None), 0)

    def test_substring_matches_partial_filename(self):
        """A substring matches anywhere within the filename."""
        files = self._files("aaa.mp4", "joe123-final.mp4", "zzz.mp4")
        self.assertEqual(select_first_clip_index(files, "joe123"), 1)

    def test_substring_match_is_case_insensitive(self):
        """Matching ignores case on both sides."""
        files = self._files("aaa.mp4", "Joe123-FINAL.mp4")
        self.assertEqual(select_first_clip_index(files, "joe123"), 1)
        self.assertEqual(select_first_clip_index(files, "JOE123"), 1)

    def test_multiple_matches_uses_alphabetically_first(self):
        """When several clips match, the alphabetically first one wins."""
        files = self._files("joe123-a.mp4", "joe123-b.mp4", "other.mp4")
        self.assertEqual(select_first_clip_index(files, "joe123"), 0)

    def test_no_match_raises_with_available_names(self):
        """An unmatched pattern is an error that names the available clips."""
        files = self._files("aaa.mp4", "bbb.mp4")
        with self.assertRaises(ValueError) as ctx:
            select_first_clip_index(files, "nonexistent")
        message = str(ctx.exception)
        self.assertIn("nonexistent", message)
        self.assertIn("aaa.mp4", message)

    def test_falls_back_to_full_path_match(self):
        """A pattern naming a subdirectory matches when no filename does."""
        files = [Path("/videos/session-a/clip.mp4"), Path("/videos/session-b/clip.mp4")]
        self.assertEqual(select_first_clip_index(files, "session-b"), 1)

    def test_filename_match_takes_precedence_over_path(self):
        """Filenames are searched first, so a directory of the same name loses."""
        files = [Path("/videos/joe123/clip.mp4"), Path("/videos/other/joe123.mp4")]
        self.assertEqual(select_first_clip_index(files, "joe123"), 1)

    def test_empty_file_list_raises(self):
        """Selecting from no clips is an error, not an index crash."""
        with self.assertRaises(ValueError):
            select_first_clip_index([], "anything")


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestSummarizeBoundary(unittest.TestCase):
    """Test boundary appearance/motion summarization."""

    def _frames(self, positions):
        """Build (timestamp, frame) tuples with a white square at each x position."""
        out = []
        for i, x in enumerate(positions):
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, x:x + 20] = 255
            out.append((i * 0.033, frame))
        return out

    def test_empty_window_returns_none(self):
        """No frames means no signature."""
        self.assertIsNone(summarize_boundary([], at_end=True))

    def test_single_frame_has_zero_motion(self):
        """One frame still yields a usable appearance-only signature."""
        sig = summarize_boundary(self._frames([10]), at_end=True)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.speed, 0.0)
        np.testing.assert_array_equal(sig.motion, np.zeros(2))

    def test_at_end_uses_last_frame_as_boundary(self):
        """A tail signature's boundary frame is the clip's final frame."""
        frames = self._frames([10, 50])
        sig = summarize_boundary(frames, at_end=True)
        np.testing.assert_array_equal(sig.frame, frames[-1][1])

    def test_at_start_uses_first_frame_as_boundary(self):
        """A head signature's boundary frame is the clip's opening frame."""
        frames = self._frames([10, 50])
        sig = summarize_boundary(frames, at_end=False)
        np.testing.assert_array_equal(sig.frame, frames[0][1])

    def test_motion_vector_points_in_direction_of_travel(self):
        """Rightward travel yields a positive column component, leftward negative."""
        right = summarize_boundary(self._frames([10, 30, 50]), at_end=True)
        left = summarize_boundary(self._frames([50, 30, 10]), at_end=True)
        self.assertGreater(right.motion[1], 0)
        self.assertLess(left.motion[1], 0)

    def test_faster_motion_yields_higher_speed(self):
        """Speed tracks how much the picture changes per frame."""
        fast = summarize_boundary(self._frames([10, 60]), at_end=True)
        slow = summarize_boundary(self._frames([10, 12]), at_end=True)
        self.assertGreater(fast.speed, slow.speed)


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestScoreClipTransition(unittest.TestCase):
    """Test the three criteria that rank one clip following another."""

    def _sig(self, positions, at_end):
        frames = []
        for i, x in enumerate(positions):
            frame = np.zeros((100, 100), dtype=np.uint8)
            frame[40:60, x:x + 20] = 255
            frames.append((i * 0.033, frame))
        return summarize_boundary(frames, at_end=at_end)

    def test_prefers_matching_appearance(self):
        """A head that looks like the previous tail beats one that does not."""
        tail = self._sig([20, 40, 60], at_end=True)          # ends with square at x=60
        looks_same = self._sig([60, 80, 60], at_end=False)   # opens at x=60
        looks_different = self._sig([10, 30, 10], at_end=False)  # opens at x=10

        self.assertLess(score_clip_transition(tail, looks_same),
                        score_clip_transition(tail, looks_different))

    def test_prefers_same_motion_direction(self):
        """Between two identical-looking heads, the one continuing rightward wins."""
        tail = self._sig([20, 40, 60], at_end=True)          # moving right
        continues_right = self._sig([60, 70, 80], at_end=False)
        reverses_left = self._sig([60, 50, 40], at_end=False)

        self.assertLess(score_clip_transition(tail, continues_right),
                        score_clip_transition(tail, reverses_left))

    def test_reversal_scores_worse_than_no_motion(self):
        """A reversal is actively penalised, not merely unrewarded.

        This is the key difference from seam matching: when ordering clips there
        is always another candidate, so moving the wrong way must cost something.
        """
        tail = self._sig([20, 40, 60], at_end=True)
        reverses_left = self._sig([60, 50, 40], at_end=False)
        static = self._sig([60, 60, 60], at_end=False)

        self.assertGreater(score_clip_transition(tail, reverses_left),
                           score_clip_transition(tail, static))

    def test_prefers_matching_speed(self):
        """Between two heads moving the same way, the closer speed wins."""
        # Tail moves 20px per frame.
        tail = self._sig([20, 40, 60], at_end=True)
        similar_speed = self._sig([60, 80, 60], at_end=False)  # ~20px per frame
        crawling = self._sig([60, 62, 60], at_end=False)       # ~2px per frame

        self.assertLess(score_clip_transition(tail, similar_speed),
                        score_clip_transition(tail, crawling))

    def test_appearance_dominates_direction(self):
        """A wildly different-looking head loses even with perfect motion match."""
        tail = self._sig([20, 40, 60], at_end=True)
        same_dir_wrong_look = self._sig([10, 30, 50], at_end=False)
        opposite_dir_right_look = self._sig([60, 50, 40], at_end=False)

        self.assertLess(score_clip_transition(tail, opposite_dir_right_look),
                        score_clip_transition(tail, same_dir_wrong_look))

    def test_identical_static_boundaries_are_finite(self):
        """Two motionless boundaries score finitely rather than dividing by zero."""
        tail = self._sig([30, 30], at_end=True)
        head = self._sig([30, 30], at_end=False)
        score = score_clip_transition(tail, head)
        self.assertTrue(np.isfinite(score))


class TestOrderClipsByMatchingEnds(unittest.TestCase):
    """Test the greedy ordering itself, independent of frame extraction."""

    def _stub(self, value):
        """A signature stand-in whose transition score is read from a lookup table."""
        return BoundarySignature(frame=value, motion=None, speed=0.0) if HAS_OPENCV else value

    def test_empty_input_returns_empty(self):
        """No clips means no ordering."""
        self.assertEqual(order_clips_by_matching_ends([], [], [], 0), [])

    def test_out_of_range_first_index_raises(self):
        """A first_index outside the clip list is a programming error."""
        with self.assertRaises(ValueError):
            order_clips_by_matching_ends(["a.mp4"], [None], [None], 5)

    def test_first_index_leads_the_order(self):
        """The chosen opening clip is always placed first."""
        names = ["a.mp4", "b.mp4", "c.mp4"]
        order = order_clips_by_matching_ends(names, [None] * 3, [None] * 3, 2)
        self.assertEqual(order[0], 2)

    def test_all_clips_appear_exactly_once(self):
        """Ordering is a permutation — nothing is dropped or duplicated."""
        names = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
        order = order_clips_by_matching_ends(names, [None] * 4, [None] * 4, 0)
        self.assertEqual(sorted(order), [0, 1, 2, 3])

    def test_unanalysable_clips_fall_back_to_alphabetical(self):
        """With every score infinite, ties break on filename for determinism."""
        names = ["b.mp4", "a.mp4", "c.mp4"]
        order = order_clips_by_matching_ends(names, [None] * 3, [None] * 3, 0)
        # Starts at index 0 ("b.mp4"), then the remaining two in name order.
        self.assertEqual([names[i] for i in order], ["b.mp4", "a.mp4", "c.mp4"])

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is required for this test")
    def test_greedy_chain_follows_best_match(self):
        """Each step appends the best-scoring remaining clip.

        Scores are injected via a patched score_clip_transition so the ordering
        logic is tested without depending on the scoring formula's constants.
        """
        names = ["start.mp4", "far.mp4", "near.mp4", "mid.mp4"]
        tails = [BoundarySignature(frame=i, motion=None, speed=0.0) for i in range(4)]
        heads = [BoundarySignature(frame=i, motion=None, speed=0.0) for i in range(4)]

        # From clip 0 the best next is 2, from 2 the best next is 3, leaving 1.
        table = {
            (0, 1): 9.0, (0, 2): 1.0, (0, 3): 5.0,
            (2, 1): 8.0, (2, 3): 2.0,
            (3, 1): 7.0,
        }

        def fake_score(tail, head):
            return table[(tail.frame, head.frame)]

        with patch('concat_clips.concat_clips.score_clip_transition', side_effect=fake_score):
            order = order_clips_by_matching_ends(names, tails, heads, 0)

        self.assertEqual(order, [0, 2, 3, 1])


class TestSortByMatchingEndsValidation(unittest.TestCase):
    """Test the guard rails around --sort-by-matching-ends."""

    def _one_fake_video(self, tmpdir_path):
        fake = tmpdir_path / "test.mp4"
        fake.write_bytes(b'\x00\x00\x00\x00')
        return fake

    def test_shuffle_and_sort_together_raises(self):
        """--shuffle and --sort-by-matching-ends are mutually exclusive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[self._one_fake_video(tmpdir_path)],
                    output_path=tmpdir_path / "output.mp4",
                    shuffle=True,
                    sort_by_matching_ends=True,
                )
            self.assertIn("cannot be combined", str(ctx.exception))

    def test_first_clip_without_sort_raises(self):
        """--first-clip is meaningless without --sort-by-matching-ends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[self._one_fake_video(tmpdir_path)],
                    output_path=tmpdir_path / "output.mp4",
                    first_clip="joe123",
                )
            self.assertIn("--first-clip", str(ctx.exception))

    @patch('concat_clips.concat_clips.HAS_OPENCV', False)
    def test_sort_without_opencv_raises(self):
        """Ordering needs OpenCV and says so."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with self.assertRaises(RuntimeError) as ctx:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[self._one_fake_video(tmpdir_path)],
                    output_path=tmpdir_path / "output.mp4",
                    sort_by_matching_ends=True,
                )
            self.assertIn("OpenCV", str(ctx.exception))

    @patch('concat_clips.concat_clips.HAS_OPENCV', True)
    def test_non_positive_sort_window_raises(self):
        """A zero-length analysis window is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with self.assertRaises(ValueError) as ctx:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[self._one_fake_video(tmpdir_path)],
                    output_path=tmpdir_path / "output.mp4",
                    sort_by_matching_ends=True,
                    sort_window=0.0,
                )
            self.assertIn("sort_window", str(ctx.exception))

    @patch('concat_clips.concat_clips.HAS_OPENCV', True)
    @patch('concat_clips.concat_clips.get_video_specs')
    @patch('concat_clips.concat_clips.sort_clips_by_matching_ends')
    def test_sort_reorders_before_concatenation(self, mock_sort, mock_get_specs):
        """The ordering pass runs, and its result drives the rest of the pipeline."""
        mock_get_specs.return_value = {
            'codec': 'h264', 'width': 1920, 'height': 1080,
            'fps': 30.0, 'duration': 10.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            first = tmpdir_path / "a_video.mp4"
            first.write_bytes(b'\x00\x00\x00\x00')
            second = tmpdir_path / "b_video.mp4"
            second.write_bytes(b'\x00\x00\x00\x00')

            # Ordering decides b comes first, reversing the alphabetical input.
            mock_sort.return_value = [second, first]

            try:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[first, second],
                    output_path=tmpdir_path / "output.mp4",
                    sort_by_matching_ends=True,
                    first_clip="b_video",
                )
            except Exception:
                pass

            self.assertTrue(mock_sort.called,
                "sort_clips_by_matching_ends should run when sort_by_matching_ends=True")
            # The requested first-clip pattern is forwarded to the ordering pass.
            self.assertEqual(mock_sort.call_args.kwargs.get("first_clip"), "b_video")
            # Specs are read from whichever clip the ordering put first.
            self.assertEqual(mock_get_specs.call_args_list[0].args[1].name, "b_video.mp4")

    @patch('concat_clips.concat_clips.HAS_OPENCV', True)
    @patch('concat_clips.concat_clips.sort_clips_by_matching_ends')
    @patch('concat_clips.concat_clips.get_video_specs')
    def test_no_sort_flag_skips_ordering(self, mock_get_specs, mock_sort):
        """Ordering is not attempted unless asked for."""
        mock_get_specs.return_value = {
            'codec': 'h264', 'width': 1920, 'height': 1080,
            'fps': 30.0, 'duration': 10.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake = self._one_fake_video(tmpdir_path)

            try:
                concatenate_videos(
                    ffmpeg_exe="ffmpeg",
                    ffprobe_exe="ffprobe",
                    video_files=[fake],
                    output_path=tmpdir_path / "output.mp4",
                )
            except Exception:
                pass

            mock_sort.assert_not_called()


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for these tests")
class TestPreprocessDownscale(unittest.TestCase):
    """Test the downscaling used by the ordering pass."""

    def test_downscale_shrinks_and_preserves_aspect(self):
        """A wide frame is reduced to the requested width, keeping its shape."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = preprocess_frame_for_comparison(frame, downscale_width=100)
        self.assertEqual(result.shape, (50, 100))

    def test_narrow_frames_are_not_upscaled(self):
        """Frames already narrower than the target are left at their own size."""
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        result = preprocess_frame_for_comparison(frame, downscale_width=256)
        self.assertEqual(result.shape, (40, 80))

    def test_downscale_is_opt_in(self):
        """Without the argument, frames keep their original resolution."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = preprocess_frame_for_comparison(frame)
        self.assertEqual(result.shape, (200, 400))


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

    def test_module_mentions_velocity(self):
        """Verify the seam matching algorithm description mentions velocity."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("velocity", module_doc.lower())

    def test_module_mentions_sort_by_matching_ends(self):
        """Verify the docstring mentions --sort-by-matching-ends."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("--sort-by-matching-ends", module_doc)

    def test_module_mentions_first_clip(self):
        """Verify the docstring mentions --first-clip."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("--first-clip", module_doc)

    def test_module_mentions_ordering_uses_motion_direction(self):
        """Verify the ordering algorithm description covers motion direction."""
        from concat_clips.concat_clips import __doc__ as module_doc
        self.assertIn("direction", module_doc.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
