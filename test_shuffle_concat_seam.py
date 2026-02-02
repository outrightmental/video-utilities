#!/usr/bin/env python3
"""
Unit tests for shuffle_concat_seam.py

Tests the core frame matching functions to ensure motion-aware seam matching works correctly.
"""

import sys
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

# Import functions from shuffle_concat_seam
if HAS_OPENCV:
    from shuffle_concat_seam import (
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
    
    @patch('shuffle_concat_seam.HAS_OPENCV', False)
    def test_no_trim_works_without_opencv(self):
        """Verify that --no-trim mode works even without OpenCV installed."""
        from shuffle_concat_seam import shuffle_and_concatenate_videos
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
    
    @patch('shuffle_concat_seam.HAS_OPENCV', True)
    @patch('shuffle_concat_seam.find_best_matching_frame_pair')
    @patch('shuffle_concat_seam.get_last_two_frames')
    def test_no_trim_skips_frame_matching(self, mock_get_last_frames, mock_find_best):
        """Verify that no_trim=True skips the frame matching function and doesn't extract last frames."""
        from shuffle_concat_seam import shuffle_and_concatenate_videos
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
        from shuffle_concat_seam import shuffle_and_concatenate_videos
        import inspect
        
        # Get the function signature
        sig = inspect.signature(shuffle_and_concatenate_videos)
        no_trim_param = sig.parameters['no_trim']
        
        # Verify default value is False
        self.assertEqual(no_trim_param.default, False)


class TestDocumentation(unittest.TestCase):
    """Test that the module docstring is accurate."""
    
    def test_module_mentions_motion_aware(self):
        """Verify the docstring mentions motion-aware matching."""
        from shuffle_concat_seam import __doc__ as module_doc
        
        self.assertIn("motion", module_doc.lower())
        self.assertIn("2 consecutive frames", module_doc.lower())
    
    def test_module_mentions_needle_pair(self):
        """Verify the algorithm description mentions needle pair."""
        from shuffle_concat_seam import __doc__ as module_doc
        
        self.assertIn("needle pair", module_doc.lower())
    
    def test_module_mentions_folder_mode(self):
        """Verify the docstring mentions the folder mode usage."""
        from shuffle_concat_seam import __doc__ as module_doc
        
        self.assertIn("--folder", module_doc.lower())


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
