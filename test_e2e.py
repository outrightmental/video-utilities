#!/usr/bin/env python3
"""
End-to-end test for motion_cctv.py

Tests that:
1. motion_cctv.py runs successfully on example footage
2. Motion segments are detected correctly
3. SUBSET AXIOM: If input B is a subset of input A, then the detected motion 
   segments for B (mapped to A's timeline) must be a subset of A's detected segments.
   
   This validates the critical property that the algorithm produces consistent
   results regardless of clip length - a shorter subset clip should never detect
   motion that isn't also detected in the parent clip.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


# Ensure static_ffmpeg is available
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass  # Fall back to system ffmpeg


def run_command(cmd):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result


def get_video_info(video_path):
    """Get video information using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def extract_clip(source_video: Path, output_path: Path, start_s: float, end_s: float) -> bool:
    """Extract a subset clip from the source video."""
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(source_video),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]
    result = run_command(cmd)
    return result.returncode == 0


def run_motion_detection(input_folder: Path, output_dir: Path, repo_root: Path) -> Tuple[bool, List[dict]]:
    """Run motion detection and return (success, segments)."""
    # Clean up any existing output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Run motion_cctv.py with consistent parameters for fair comparison
    cmd = [
        sys.executable,
        str(repo_root / "motion_cctv.py"),
        str(input_folder),
        "--out", str(output_dir),
        "--merge-gap", "2.5",
        "--min-duration", "3.0",
        "--pad", "1.0",
        "--warmup-seconds", "1.0",
    ]
    
    result = run_command(cmd)
    if result.returncode != 0:
        print(f"motion_cctv.py failed: {result.stderr}")
        return False, []
    
    csv_path = output_dir / "segments.csv"
    if not csv_path.exists():
        return True, []  # No segments is valid
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        segments = list(reader)
    
    return True, segments


# Default tolerance for timing comparisons (in seconds).
# This accounts for differences in warmup periods, frame timing variations,
# and padding differences between parent and subset clips.
DEFAULT_TIMING_TOLERANCE = 5.0


def segments_to_intervals(segments: List[dict]) -> List[Tuple[float, float]]:
    """Convert segment dicts to list of (start, end) tuples."""
    intervals = []
    for seg in segments:
        start = float(seg['start_seconds'])
        end = float(seg['end_seconds'])
        intervals.append((start, end))
    return intervals


def offset_intervals(intervals: List[Tuple[float, float]], offset: float) -> List[Tuple[float, float]]:
    """Add an offset to all intervals (to map subset timeline to parent timeline)."""
    return [(s + offset, e + offset) for s, e in intervals]


def interval_is_covered(interval: Tuple[float, float], covering_intervals: List[Tuple[float, float]], tolerance: float = DEFAULT_TIMING_TOLERANCE) -> bool:
    """
    Check if an interval is approximately covered by a set of covering intervals.
    
    The interval (start, end) is considered "covered" if there exists a covering 
    interval (c_start, c_end) such that:
      - c_start <= start + tolerance  (covering starts at or before target, with slack)
      - c_end >= end - tolerance       (covering ends at or after target, with slack)
    
    This provides symmetric tolerance on both ends: we allow the covering interval
    to be up to `tolerance` seconds "inside" the target interval on either end.
    
    Example with tolerance=5:
      - Target: [10, 50]
      - Covering [8, 48] passes: 8 <= 15 and 48 >= 45
      - Covering [20, 40] fails: 20 > 15 (starts too late)
    
    This allows for small timing differences due to:
    - Warmup period differences between clips
    - Frame-level timing variations  
    - Padding differences
    """
    start, end = interval
    for c_start, c_end in covering_intervals:
        # Check if this covering interval approximately contains our interval
        if c_start <= start + tolerance and c_end >= end - tolerance:
            return True
    return False


def check_subset_axiom(
    parent_intervals: List[Tuple[float, float]], 
    subset_intervals: List[Tuple[float, float]], 
    subset_offset: float,
    tolerance: float = DEFAULT_TIMING_TOLERANCE
) -> Tuple[bool, List[str]]:
    """
    Verify the subset axiom: all motion detected in the subset (when mapped to 
    parent timeline) should be covered by motion detected in the parent.
    
    Returns (success, list of error messages).
    """
    errors = []
    
    # Map subset intervals to parent timeline
    mapped_subset_intervals = offset_intervals(subset_intervals, subset_offset)
    
    print(f"    Parent intervals: {parent_intervals}")
    print(f"    Subset intervals (in subset timeline): {subset_intervals}")
    print(f"    Subset intervals (mapped to parent at offset {subset_offset}s): {mapped_subset_intervals}")
    
    for i, (sub_start, sub_end) in enumerate(mapped_subset_intervals):
        if not interval_is_covered((sub_start, sub_end), parent_intervals, tolerance):
            errors.append(
                f"Subset motion [{sub_start:.1f}s, {sub_end:.1f}s] (parent timeline) "
                f"is not covered by any parent motion segment"
            )
    
    return len(errors) == 0, errors


def test_basic_detection(repo_root: Path, example_footage: Path) -> int:
    """Test 1: Basic motion detection works correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Motion Detection")
    print("=" * 80)
    
    output_dir = example_footage / "motion_output"
    
    success, segments = run_motion_detection(example_footage, output_dir, repo_root)
    
    if not success:
        print("❌ FAILED: motion_cctv.py failed to run")
        return 1
    
    print(f"✓ motion_cctv.py completed successfully")
    print(f"  Found {len(segments)} segment(s)")
    
    if len(segments) == 0:
        print("❌ FAILED: Expected at least one motion segment")
        return 1
    
    for i, seg in enumerate(segments):
        start_s = float(seg['start_seconds'])
        end_s = float(seg['end_seconds'])
        print(f"  Segment {i+1}: {start_s:.3f}s -> {end_s:.3f}s (duration: {end_s - start_s:.3f}s)")
    
    print("✓ Basic motion detection test passed")
    return 0


def test_subset_axiom(repo_root: Path, example_footage: Path) -> int:
    """
    Test 2: Subset Axiom Verification
    
    AXIOM: If input B is a time-subset of input A, then the motion segments 
    detected in B (mapped to A's timeline) should be covered by the motion 
    segments detected in A.
    
    We create subset clips from the source video and verify this property holds.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Subset Axiom Verification")
    print("=" * 80)
    print("AXIOM: If B ⊂ A (B is a time-subset of A), then Bc ⊂ Ac")
    print("       (motion detected in B should be subset of motion detected in A)")
    print("=" * 80)
    
    # Find source video
    source_videos = list(example_footage.glob("*.mp4"))
    if not source_videos:
        print("❌ FAILED: No source video found in example_footage")
        return 1
    
    source_video = source_videos[0]
    print(f"\nSource video: {source_video.name}")
    
    # Get source video duration
    try:
        video_info = get_video_info(source_video)
        source_duration = float(video_info['format']['duration'])
        print(f"Source duration: {source_duration:.2f}s")
    except Exception as e:
        print(f"❌ FAILED: Could not probe source video: {e}")
        return 1
    
    # Create temporary directory for subset clips
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Step 1: Run motion detection on the full parent video (A)
        print("\n[Step 1] Analyzing parent video (A)...")
        parent_input_dir = tmpdir_path / "parent"
        parent_input_dir.mkdir()
        parent_output_dir = tmpdir_path / "parent_output"
        
        # Copy source video to parent directory
        parent_video = parent_input_dir / source_video.name
        shutil.copy(source_video, parent_video)
        
        parent_success, parent_segments = run_motion_detection(parent_input_dir, parent_output_dir, repo_root)
        if not parent_success:
            print("❌ FAILED: Motion detection on parent video failed")
            return 1
        
        parent_intervals = segments_to_intervals(parent_segments)
        print(f"  Parent (A) detected {len(parent_intervals)} motion segment(s)")
        for i, (s, e) in enumerate(parent_intervals):
            print(f"    Ac[{i}]: {s:.1f}s -> {e:.1f}s")
        
        if len(parent_intervals) == 0:
            print("⚠ WARNING: No motion detected in parent video, cannot verify axiom")
            print("✓ Subset axiom test skipped (no baseline motion)")
            return 0
        
        # Step 2: Create and analyze subset clips at different offsets
        # We'll create multiple subsets to thoroughly test the axiom
        
        # Define subset configurations: (start_offset, duration, name)
        # These are chosen to test different parts of the video
        subset_configs = []
        
        # Create subsets that cover different portions of the video
        if source_duration >= 60:
            # For longer videos, create meaningful subsets
            mid_point = source_duration / 2
            subset_configs = [
                (0, source_duration * 0.5, "first_half"),          # First half
                (source_duration * 0.25, source_duration * 0.5, "middle_half"),  # Middle half
            ]
        else:
            # For shorter videos, create proportional subsets
            subset_configs = [
                (0, source_duration * 0.6, "first_60pct"),
                (source_duration * 0.2, source_duration * 0.6, "middle_60pct"),
            ]
        
        all_passed = True
        
        for subset_offset, subset_duration, subset_name in subset_configs:
            subset_end = min(subset_offset + subset_duration, source_duration)
            actual_duration = subset_end - subset_offset
            
            if actual_duration < 5:  # Skip if too short
                continue
            
            print(f"\n[Step 2.{subset_name}] Creating and analyzing subset B ({subset_name})...")
            print(f"  Subset time range: {subset_offset:.1f}s -> {subset_end:.1f}s (duration: {actual_duration:.1f}s)")
            
            # Create subset input directory
            subset_input_dir = tmpdir_path / f"subset_{subset_name}"
            subset_input_dir.mkdir(exist_ok=True)
            subset_output_dir = tmpdir_path / f"subset_{subset_name}_output"
            
            subset_video = subset_input_dir / f"subset_{subset_name}.mp4"
            
            # Extract subset clip
            if not extract_clip(source_video, subset_video, subset_offset, subset_end):
                print(f"  ⚠ WARNING: Failed to extract subset clip, skipping")
                continue
            
            # Run motion detection on subset
            subset_success, subset_segments = run_motion_detection(subset_input_dir, subset_output_dir, repo_root)
            if not subset_success:
                print(f"  ❌ FAILED: Motion detection on subset failed")
                all_passed = False
                continue
            
            subset_intervals = segments_to_intervals(subset_segments)
            print(f"  Subset (B) detected {len(subset_intervals)} motion segment(s)")
            for i, (s, e) in enumerate(subset_intervals):
                print(f"    Bc[{i}]: {s:.1f}s -> {e:.1f}s (subset timeline)")
            
            # Verify subset axiom: all motion in B should be covered by motion in A
            print(f"\n  Verifying subset axiom: Bc ⊂ Ac")
            
            axiom_passed, errors = check_subset_axiom(
                parent_intervals, 
                subset_intervals, 
                subset_offset
                # Uses DEFAULT_TIMING_TOLERANCE
            )
            
            if axiom_passed:
                print(f"  ✓ Subset axiom verified for {subset_name}")
            else:
                print(f"  ❌ Subset axiom VIOLATED for {subset_name}:")
                for err in errors:
                    print(f"    - {err}")
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 80)
            print("✓ All subset axiom tests passed")
            return 0
        else:
            print("\n" + "=" * 80)
            print("❌ Some subset axiom tests failed")
            return 1


def main():
    print("=" * 80)
    print("End-to-End Test Suite for motion_cctv.py")
    print("=" * 80)
    
    # Setup paths
    repo_root = Path(__file__).parent
    example_footage = repo_root / "example_footage"
    
    if not example_footage.exists():
        print(f"❌ FAILED: Example footage directory not found: {example_footage}")
        return 1
    
    # Run all tests
    failures = 0
    
    # Test 1: Basic motion detection
    failures += test_basic_detection(repo_root, example_footage)
    
    # Test 2: Subset axiom verification
    failures += test_subset_axiom(repo_root, example_footage)
    
    # Final summary
    print("\n" + "=" * 80)
    if failures == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {failures} TEST(S) FAILED")
    print("=" * 80)
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
