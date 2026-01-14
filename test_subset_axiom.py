#!/usr/bin/env python3
"""
Subset Axiom Test for motion_cctv.py

Tests the subset axiom: C ⊂ B ⊂ A implies Cc ⊂ Bc ⊂ Ac

Given:
- A = 6min/6min.mp4 (full video)
- B = 4min/4min-subset-of-6min.mp4 (subset of A)
- C = 2min/2min-subset-of-4min.mp4 (subset of B)

Then:
- Ac = motion segments detected in A
- Bc = motion segments detected in B
- Cc = motion segments detected in C

The test verifies:
1. Every segment in Bc corresponds to a segment in Ac
2. Every segment in Cc corresponds to a segment in Bc
3. Therefore: Cc ⊂ Bc ⊂ Ac (transitive subset property)
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict


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


def parse_segments_csv(csv_path):
    """Parse segments.csv and return list of segments with all details."""
    segments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'source_file': row['source_file'],
                'start_seconds': float(row['start_seconds']),
                'end_seconds': float(row['end_seconds']),
                'duration_seconds': float(row['duration_seconds']),
                'peak_motion_ratio': float(row['peak_motion_ratio'])
            })
    return segments


def segments_overlap(seg1, seg2, tolerance=2.0):
    """
    Check if two segments overlap with tolerance.
    
    Args:
        seg1, seg2: Dicts with 'start_seconds' and 'end_seconds'
        tolerance: Tolerance in seconds for timestamp matching
    
    Returns:
        True if segments overlap (accounting for tolerance)
    """
    start1 = seg1['start_seconds']
    end1 = seg1['end_seconds']
    start2 = seg2['start_seconds']
    end2 = seg2['end_seconds']
    
    # Check if seg1 and seg2 overlap
    # Expand each segment by tolerance on both sides
    return (start1 - tolerance) < end2 and (start2 - tolerance) < end1


def infer_temporal_offset(subset_segments, superset_segments, tolerance=5.0):
    """
    Infer the temporal offset between subset and superset videos.
    
    Tries to find an offset value such that subset_segments + offset aligns with superset_segments.
    
    Returns:
        offset (float or None): The inferred offset, or None if no consistent offset found
    """
    if not subset_segments or not superset_segments:
        return 0.0
    
    # Try different offsets and see which one maximizes overlap
    # Start with offsets derived from the first segments
    candidate_offsets = []
    
    for sub_seg in subset_segments[:min(3, len(subset_segments))]:
        for super_seg in superset_segments:
            # Calculate offset that would align these segments
            offset = super_seg['start_seconds'] - sub_seg['start_seconds']
            candidate_offsets.append(offset)
    
    # Test each candidate offset
    best_offset = None
    best_match_count = 0
    
    for offset in candidate_offsets:
        match_count = 0
        for sub_seg in subset_segments:
            # Create adjusted segment
            adjusted_seg = {
                'start_seconds': sub_seg['start_seconds'] + offset,
                'end_seconds': sub_seg['end_seconds'] + offset,
                'duration_seconds': sub_seg['duration_seconds'],
                'peak_motion_ratio': sub_seg['peak_motion_ratio']
            }
            
            # Check if it overlaps with any superset segment
            for super_seg in superset_segments:
                if segments_overlap(adjusted_seg, super_seg, tolerance):
                    match_count += 1
                    break
        
        if match_count > best_match_count:
            best_match_count = match_count
            best_offset = offset
    
    # Require at least half of segments to match for a valid offset
    if best_match_count >= max(1, len(subset_segments) // 2):
        return best_offset
    
    return None


def verify_subset_relation(subset_segments, superset_segments, subset_name, superset_name, tolerance=2.0):
    """
    Verify that subset_segments ⊆ superset_segments with temporal offset inference.
    
    Returns:
        (is_subset: bool, missing_segments: list, details: str, offset: float)
    """
    missing = []
    details = []
    
    print(f"\n  Checking {subset_name} ⊆ {superset_name}:")
    print(f"    {subset_name}: {len(subset_segments)} segment(s)")
    print(f"    {superset_name}: {len(superset_segments)} segment(s)")
    
    # Infer temporal offset
    offset = infer_temporal_offset(subset_segments, superset_segments, tolerance)
    
    if offset is not None:
        print(f"    Inferred temporal offset: {offset:.2f}s")
        details.append(f"    Temporal offset: {offset:.2f}s")
    else:
        print(f"    No temporal offset inferred (direct comparison)")
        offset = 0.0
    
    for i, sub_seg in enumerate(subset_segments, 1):
        sub_start = sub_seg['start_seconds']
        sub_end = sub_seg['end_seconds']
        
        # Apply temporal offset
        adjusted_start = sub_start + offset
        adjusted_end = sub_end + offset
        adjusted_seg = {
            'start_seconds': adjusted_start,
            'end_seconds': adjusted_end,
            'duration_seconds': sub_seg['duration_seconds'],
            'peak_motion_ratio': sub_seg['peak_motion_ratio']
        }
        
        # Check if this adjusted segment overlaps with any segment in superset
        found = False
        for j, super_seg in enumerate(superset_segments, 1):
            if segments_overlap(adjusted_seg, super_seg, tolerance):
                super_start = super_seg['start_seconds']
                super_end = super_seg['end_seconds']
                details.append(
                    f"    ✓ {subset_name}[{i}] [{sub_start:.1f}s-{sub_end:.1f}s] "
                    f"→ [{adjusted_start:.1f}s-{adjusted_end:.1f}s] "
                    f"⊆ {superset_name}[{j}] [{super_start:.1f}s-{super_end:.1f}s]"
                )
                found = True
                break
        
        if not found:
            missing.append(sub_seg)
            details.append(
                f"    ✗ {subset_name}[{i}] [{sub_start:.1f}s-{sub_end:.1f}s] "
                f"→ [{adjusted_start:.1f}s-{adjusted_end:.1f}s] "
                f"NOT FOUND in {superset_name}"
            )
    
    return len(missing) == 0, missing, "\n".join(details), offset


def process_video(video_dir, output_dir, video_name):
    """Process a video directory and return the segments."""
    print(f"\n{'=' * 80}")
    print(f"Processing {video_name}")
    print('=' * 80)
    
    # Clean up output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run motion detection
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "motion_cctv.py"),
        str(video_dir),
        "--out", str(output_dir)
    ]
    
    result = run_command(cmd)
    if result.returncode != 0:
        print(f"❌ FAILED: {video_name} processing failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    print(f"✓ {video_name} processed successfully")
    
    # Parse segments
    csv_path = output_dir / "segments.csv"
    if not csv_path.exists():
        print(f"❌ FAILED: CSV not found for {video_name}: {csv_path}")
        return None
    
    segments = parse_segments_csv(csv_path)
    print(f"✓ Found {len(segments)} segment(s) in {video_name}")
    for i, seg in enumerate(segments, 1):
        print(f"  [{i}] {seg['start_seconds']:.2f}s - {seg['end_seconds']:.2f}s "
              f"(duration: {seg['duration_seconds']:.2f}s, peak: {seg['peak_motion_ratio']:.4f})")
    
    return segments


def main():
    print("=" * 80)
    print("SUBSET AXIOM TEST: C ⊂ B ⊂ A implies Cc ⊆ Bc ⊆ Ac")
    print("=" * 80)
    
    repo_root = Path(__file__).parent
    example_footage = repo_root / "example_footage"
    test_output = repo_root / "test_subset_output"
    
    # Define video paths
    video_a_dir = example_footage / "6min"
    video_b_dir = example_footage / "4min"
    video_c_dir = example_footage / "2min"
    
    # Check videos exist
    video_a = video_a_dir / "6min.mp4"
    video_b = video_b_dir / "4min-subset-of-6min.mp4"
    video_c = video_c_dir / "2min-subset-of-4min.mp4"
    
    for video, name in [(video_a, "A (6min)"), (video_b, "B (4min)"), (video_c, "C (2min)")]:
        if not video.exists():
            print(f"❌ FAILED: Video {name} not found: {video}")
            return 1
    
    print(f"\n✓ All videos found:")
    print(f"  A = {video_a}")
    print(f"  B = {video_b}")
    print(f"  C = {video_c}")
    
    # Clean up test output directory
    if test_output.exists():
        shutil.rmtree(test_output)
    test_output.mkdir(parents=True, exist_ok=True)
    
    # Process all three videos
    segments_a = process_video(video_a_dir, test_output / "output_a", "A (6min)")
    if segments_a is None:
        return 1
    
    segments_b = process_video(video_b_dir, test_output / "output_b", "B (4min)")
    if segments_b is None:
        return 1
    
    segments_c = process_video(video_c_dir, test_output / "output_c", "C (2min)")
    if segments_c is None:
        return 1
    
    # Verify subset relations
    print("\n" + "=" * 80)
    print("VERIFYING SUBSET AXIOM")
    print("=" * 80)
    
    tolerance = 3.0  # 3 second tolerance for timestamp matching
    
    # Test 1: Cc ⊆ Bc
    print("\n[Test 1] Verifying Cc ⊆ Bc")
    is_cc_subset_bc, missing_cc, details_cc_bc, offset_c_to_b = verify_subset_relation(
        segments_c, segments_b, "Cc", "Bc", tolerance
    )
    print(details_cc_bc)
    
    # Test 2: Bc ⊆ Ac
    print("\n[Test 2] Verifying Bc ⊆ Ac")
    is_bc_subset_ac, missing_bc, details_bc_ac, offset_bc_ac = verify_subset_relation(
        segments_b, segments_a, "Bc", "Ac", tolerance
    )
    print(details_bc_ac)
    
    # Test 3: By transitivity, Cc ⊆ Ac
    print("\n[Test 3] Verifying Cc ⊆ Ac (transitivity)")
    is_cc_subset_ac, missing_cc_ac, details_cc_ac, offset_cc_ac = verify_subset_relation(
        segments_c, segments_a, "Cc", "Ac", tolerance
    )
    print(details_cc_ac)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    all_passed = True
    
    if is_cc_subset_bc:
        print("✅ Test 1 PASSED: Cc ⊆ Bc")
    else:
        print(f"❌ Test 1 FAILED: Cc ⊄ Bc ({len(missing_cc)} segment(s) not found)")
        all_passed = False
    
    if is_bc_subset_ac:
        print("✅ Test 2 PASSED: Bc ⊆ Ac")
    else:
        print(f"❌ Test 2 FAILED: Bc ⊄ Ac ({len(missing_bc)} segment(s) not found)")
        all_passed = False
    
    if is_cc_subset_ac:
        print("✅ Test 3 PASSED: Cc ⊆ Ac (transitivity)")
    else:
        print(f"❌ Test 3 FAILED: Cc ⊄ Ac ({len(missing_cc_ac)} segment(s) not found)")
        all_passed = False
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✅ SUBSET AXIOM VERIFIED: Cc ⊆ Bc ⊆ Ac")
        print("=" * 80)
        print("\nThe algorithm correctly satisfies the subset axiom!")
        print(f"  - Ac: {len(segments_a)} segment(s)")
        print(f"  - Bc: {len(segments_b)} segment(s)")
        print(f"  - Cc: {len(segments_c)} segment(s)")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ SUBSET AXIOM VIOLATED")
        print("=" * 80)
        print("\nThe algorithm does NOT satisfy the subset axiom.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
