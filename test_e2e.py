#!/usr/bin/env python3
"""
End-to-end test for motion_cctv.py

Tests basic functionality with the 2min example footage.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def main():
    print("=" * 80)
    print("End-to-End Test for motion_cctv.py")
    print("=" * 80)
    
    # Setup paths
    repo_root = Path(__file__).parent
    test_video_dir = repo_root / "example_footage" / "2min"
    output_dir = repo_root / "test_e2e_output"
    
    # Clean up any existing output
    if output_dir.exists():
        print(f"\nCleaning up existing output: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Verify test video exists
    test_video = test_video_dir / "2min-subset-of-4min.mp4"
    if not test_video.exists():
        print(f"❌ FAILED: Test video not found: {test_video}")
        return 1
    
    print(f"\n✓ Test video found: {test_video}")
    
    # Step 1: Run motion_cctv.py
    print("\n[Step 1] Running motion_cctv.py on 2min example footage...")
    cmd = [
        sys.executable,
        str(repo_root / "motion_cctv.py"),
        str(test_video_dir),
        "--out", str(output_dir)
    ]
    
    result = run_command(cmd)
    if result.returncode != 0:
        print(f"❌ FAILED: motion_cctv.py failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return 1
    
    print("✓ motion_cctv.py completed successfully")
    
    # Step 2: Verify output structure
    print("\n[Step 2] Verifying output structure...")
    
    if not output_dir.exists():
        print(f"❌ FAILED: Output directory not created: {output_dir}")
        return 1
    
    csv_path = output_dir / "segments.csv"
    if not csv_path.exists():
        print(f"❌ FAILED: CSV file not created: {csv_path}")
        return 1
    
    print(f"✓ Output directory created: {output_dir}")
    print(f"✓ CSV file created: {csv_path}")
    
    # Step 3: Parse CSV
    print("\n[Step 3] Parsing segments.csv...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        segments = list(reader)
    
    print(f"Found {len(segments)} segment(s) in CSV")
    
    if len(segments) == 0:
        print(f"⚠️  WARNING: No segments detected (video may not contain motion)")
        return 0
    
    # Display segments
    for i, seg in enumerate(segments, 1):
        start_s = float(seg['start_seconds'])
        end_s = float(seg['end_seconds'])
        duration_s = float(seg['duration_seconds'])
        peak = float(seg['peak_motion_ratio'])
        print(f"  Segment {i}: {start_s:.2f}s -> {end_s:.2f}s (duration: {duration_s:.2f}s, peak: {peak:.4f})")
    
    # Step 4: Verify at least one clip file exists
    print("\n[Step 4] Verifying output clip files...")
    
    clips_found = 0
    for seg in segments:
        clip_path_rel = seg['clip_path']
        clip_path = output_dir / clip_path_rel
        
        if clip_path.exists():
            clips_found += 1
            print(f"✓ Clip file exists: {clip_path}")
            
            # Verify with ffprobe
            try:
                video_info = get_video_info(clip_path)
                clip_duration = float(video_info['format']['duration'])
                video_streams = [s for s in video_info.get('streams', []) if s.get('codec_type') == 'video']
                if video_streams:
                    codec = video_streams[0]['codec_name']
                    width = video_streams[0]['width']
                    height = video_streams[0]['height']
                    print(f"  Duration: {clip_duration:.2f}s, Codec: {codec}, Resolution: {width}x{height}")
            except Exception as e:
                print(f"  ⚠️  Could not probe clip: {e}")
        else:
            print(f"✗ Clip file missing: {clip_path}")
    
    if clips_found == 0:
        print(f"❌ FAILED: No clip files found")
        return 1
    
    print(f"✓ Found {clips_found}/{len(segments)} clip file(s)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print(f"Summary:")
    print(f"  - Segments detected: {len(segments)}")
    print(f"  - Clips generated: {clips_found}")
    print(f"  - Output location: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
