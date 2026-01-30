#!/usr/bin/env python3
"""
shuffle_concat_seam.py — Shuffle and concatenate video clips with seam frame matching.

Usage:
    python shuffle_concat_seam.py /path/to/videos output.mp4

Features:
- Reads all video files in the target folder
- Shuffles them into a random order
- For each successive clip, trims frames from the beginning to find the best
  matching frame (closest to the last frame of the preceding clip)
- Creates a smooth continuous sequence by minimizing visual jumps between clips
- Uses industry-standard frame comparison (Gaussian blur, grayscale, MSE)

Algorithm:
1. Get the last frame of the preceding clip ("needle")
2. Examine frames in the first N seconds of the successive clip ("haystack")
3. Compare each haystack frame to the needle using pixel difference
4. Select the frame with the minimum difference as the trim start point
5. Concatenate all clips with the determined trim points

Requires:
- ffmpeg + ffprobe (on PATH or pass --ffmpeg/--ffprobe)
- OpenCV (pip install opencv-python)
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Any

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    np = None  # type: ignore

# ---------------------------
# Safe logging (Windows codepages)
# ---------------------------

def log(msg: str = "", *, flush: bool = True) -> None:
    try:
        print(msg, flush=flush)
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", "replace").decode("utf-8", "replace")
        print(safe, flush=flush)

# ---------------------------
# FFmpeg/FFprobe utilities
# ---------------------------

def find_exe(name: str) -> Optional[str]:
    """Find executable on PATH or common locations (Windows)."""
    p = shutil.which(name)
    if p:
        return p
    if os.name == "nt":
        if name.lower() == "ffmpeg":
            candidates = [
                r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            ]
        else:
            candidates = [
                r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
                r"C:\ffmpeg\bin\ffprobe.exe",
                r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
            ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None

def run_capture(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run command and capture output."""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

def require_ffmpeg(custom_ffmpeg: Optional[str], custom_ffprobe: Optional[str]) -> Tuple[str, str]:
    """Find and validate ffmpeg/ffprobe executables."""
    ffmpeg = custom_ffmpeg or find_exe("ffmpeg")
    ffprobe = custom_ffprobe or find_exe("ffprobe")
    if not ffmpeg or not os.path.exists(ffmpeg):
        raise FileNotFoundError("ffmpeg not found. Install FFmpeg or pass --ffmpeg path_to_ffmpeg.exe")
    if not ffprobe or not os.path.exists(ffprobe):
        raise FileNotFoundError("ffprobe not found. Install FFmpeg or pass --ffprobe path_to_ffprobe.exe")

    log(f"[env] ffmpeg : {ffmpeg}")
    log(f"[env] ffprobe: {ffprobe}")

    for exe in (ffmpeg, ffprobe):
        p = run_capture([exe, "-version"])
        if p.returncode != 0:
            raise RuntimeError(f"Failed to run {exe}:\n{p.stderr}")

    return ffmpeg, ffprobe

def ffprobe_json(ffprobe_exe: str, path: Path) -> dict:
    """Get ffprobe JSON output for a video file."""
    cmd = [ffprobe_exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    p = run_capture(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{p.stderr}")
    return json.loads(p.stdout)

# ---------------------------
# Video file discovery
# ---------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg"}

def find_video_files(folder: Path, recursive: bool = False) -> List[Path]:
    """Find all video files in folder (optionally recursive)."""
    files = []
    if recursive:
        for root, _, filenames in os.walk(folder):
            for name in filenames:
                p = Path(root) / name
                if p.suffix.lower() in VIDEO_EXTENSIONS:
                    files.append(p)
    else:
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(item)
    return sorted(files)

# ---------------------------
# Video specs extraction
# ---------------------------

def get_video_specs(ffprobe_exe: str, path: Path) -> dict:
    """Extract video codec, resolution, framerate, and duration from first video stream."""
    info = ffprobe_json(ffprobe_exe, path)
    
    duration = 0.0
    if "format" in info and "duration" in info["format"]:
        try:
            duration = float(info["format"]["duration"])
        except (ValueError, TypeError):
            pass
    
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            codec = stream.get("codec_name", "unknown")
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            
            # Parse framerate
            fr = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            fps = 30.0
            if fr and fr != "0/0":
                try:
                    num, den = fr.split("/")
                    if float(den) != 0:
                        fps = float(num) / float(den)
                except (ValueError, ZeroDivisionError):
                    pass
            
            return {
                "codec": codec,
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
            }
    
    raise ValueError(f"No video stream found in {path}")

def specs_match(specs1: dict, specs2: dict, tolerance: float = 0.1) -> bool:
    """Check if two video specs are compatible (within tolerance)."""
    if specs1["width"] != specs2["width"] or specs1["height"] != specs2["height"]:
        return False
    
    # Allow small FPS variation
    fps_diff = abs(specs1["fps"] - specs2["fps"])
    if fps_diff > tolerance:
        return False
    
    return True

# ---------------------------
# Frame extraction and comparison
# ---------------------------

def extract_frame_at_time(ffmpeg_exe: str, video_path: Path, time_sec: float, output_path: Path) -> bool:
    """Extract a single frame at the specified time."""
    cmd = [
        ffmpeg_exe,
        "-ss", str(time_sec),
        "-i", str(video_path),
        "-vframes", "1",
        "-y",
        str(output_path)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0 and output_path.exists()

def get_last_frame(ffmpeg_exe: str, ffprobe_exe: str, video_path: Path, tmpdir: Path) -> Optional[Any]:
    """Extract and return the last frame of a video as a numpy array."""
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV is required for frame comparison. Install with: pip install opencv-python")
    
    specs = get_video_specs(ffprobe_exe, video_path)
    duration = specs["duration"]
    
    if duration <= 0:
        log(f"  WARNING: Could not determine duration for {video_path.name}")
        return None
    
    # Get the last frame (slightly before end to avoid edge cases)
    last_frame_time = max(0, duration - 0.1)
    frame_path = tmpdir / "last_frame.png"
    
    if not extract_frame_at_time(ffmpeg_exe, video_path, last_frame_time, frame_path):
        log(f"  WARNING: Could not extract last frame from {video_path.name}")
        return None
    
    frame = cv2.imread(str(frame_path))
    if frame is None:
        log(f"  WARNING: Could not read extracted frame from {video_path.name}")
        return None
    
    return frame

def preprocess_frame_for_comparison(frame: Any, blur_size: int = 5) -> Any:
    """Preprocess frame for comparison: grayscale + Gaussian blur."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    return blurred

def compute_frame_difference(frame1: Any, frame2: Any) -> float:
    """Compute the mean squared error (MSE) between two frames.
    
    Lower values indicate more similar frames.
    """
    # Resize to match dimensions if different
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Compute MSE
    diff = frame1.astype(np.float64) - frame2.astype(np.float64)
    mse = np.mean(diff ** 2)
    return mse

def find_best_matching_frame(
    ffmpeg_exe: str,
    needle_frame: Any,
    video_path: Path,
    haystack_duration: float,
    fps: float,
    tmpdir: Path
) -> Tuple[float, float]:
    """Find the frame in the first haystack_duration seconds of video that best matches needle_frame.
    
    Returns:
        Tuple of (best_time_seconds, best_mse_score)
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV is required for frame comparison. Install with: pip install opencv-python")
    
    # Preprocess needle frame
    needle_processed = preprocess_frame_for_comparison(needle_frame)
    
    # Calculate frame interval based on fps
    frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
    
    best_time = 0.0
    best_mse = float('inf')
    
    # Sample frames in the haystack duration
    current_time = 0.0
    frame_count = 0
    
    while current_time < haystack_duration:
        frame_path = tmpdir / f"haystack_frame_{frame_count:04d}.png"
        
        if extract_frame_at_time(ffmpeg_exe, video_path, current_time, frame_path):
            haystack_frame = cv2.imread(str(frame_path))
            if haystack_frame is not None:
                haystack_processed = preprocess_frame_for_comparison(haystack_frame)
                mse = compute_frame_difference(needle_processed, haystack_processed)
                
                if mse < best_mse:
                    best_mse = mse
                    best_time = current_time
            
            # Clean up frame file
            try:
                frame_path.unlink()
            except OSError:
                pass
        
        current_time += frame_interval
        frame_count += 1
    
    return best_time, best_mse

# ---------------------------
# Video re-encoding and trimming
# ---------------------------

def reencode_video(ffmpeg_exe: str, input_path: Path, output_path: Path, target_specs: dict, start_time: float = 0.0) -> bool:
    """Re-encode video to match target specs, optionally starting from a specific time."""
    log(f"  Re-encoding {input_path.name} (start={start_time:.3f}s)...")
    
    cmd = [
        ffmpeg_exe,
        "-ss", str(start_time),
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-s", f"{target_specs['width']}x{target_specs['height']}",
        "-r", str(target_specs['fps']),
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        str(output_path)
    ]
    
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        log(f"  ERROR: Re-encoding failed:\n{p.stderr}")
        return False
    
    log(f"  Re-encoded successfully")
    return True

def trim_video_streamcopy(ffmpeg_exe: str, input_path: Path, output_path: Path, start_time: float) -> bool:
    """Trim video using stream copy (fast, no re-encoding) starting from a specific time."""
    log(f"  Trimming {input_path.name} from {start_time:.3f}s (stream copy)...")
    
    cmd = [
        ffmpeg_exe,
        "-ss", str(start_time),
        "-i", str(input_path),
        "-c", "copy",
        "-y",
        str(output_path)
    ]
    
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        log(f"  ERROR: Trimming failed:\n{p.stderr}")
        return False
    
    log(f"  Trimmed successfully")
    return True

# ---------------------------
# Main concatenation with seam matching
# ---------------------------

def shuffle_and_concatenate_videos(
    ffmpeg_exe: str,
    ffprobe_exe: str,
    video_files: List[Path],
    output_path: Path,
    haystack_duration: float = 1.0,
    seed: Optional[int] = None
) -> None:
    """Shuffle videos and concatenate with seam frame matching."""
    if not video_files:
        raise ValueError("No video files found to concatenate")
    
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV is required for frame comparison. Install with: pip install opencv-python")
    
    # Shuffle the files
    shuffled_files = video_files.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled_files)
    
    log(f"\n[shuffle] Shuffled {len(shuffled_files)} video files")
    for i, f in enumerate(shuffled_files):
        log(f"  {i+1}. {f.name}")
    
    # Get specs from first file
    log(f"\n[specs] Detecting specs from first file: {shuffled_files[0].name}")
    target_specs = get_video_specs(ffprobe_exe, shuffled_files[0])
    log(f"[specs] Target specs: {target_specs['width']}x{target_specs['height']} @ {target_specs['fps']:.2f} fps")
    
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        concat_list_path = tmpdir_path / "concat_list.txt"
        
        processed_files = []
        prev_last_frame = None
        
        for i, video_file in enumerate(shuffled_files):
            log(f"\n[{i+1}/{len(shuffled_files)}] Processing {video_file.name}")
            
            # Get specs for this file
            try:
                file_specs = get_video_specs(ffprobe_exe, video_file)
            except Exception as e:
                log(f"  WARNING: Could not get specs, skipping: {e}")
                continue
            
            # Determine trim start time
            trim_start = 0.0
            
            if i > 0 and prev_last_frame is not None:
                # Find best matching frame in haystack
                log(f"  Finding best seam match (haystack={haystack_duration:.1f}s)...")
                
                trim_start, mse = find_best_matching_frame(
                    ffmpeg_exe,
                    prev_last_frame,
                    video_file,
                    haystack_duration,
                    file_specs["fps"],
                    tmpdir_path
                )
                
                log(f"  Best match at {trim_start:.3f}s (MSE={mse:.2f})")
            
            # Determine output file path
            temp_output = tmpdir_path / f"processed_{i:04d}.mp4"
            
            # Check if we need to re-encode or can stream copy
            needs_reencode = not specs_match(target_specs, file_specs)
            
            if needs_reencode:
                log(f"  Specs differ: {file_specs['width']}x{file_specs['height']} @ {file_specs['fps']:.2f} fps")
                if reencode_video(ffmpeg_exe, video_file, temp_output, target_specs, trim_start):
                    processed_files.append(temp_output)
                else:
                    log(f"  WARNING: Skipping file due to re-encoding failure")
                    continue
            elif trim_start > 0:
                # Need to trim, but specs match - try stream copy
                if trim_video_streamcopy(ffmpeg_exe, video_file, temp_output, trim_start):
                    processed_files.append(temp_output)
                else:
                    log(f"  WARNING: Skipping file due to trimming failure")
                    continue
            else:
                # No trimming needed, specs match - use original
                log(f"  Using original file (no trimming needed)")
                processed_files.append(video_file)
            
            # Get the last frame of this processed file for the next iteration
            processed_file = processed_files[-1]
            prev_last_frame = get_last_frame(ffmpeg_exe, ffprobe_exe, processed_file, tmpdir_path)
            
            if prev_last_frame is None:
                log(f"  WARNING: Could not extract last frame for seam matching")
        
        if not processed_files:
            raise ValueError("No video files could be processed successfully")
        
        log(f"\n[concat] Successfully processed {len(processed_files)}/{len(shuffled_files)} files")
        
        # Create concat list file
        log(f"[concat] Creating concat list...")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for video_file in processed_files:
                abs_path = video_file.resolve()
                escaped = str(abs_path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        
        # Concatenate using concat demuxer
        log(f"[concat] Concatenating {len(processed_files)} files into {output_path}")
        cmd = [
            ffmpeg_exe,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            "-y",
            str(output_path)
        ]
        
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Concatenation failed:\n{p.stderr}")
        
        log(f"\n[concat] ✓ Successfully created {output_path}")
        
        # Show output file info
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        log(f"[concat] Output size: {output_size_mb:.2f} MB")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Shuffle and concatenate video clips with seam frame matching for smoother transitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python shuffle_concat_seam.py /path/to/videos output.mp4
  python shuffle_concat_seam.py /path/to/videos output.mp4 --haystack-duration 2.0
  python shuffle_concat_seam.py /path/to/videos output.mp4 --seed 42
  python shuffle_concat_seam.py /path/to/videos output.mp4 --recursive

Algorithm:
  For each successive clip, the script examines frames in the first N seconds
  (haystack) and finds the frame that best matches the last frame of the 
  preceding clip (needle). The successive clip is trimmed to start at this
  best-matching frame, creating a smoother visual transition.
        """
    )
    
    ap.add_argument("input_dir", type=str, help="Directory containing video files to concatenate")
    ap.add_argument("output_file", type=str, help="Output file path (e.g., output.mp4)")
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories for video files")
    ap.add_argument("--haystack-duration", type=float, default=1.0,
                    help="Duration in seconds to search for best matching frame (default: 1.0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible shuffling (default: random)")
    ap.add_argument("--ffmpeg", type=str, default=None, help="Path to ffmpeg executable")
    ap.add_argument("--ffprobe", type=str, default=None, help="Path to ffprobe executable")
    
    args = ap.parse_args()
    
    # Check OpenCV availability
    if not HAS_OPENCV:
        log("ERROR: OpenCV is required for frame comparison.")
        log("Install with: pip install opencv-python")
        sys.exit(1)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        log(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        log(f"ERROR: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Validate output file
    output_file = Path(args.output_file)
    if output_file.exists():
        log(f"WARNING: Output file already exists and will be overwritten: {output_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find ffmpeg/ffprobe
    try:
        ffmpeg_exe, ffprobe_exe = require_ffmpeg(args.ffmpeg, args.ffprobe)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
    
    # Find video files
    log(f"\n[scan] Scanning {input_dir} (recursive={args.recursive})")
    video_files = find_video_files(input_dir, recursive=args.recursive)
    
    if not video_files:
        log(f"ERROR: No video files found in {input_dir}")
        sys.exit(1)
    
    log(f"[scan] Found {len(video_files)} video files")
    
    # Shuffle and concatenate
    try:
        shuffle_and_concatenate_videos(
            ffmpeg_exe,
            ffprobe_exe,
            video_files,
            output_file,
            haystack_duration=args.haystack_duration,
            seed=args.seed
        )
    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    log("\n[done] ✓ Shuffle and concatenation complete!")

if __name__ == "__main__":
    main()
