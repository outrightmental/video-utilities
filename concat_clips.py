#!/usr/bin/env python3
"""
concat_clips.py — Concatenate all video clips recursively into one file.

Usage:
    python concat_clips.py /path/to/videos output.mp4

Features:
- Recursively finds all video files in the input directory
- Preserves video specs (codec, resolution, framerate) from the first clip
- Re-encodes non-conformant clips to match the first clip's specs
- Concatenates all clips using FFmpeg concat demuxer
- Handles various video formats (mp4, avi, mkv, mov, etc.)

Requires:
- ffmpeg + ffprobe (on PATH or pass --ffmpeg/--ffprobe)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

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

def find_video_files(folder: Path, recursive: bool = True) -> List[Path]:
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
    """Extract video codec, resolution, framerate from first video stream."""
    info = ffprobe_json(ffprobe_exe, path)
    
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
# Video re-encoding
# ---------------------------

def reencode_video(ffmpeg_exe: str, input_path: Path, output_path: Path, target_specs: dict) -> bool:
    """Re-encode video to match target specs."""
    log(f"  Re-encoding {input_path.name} to match target specs...")
    
    cmd = [
        ffmpeg_exe,
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

# ---------------------------
# Concatenation
# ---------------------------

def concatenate_videos(ffmpeg_exe: str, ffprobe_exe: str, video_files: List[Path], output_path: Path) -> None:
    """Concatenate all video files into one output file."""
    if not video_files:
        raise ValueError("No video files found to concatenate")
    
    log(f"\n[concat] Found {len(video_files)} video files")
    
    # Get specs from first file
    log(f"[concat] Detecting specs from first file: {video_files[0].name}")
    target_specs = get_video_specs(ffprobe_exe, video_files[0])
    log(f"[concat] Target specs: {target_specs['width']}x{target_specs['height']} @ {target_specs['fps']:.2f} fps, codec={target_specs['codec']}")
    
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        concat_list_path = tmpdir_path / "concat_list.txt"
        
        # Process each video file
        processed_files = []
        for i, video_file in enumerate(video_files):
            log(f"\n[{i+1}/{len(video_files)}] Processing {video_file.name}")
            
            # Check specs
            try:
                file_specs = get_video_specs(ffprobe_exe, video_file)
            except Exception as e:
                log(f"  WARNING: Could not get specs, skipping: {e}")
                continue
            
            # If specs match, use original file
            if specs_match(target_specs, file_specs):
                log(f"  Specs match, using original file")
                processed_files.append(video_file)
            else:
                # Re-encode to match target specs
                log(f"  Specs differ: {file_specs['width']}x{file_specs['height']} @ {file_specs['fps']:.2f} fps")
                temp_output = tmpdir_path / f"reencoded_{i:04d}.mp4"
                if reencode_video(ffmpeg_exe, video_file, temp_output, target_specs):
                    processed_files.append(temp_output)
                else:
                    log(f"  WARNING: Skipping file due to re-encoding failure")
        
        if not processed_files:
            raise ValueError("No video files could be processed successfully")
        
        log(f"\n[concat] Successfully processed {len(processed_files)}/{len(video_files)} files")
        
        # Create concat list file
        log(f"[concat] Creating concat list...")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for video_file in processed_files:
                # FFmpeg concat demuxer requires absolute paths or proper escaping
                # Use absolute paths to avoid issues
                abs_path = video_file.resolve()
                # Escape single quotes in path
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
        description="Concatenate all video clips recursively into one file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python concat_clips.py /path/to/videos output.mp4
  python concat_clips.py /path/to/videos output.mp4 --no-recursive
  python concat_clips.py /path/to/videos output.mp4 --ffmpeg /custom/path/ffmpeg
        """
    )
    
    ap.add_argument("input_dir", type=str, help="Directory containing video files to concatenate")
    ap.add_argument("output_file", type=str, help="Output file path (e.g., output.mp4)")
    ap.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    ap.add_argument("--ffmpeg", type=str, default=None, help="Path to ffmpeg executable")
    ap.add_argument("--ffprobe", type=str, default=None, help="Path to ffprobe executable")
    
    args = ap.parse_args()
    
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
    recursive = not args.no_recursive
    log(f"\n[scan] Scanning {input_dir} (recursive={recursive})")
    video_files = find_video_files(input_dir, recursive=recursive)
    
    if not video_files:
        log(f"ERROR: No video files found in {input_dir}")
        sys.exit(1)
    
    # Concatenate videos
    try:
        concatenate_videos(ffmpeg_exe, ffprobe_exe, video_files, output_file)
    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    log("\n[done] ✓ Concatenation complete!")

if __name__ == "__main__":
    main()
