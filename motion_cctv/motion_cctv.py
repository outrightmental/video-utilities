#!/usr/bin/env python3
"""
motion_cctv.py — Motion-only clip extraction optimized for security camera footage.

Detection: OpenCV background subtraction (MOG2) + morphology + sustained-motion gating.
Cutting: ffmpeg segment extraction with safe defaults for CCTV weird audio codecs.

Outputs:
- motion_output/<video_stem>/*.mp4 clips
- motion_output/segments.csv with ALL segments + clip status/errors
- motion_output/_logs/ffmpeg logs

Requires:
- ffmpeg + ffprobe (on PATH or pass --ffmpeg/--ffprobe)
- pip install opencv-python
- For GPU acceleration: pip install opencv-contrib-python (with CUDA support)
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
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
# GPU/CUDA Detection
# ---------------------------

def check_cuda_available() -> bool:
    """Check if CUDA is available for OpenCV."""
    try:
        import cv2
        if hasattr(cv2, 'cuda'):
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                return True
    except Exception:
        pass
    return False

def get_cuda_unavailable_reason() -> str:
    """Determine why CUDA is not available and provide guidance."""
    try:
        import cv2
        if not hasattr(cv2, 'cuda'):
            return "OpenCV was built without CUDA support. Install a CUDA-enabled build to use GPU acceleration."
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count == 0:
                # Check if this is because OpenCV wasn't built with CUDA or no devices
                # Standard pip opencv-python has the cuda module but it's non-functional
                return "OpenCV was built without CUDA support. To enable: build OpenCV from source with CUDA, or use a CUDA-enabled package."
        except Exception:
            return "OpenCV CUDA module present but non-functional. Install a CUDA-enabled OpenCV build."
        return "CUDA is available but could not be initialized."
    except Exception as e:
        return f"Error checking CUDA: {str(e)}"

def get_gpu_encoder_for_ffmpeg() -> Optional[str]:
    """Detect available GPU encoder for FFmpeg."""
    # Try to detect GPU encoders and verify they work
    result = run_capture([FFMPEG_EXE, "-hide_banner", "-encoders"])
    if result.returncode == 0:
        output = result.stdout.lower()
        
        # Check for various GPU encoders in order of preference
        candidates = []
        if "h264_nvenc" in output:
            candidates.append("h264_nvenc")
        if "h264_qsv" in output:  # Intel Quick Sync
            candidates.append("h264_qsv")
        if "h264_vaapi" in output:  # VA-API (Linux)
            candidates.append("h264_vaapi")
        if "h264_videotoolbox" in output:  # macOS
            candidates.append("h264_videotoolbox")
        
        # Verify the encoder can actually be initialized
        for encoder in candidates:
            test_cmd = [
                FFMPEG_EXE, "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                "-c:v", encoder, "-f", "null", "-"
            ]
            test_result = run_capture(test_cmd)
            if test_result.returncode == 0:
                return encoder
    
    return None

# ---------------------------
# FFmpeg discovery + helpers
# ---------------------------

FFMPEG_EXE = "ffmpeg"
FFPROBE_EXE = "ffprobe"

def find_exe(name: str) -> Optional[str]:
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
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

def require_ffmpeg(custom_ffmpeg: Optional[str], custom_ffprobe: Optional[str]) -> Tuple[str, str]:
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

def ffprobe_json(path: Path) -> dict:
    cmd = [FFPROBE_EXE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    p = run_capture(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{p.stderr}")
    return json.loads(p.stdout)

def get_duration_fps(path: Path) -> Tuple[float, float]:
    info = ffprobe_json(path)
    dur = float(info["format"]["duration"])
    fps = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            fr = s.get("avg_frame_rate") or s.get("r_frame_rate")
            if fr and fr != "0/0":
                num, den = fr.split("/")
                fps = float(num) / float(den) if float(den) != 0 else None
                break
    if not fps or fps <= 0:
        fps = 30.0
    return dur, fps

def fmt_time(t: float) -> str:
    return f"{t:.3f}"

# ---------------------------
# OpenCV motion detection (CCTV-optimized)
# ---------------------------

def detect_motion_segments_opencv(
    video_path: Path,
    *,
    duration_s: float,
    fps: float,
    downscale_width: int,
    warmup_seconds: float,
    motion_ratio_threshold: float,
    min_motion_frames: int,
    min_still_frames: int,
    min_contour_area: int,
    merge_gap_s: float,
    min_duration_s: float,
    pad_s: float,
    roi: Optional[Tuple[float, float, float, float]],
    progress_every_s: float = 2.0,
    use_gpu: bool = True,
    frame_skip: int = 1,
    hwaccel_decode: bool = False,
) -> List[Tuple[float, float, float]]:
    """
    Returns list of (start_s, end_s, peak_motion_ratio)
    
    If use_gpu=True and CUDA is available, uses GPU-accelerated operations.
    If frame_skip > 1, processes every Nth frame (e.g., frame_skip=2 processes every 2nd frame).
    If hwaccel_decode=True, attempts to use hardware-accelerated video decoding.
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        raise RuntimeError("OpenCV not installed. Run: pip install opencv-python") from e
    
    # Verify cv2 has required attributes (detects broken/incomplete installations)
    required_attrs = ['VideoCapture', 'createBackgroundSubtractorMOG2', 'cvtColor',
                      'GaussianBlur', 'threshold', 'findContours', 'contourArea']
    missing_attrs = [attr for attr in required_attrs if not hasattr(cv2, attr)]
    
    if missing_attrs:
        raise RuntimeError(
            f"OpenCV installation is incomplete or broken. Missing attributes: {', '.join(missing_attrs)}\n"
            "Try reinstalling: pip uninstall opencv-python opencv-contrib-python -y && "
            "pip install opencv-python"
        )

    # Check GPU availability
    gpu_available = use_gpu and check_cuda_available()
    if use_gpu and not gpu_available:
        log("  [GPU] CUDA not available, falling back to CPU")
    elif gpu_available:
        log(f"  [GPU] CUDA enabled with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Enable hardware-accelerated video decoding if requested
    # OpenCV can use various hardware acceleration backends depending on build and platform
    if hwaccel_decode and gpu_available:
        try:
            # Try to enable hardware acceleration via backend hints
            # Note: This depends on how OpenCV was built and platform support
            # On Windows with CUDA: use FFMPEG backend with CUDA
            # On Linux: FFMPEG backend may use VAAPI/VDPAU automatically
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            log("  [HW Decode] Hardware-accelerated decoding enabled (if supported by OpenCV build)")
        except Exception as e:
            log(f"  [HW Decode] Could not enable hardware decode: {e}")
            log("  [HW Decode] Falling back to software decode")
    elif hwaccel_decode and not gpu_available:
        log("  [HW Decode] Hardware decode requested but CUDA not available, using software decode")

    # Try to get properties from OpenCV too
    cv_fps = cap.get(cv2.CAP_PROP_FPS)
    if cv_fps and cv_fps > 1:
        fps = float(cv_fps)

    # Adjust effective FPS if we're skipping frames
    effective_fps = fps / frame_skip if frame_skip > 1 else fps
    if frame_skip > 1:
        log(f"  [Frame Skip] Processing every {frame_skip} frame(s), effective FPS: {effective_fps:.2f}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = int(duration_s * fps)

    # Background subtractor tuned for CCTV-ish footage
    # history: Number of frames used for background learning. Must be large enough to handle
    # long-duration motion events without absorbing moving objects into the background.
    # With frame_skip=2 and fps=30, history=500 means only 33 seconds of coverage.
    # Use much larger history to prevent objects from becoming "background" during long events.
    bg_history = 5000  # ~5-6 minutes of learning window with default frame_skip
    # varThreshold: lower = more sensitive to motion, higher = only obvious motion
    bg_var_threshold = 16
    
    if gpu_available:
        try:
            # Try CUDA version first
            fgbg = cv2.cuda.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=bg_var_threshold, detectShadows=True)
            log(f"  [GPU] Using CUDA BackgroundSubtractorMOG2 (history={bg_history}, varThreshold={bg_var_threshold})")
        except Exception:
            gpu_available = False
            fgbg = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=bg_var_threshold, detectShadows=True)
            log(f"  [MOG2] Using CPU BackgroundSubtractorMOG2 (history={bg_history}, varThreshold={bg_var_threshold})")
    else:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=bg_var_threshold, detectShadows=True)
        log(f"  [MOG2] Using CPU BackgroundSubtractorMOG2 (history={bg_history}, varThreshold={bg_var_threshold})")

    warmup_frames = max(0, int(warmup_seconds * effective_fps))

    in_event = False
    event_start = 0.0
    peak_ratio = 0.0
    potential_event_start = 0.0  # Track when motion run actually began

    motion_run = 0     # consecutive "motion" frames
    still_run = 0      # consecutive "still" frames

    segments: List[Tuple[float, float, float]] = []

    last_progress_t = time.time()
    start_wall = time.time()

    # ROI mask (optional): roi = (x, y, w, h) as 0..1 fractions
    roi_rect = None

    # Read first frame to compute scaling + ROI mask shape and initialize GPU resources
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    h0, w0 = frame.shape[:2]
    scale = 1.0
    if downscale_width and w0 > downscale_width:
        scale = downscale_width / float(w0)

    # Pre-create filters and kernels outside the loop for better performance
    cpu_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # GPU resources will be initialized after first frame determines final image size
    gpu_gaussian_filter = None
    gpu_morph_open_filter = None
    gpu_morph_dilate_filter = None
    gpu_filters_initialized = False

    def prep(frame_in):
        nonlocal roi_rect, gpu_gaussian_filter, gpu_morph_open_filter, gpu_morph_dilate_filter, gpu_filters_initialized
        fr = frame_in
        
        # GPU-accelerated resize if available
        if scale != 1.0:
            if gpu_available:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(fr)
                    gpu_frame = cv2.cuda.resize(gpu_frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
                    fr = gpu_frame.download()
                except Exception:
                    fr = cv2.resize(fr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
            else:
                fr = cv2.resize(fr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

        if roi and roi_rect is None:
            x, y, w, h = roi
            W = fr.shape[1]
            H = fr.shape[0]
            rx = max(0, min(W - 1, int(x * W)))
            ry = max(0, min(H - 1, int(y * H)))
            rw = max(1, min(W - rx, int(w * W)))
            rh = max(1, min(H - ry, int(h * H)))
            roi_rect = (rx, ry, rw, rh)

        if roi_rect:
            rx, ry, rw, rh = roi_rect
            fr = fr[ry:ry+rh, rx:rx+rw]

        # GPU-accelerated color conversion and blur if available
        if gpu_available:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(fr)
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                
                # Create Gaussian filter once on first call (after ROI is determined)
                if gpu_gaussian_filter is None:
                    gpu_gaussian_filter = cv2.cuda.createGaussianFilter(gpu_gray.type(), gpu_gray.type(), (5, 5), 0)
                
                # Apply pre-created filter
                gpu_gray = gpu_gaussian_filter.apply(gpu_gray)
                
                gray = gpu_gray.download()
            except Exception:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
        else:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray

    # Process the first frame as part of warmup stream
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    processed_frame_idx = 0  # Index of frames actually processed (after skipping)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # Skip frames if frame_skip > 1
        if frame_skip > 1 and (frame_idx - 1) % frame_skip != 0:
            continue
        
        processed_frame_idx += 1
        # t_s is the actual timestamp in the video for this frame
        # This is based on frame_idx (not processed_frame_idx) to give accurate video timestamps
        t_s = frame_idx / fps

        gray = prep(frame)

        # Apply background subtraction (GPU or CPU)
        # Use a very slow learning rate to prevent moving objects from being absorbed into background
        # Default (-1) uses automatic learning rate which adapts too quickly for long events
        # Use 0.0001 to make background model very stable (only adapts to genuine background changes)
        learning_rate = 0.0001
        
        if gpu_available:
            try:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                gpu_fgmask = fgbg.apply(gpu_gray, learning_rate)
                fgmask = gpu_fgmask.download()
            except Exception:
                fgmask = fgbg.apply(gray, learning_rate)
        else:
            fgmask = fgbg.apply(gray, learning_rate)

        # Clean up mask:
        # - remove shadows (MOG2 shadows are 127)
        # - threshold to binary
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Morphology to suppress speckle noise (GPU-accelerated if available)
        if gpu_available:
            try:
                gpu_fgmask = cv2.cuda_GpuMat()
                gpu_fgmask.upload(fgmask)
                
                # Create morphology filters once after first frame (when size is known)
                if not gpu_filters_initialized:
                    gpu_morph_open_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, gpu_fgmask.type(), cpu_kernel)
                    gpu_morph_dilate_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, gpu_fgmask.type(), cpu_kernel)
                    gpu_filters_initialized = True
                
                # Apply pre-created filters
                gpu_fgmask = gpu_morph_open_filter.apply(gpu_fgmask)
                gpu_fgmask = gpu_morph_dilate_filter.apply(gpu_fgmask)
                
                fgmask = gpu_fgmask.download()
            except Exception:
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cpu_kernel, iterations=1)
                fgmask = cv2.dilate(fgmask, cpu_kernel, iterations=1)
        else:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cpu_kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, cpu_kernel, iterations=1)

        # Ignore motion until warmup completes (let background model stabilize)
        if processed_frame_idx <= warmup_frames:
            continue

        # Measure motion: sum area of significant contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = 0
        for c in contours:
            a = cv2.contourArea(c)
            if a >= min_contour_area:
                motion_area += int(a)

        total_area = fgmask.shape[0] * fgmask.shape[1]
        ratio = (motion_area / float(total_area)) if total_area > 0 else 0.0

        # Determine motion frame
        is_motion = ratio >= motion_ratio_threshold

        if is_motion:
            if motion_run == 0:
                # Motion just started - record when this motion run began
                potential_event_start = t_s
            motion_run += 1
            still_run = 0
            if in_event:
                peak_ratio = max(peak_ratio, ratio)
        else:
            still_run += 1
            motion_run = 0

        # Start event only if motion persists
        if (not in_event) and is_motion and motion_run >= min_motion_frames:
            in_event = True
            # Use the actual start of the motion run, not a backdated guess
            event_start = potential_event_start
            peak_ratio = ratio

        # End event only if stillness persists
        if in_event and (still_run >= min_still_frames):
            event_end = t_s
            segments.append((event_start, event_end, peak_ratio))
            in_event = False
            peak_ratio = 0.0

        # Periodic progress
        now = time.time()
        if now - last_progress_t >= progress_every_s:
            if total_frames > 0:
                pct = (frame_idx / total_frames) * 100.0
                eta = (now - start_wall) * (100.0 / max(pct, 0.1) - 1.0)
                log(f"  [detect] {frame_idx}/{total_frames} ({pct:5.1f}%) t={t_s:7.1f}s  ratio={ratio:0.4f}  ETA~{eta:0.0f}s")
            else:
                log(f"  [detect] frame={frame_idx} t={t_s:7.1f}s ratio={ratio:0.4f}")
            last_progress_t = now

    cap.release()

    # Close any open event at EOF
    if in_event:
        segments.append((event_start, duration_s, peak_ratio))

    # Merge nearby segments + add padding + enforce min duration
    merged: List[Tuple[float, float, float]] = []
    for s, e, p in segments:
        s = max(0.0, s - pad_s)
        e = min(duration_s, e + pad_s)
        if not merged:
            merged.append((s, e, p))
        else:
            ps, pe, pp = merged[-1]
            if s - pe <= merge_gap_s:
                merged[-1] = (ps, max(pe, e), max(pp, p))
            else:
                merged.append((s, e, p))

    # Min duration filter
    final: List[Tuple[float, float, float]] = []
    for s, e, p in merged:
        if (e - s) >= min_duration_s:
            final.append((s, e, p))

    return final

# ---------------------------
# Cutting (FFmpeg) with CCTV-safe defaults
# ---------------------------

def make_clip(
    input_path: Path,
    out_path: Path,
    start_s: float,
    end_s: float,
    *,
    keep_audio: bool,
    reencode_video: bool,
    crf: int,
    preset: str,
    ffmpeg_log_path: Path,
    gpu_encoder: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Returns (ok, error_message). Never raises.
    
    If gpu_encoder is provided (e.g., 'h264_nvenc'), uses GPU-accelerated encoding.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Default: copy video; audio often uses odd codecs in CCTV. Safest is to drop audio.
    if reencode_video:
        # frame-accurate, slower
        if gpu_encoder:
            # GPU-accelerated encoding with appropriate hardware acceleration
            hwaccel_method = "auto"
            if "vaapi" in gpu_encoder:
                hwaccel_method = "vaapi"
            elif "qsv" in gpu_encoder:
                hwaccel_method = "qsv"
            elif "nvenc" in gpu_encoder:
                hwaccel_method = "cuda"
            
            cmd = [
                FFMPEG_EXE, "-hide_banner", "-y",
                "-hwaccel", hwaccel_method,  # Hardware-specific acceleration
                "-ss", fmt_time(start_s), "-to", fmt_time(end_s),
                "-i", str(input_path),
                "-c:v", gpu_encoder,
            ]
            # Add encoder-specific options with appropriate quality mapping
            if "nvenc" in gpu_encoder:
                # NVENC: CRF range 0-51, use preset p1-p7
                cmd.extend(["-preset", "p4", "-cq", str(min(51, max(0, crf)))])
            elif "qsv" in gpu_encoder:
                # QSV: Use global_quality 1-51 (lower is better)
                cmd.extend(["-preset", preset, "-global_quality", str(min(51, max(1, crf)))])
            elif "vaapi" in gpu_encoder:
                # VA-API: qp parameter, reasonable range 0-51
                cmd.extend(["-qp", str(min(51, max(0, crf)))])
            elif "videotoolbox" in gpu_encoder:
                # VideoToolbox: q:v parameter, 0-100 (lower is better)
                # Map libx264 CRF (0-51) to VideoToolbox quality (0-100)
                VIDEOTOOLBOX_CRF_MULTIPLIER = 1.96  # 100/51 ≈ 1.96
                quality = min(100, max(0, int(crf * VIDEOTOOLBOX_CRF_MULTIPLIER)))
                cmd.extend(["-q:v", str(quality)])
            else:
                # Fallback for unknown encoder
                cmd.extend(["-crf", str(crf), "-preset", preset])
            
            cmd.extend([
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(out_path)
            ])
        else:
            # CPU encoding
            cmd = [
                FFMPEG_EXE, "-hide_banner", "-y",
                "-ss", fmt_time(start_s), "-to", fmt_time(end_s),
                "-i", str(input_path),
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(out_path)
            ]
    else:
        if keep_audio:
            # copy video, re-encode audio to AAC (fixes pcm_mulaw-in-mp4 etc.)
            cmd = [
                FFMPEG_EXE, "-hide_banner", "-y",
                "-ss", fmt_time(start_s), "-to", fmt_time(end_s),
                "-i", str(input_path),
                "-map", "0:v:0", "-map", "0:a:0?",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                str(out_path)
            ]
        else:
            # fastest, safest: video only
            cmd = [
                FFMPEG_EXE, "-hide_banner", "-y",
                "-ss", fmt_time(start_s), "-to", fmt_time(end_s),
                "-i", str(input_path),
                "-map", "0:v:0",
                "-c:v", "copy",
                "-an",
                "-avoid_negative_ts", "make_zero",
                str(out_path)
            ]

    p = run_capture(cmd)
    # Write ffmpeg stderr for debugging
    try:
        with open(ffmpeg_log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(p.stderr or "")
    except Exception:
        pass

    if p.returncode != 0:
        return False, f"ffmpeg rc={p.returncode} (see {ffmpeg_log_path.name})"
    return True, ""

# ---------------------------
# Main
# ---------------------------

def iter_mp4s(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in folder.rglob("*.mp4") if p.is_file()])
    return sorted([p for p in folder.glob("*.mp4") if p.is_file()])

def safe_relpath(p: Path, base: Path) -> Path:
    try:
        return p.relative_to(base)
    except Exception:
        return Path(p.name)

def parse_roi(s: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """
    ROI string format: "x,y,w,h" where values are 0..1 fractions.
    Example: "0.0,0.2,1.0,0.8" to ignore top 20% (sky/trees).
    """
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x,y,w,h' with 4 floats (0..1).")
    x, y, w, h = parts
    return (x, y, w, h)

def main():
    ap = argparse.ArgumentParser(description="Motion-only clip extraction optimized for security camera footage.")
    ap.add_argument("input_folder", type=str)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--recursive", action="store_true")

    # GPU acceleration
    ap.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration (use CPU only).")
    ap.add_argument("--hwaccel-decode", action="store_true", help="Enable hardware-accelerated video decoding (if supported).")

    # Detection tuning
    ap.add_argument("--downscale-width", type=int, default=640, help="Downscale width for detection (speeds up, reduces noise).")
    ap.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame (e.g., 2 = every 2nd frame, 3 = every 3rd frame). Default: 2 (process every other frame).")
    ap.add_argument("--warmup-seconds", type=float, default=2.0, help="Seconds to let background model stabilize.")
    ap.add_argument("--motion-ratio", type=float, default=0.003, help="Motion ratio threshold (fraction of ROI area). Start 0.002–0.01.")
    ap.add_argument("--min-motion-frames", type=int, default=8, help="Require motion persists this many frames to start an event.")
    ap.add_argument("--min-still-frames", type=int, default=15, help="Require stillness persists this many frames to end an event.")
    ap.add_argument("--min-contour-area", type=int, default=200, help="Ignore small moving blobs (noise).")
    ap.add_argument("--roi", type=str, default=None, help="ROI as 'x,y,w,h' fractions (0..1). Example: 0,0.2,1,0.8")

    # Segmentation
    ap.add_argument("--merge-gap", type=float, default=2.5, help="Merge events separated by <= this many seconds.")
    ap.add_argument("--min-duration", type=float, default=3.0, help="Drop segments shorter than this many seconds.")
    ap.add_argument("--pad", type=float, default=1.0, help="Padding added before/after each segment.")

    # Cutting
    ap.add_argument("--keep-audio", action="store_true", help="Keep audio (re-encode to AAC if needed). Default drops audio for CCTV compatibility.")
    ap.add_argument("--reencode-video", action="store_true", help="Re-encode video (frame-accurate, slower). Default stream-copies video.")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", type=str, default="veryfast")

    # Tools
    ap.add_argument("--ffmpeg", type=str, default=None)
    ap.add_argument("--ffprobe", type=str, default=None)

    ap.add_argument("--csv-name", type=str, default="segments.csv")
    args = ap.parse_args()
    
    # Validate frame_skip
    if args.frame_skip < 1:
        log("ERROR: --frame-skip must be at least 1")
        sys.exit(2)

    in_dir = Path(args.input_folder).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        log(f"ERROR: input folder not found: {in_dir}")
        sys.exit(2)

    out_dir = Path(args.out).expanduser().resolve() if args.out else (in_dir / "motion_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg discovery
    try:
        ffmpeg_exe, ffprobe_exe = require_ffmpeg(args.ffmpeg, args.ffprobe)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(2)

    global FFMPEG_EXE, FFPROBE_EXE
    FFMPEG_EXE, FFPROBE_EXE = ffmpeg_exe, ffprobe_exe

    # Check GPU availability
    use_gpu = not args.no_gpu
    cuda_available = use_gpu and check_cuda_available()
    gpu_encoder = None
    
    if use_gpu:
        if cuda_available:
            log("[GPU] CUDA is available for OpenCV operations")
        else:
            reason = get_cuda_unavailable_reason()
            log(f"[GPU] CUDA not available: {reason}")
            log("[GPU] OpenCV will use CPU for motion detection")
        
        # Detect GPU encoder for FFmpeg
        gpu_encoder = get_gpu_encoder_for_ffmpeg()
        if gpu_encoder:
            log(f"[GPU] FFmpeg GPU encoder detected: {gpu_encoder}")
        else:
            log("[GPU] No GPU encoder detected for FFmpeg, will use CPU encoding")
    else:
        log("[GPU] GPU acceleration disabled by --no-gpu flag")

    roi = parse_roi(args.roi)

    log(f"[io] input : {in_dir}")
    log(f"[io] output: {out_dir}")
    log("[cfg] DETECT:"
        f" downscale_width={args.downscale_width}"
        f" frame_skip={args.frame_skip}"
        f" warmup={args.warmup_seconds}s"
        f" motion_ratio={args.motion_ratio}"
        f" min_motion_frames={args.min_motion_frames}"
        f" min_still_frames={args.min_still_frames}"
        f" min_contour_area={args.min_contour_area}"
        f" roi={roi}"
        f" hwaccel_decode={args.hwaccel_decode}")
    log("[cfg] SEGMENT:"
        f" merge_gap={args.merge_gap}s"
        f" min_duration={args.min_duration}s"
        f" pad={args.pad}s")
    log("[cfg] CUT:"
        f" keep_audio={args.keep_audio}"
        f" reencode_video={args.reencode_video}"
        f" crf={args.crf}"
        f" preset={args.preset}")

    mp4s = iter_mp4s(in_dir, args.recursive)
    log(f"[scan] found {len(mp4s)} mp4 file(s).")
    if not mp4s:
        return

    csv_path = out_dir / args.csv_name
    logs_dir = out_dir / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    total_ok = 0
    total_failed = 0

    start_all = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source_file","clip_index","start_seconds","end_seconds","duration_seconds",
            "peak_motion_ratio","clip_path","status","error"
        ])

        for vi, vid in enumerate(mp4s, start=1):
            rel = safe_relpath(vid, in_dir)
            log("\n" + "=" * 80)
            log(f"[{vi}/{len(mp4s)}] {rel}")
            log("=" * 80)

            try:
                dur, fps = get_duration_fps(vid)
                log(f"[probe] duration={dur:.2f}s fps~{fps:.3f}")

                log("[detect] OpenCV background subtraction…")
                segs = detect_motion_segments_opencv(
                    vid,
                    duration_s=dur,
                    fps=fps,
                    downscale_width=args.downscale_width,
                    warmup_seconds=args.warmup_seconds,
                    motion_ratio_threshold=args.motion_ratio,
                    min_motion_frames=args.min_motion_frames,
                    min_still_frames=args.min_still_frames,
                    min_contour_area=args.min_contour_area,
                    merge_gap_s=args.merge_gap,
                    min_duration_s=args.min_duration,
                    pad_s=args.pad,
                    roi=roi,
                    use_gpu=use_gpu,
                    frame_skip=args.frame_skip,
                    hwaccel_decode=args.hwaccel_decode,
                )

                log(f"[detect] segments={len(segs)}")
                if not segs:
                    continue

                total_segments += len(segs)

                clip_dir = out_dir / rel.parent / vid.stem
                clip_dir.mkdir(parents=True, exist_ok=True)

                for i, (s, e, peak) in enumerate(segs, start=1):
                    clip_name = f"{vid.stem}_motion_{i:03d}_{fmt_time(s)}-{fmt_time(e)}.mp4"
                    clip_path = clip_dir / clip_name
                    clip_log = logs_dir / rel.parent / f"{vid.stem}__clip_{i:03d}.log"

                    log(f"[clip {i:03d}/{len(segs)}] {fmt_time(s)} -> {fmt_time(e)} ({(e-s):.2f}s) peak_ratio={peak:.4f}")

                    ok, err = make_clip(
                        vid, clip_path, s, e,
                        keep_audio=args.keep_audio,
                        reencode_video=args.reencode_video,
                        crf=args.crf,
                        preset=args.preset,
                        ffmpeg_log_path=clip_log,
                        gpu_encoder=gpu_encoder if args.reencode_video else None,
                    )

                    status = "ok" if ok else "failed"
                    if ok:
                        total_ok += 1
                    else:
                        total_failed += 1
                        log(f"  !! clip failed: {err}")

                    w.writerow([
                        str(rel).replace("\\", "/"),
                        i,
                        f"{s:.3f}",
                        f"{e:.3f}",
                        f"{(e-s):.3f}",
                        f"{peak:.6f}",
                        str(clip_path.relative_to(out_dir)).replace("\\", "/"),
                        status,
                        err
                    ])

            except Exception as ex:
                log(f"[FAIL] {rel}: {ex}")

    elapsed = time.time() - start_all
    log("\n" + "#" * 80)
    log("SUMMARY")
    log("#" * 80)
    log(f"Segments found : {total_segments}")
    log(f"Clips OK       : {total_ok}")
    log(f"Clips failed   : {total_failed}")
    log(f"Output folder  : {out_dir}")
    log(f"CSV            : {csv_path}")
    log(f"Logs           : {logs_dir}")
    log(f"Elapsed        : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
