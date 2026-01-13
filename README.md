# Motion-Only Clip Extraction for Security Footage

**Motion CCTV** is a free, open-source command-line tool for extracting **motion-only clips from security camera footage** on Windows (and other platforms).

It is specifically designed to solve a problem that most video editors and scene-detection tools handle poorly:

> **Security camera footage contains long static periods, noisy lighting changes, and compressed video — traditional scene detection produces thousands of useless micro-clips.**

This tool instead uses **background subtraction + sustained motion detection**, similar in spirit to professional NVR systems, and produces **human-scale motion events** you can actually review.

---

## Features

- ✅ **Optimized for security camera footage**
  - Background subtraction (OpenCV MOG2)
  - Noise suppression (blur + morphology)
  - Requires **sustained motion**, not single-frame spikes
- ✅ **Batch processing**
  - Process folders of MP4 files
  - Optional recursive scanning
- ✅ **Motion-only clip extraction**
  - Fast FFmpeg cutting (stream copy by default)
  - Handles odd CCTV audio codecs safely
- ✅ **Always produces a CSV**
  - Every detected motion segment is recorded
  - Clip success/failure tracked per segment
- ✅ **Verbose CLI progress**
  - Continuous detection progress
  - No “silent hangs”
- ✅ **Fully open-source**
  - Python + OpenCV + FFmpeg
  - No proprietary dependencies

---

## Why This Exists

Most tools use **frame-difference or scene-cut detection**, which fails on CCTV footage because:

- Lighting/exposure flickers trigger false positives
- Motion often causes **brief spikes**, not sustained frame differences
- Compression artifacts look like motion
- Result: hundreds of useless 1–2 second clips

This tool instead answers the question:

> **“Has something actually been moving in the scene for long enough to matter?”**

---

## Requirements

### Software
- **Python 3.8+**
- **FFmpeg + FFprobe**
- **OpenCV for Python**

### Install dependencies

```bash
pip install opencv-python
```

**Note:** The standard `opencv-python` package from PyPI does **not** include CUDA support.

### Enabling GPU Acceleration (OpenCV CUDA)

To enable CUDA acceleration in OpenCV, you need a CUDA-enabled build. Options:

**Option 1: Build OpenCV from source with CUDA** (recommended for best performance)
- Follow the official OpenCV build instructions with `-DWITH_CUDA=ON`
- See: https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html

**Option 2: Use pre-built CUDA packages** (if available for your platform)
- Windows: Check https://github.com/opencv/opencv/releases or community builds
- Linux: Some distributions provide `python3-opencv-cuda` packages

**Verify CUDA support:**
```python
python -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

If CUDA is properly configured, this will show the number of available CUDA devices (e.g., "CUDA devices: 1").

**Note:** Even without CUDA, the script will still benefit from FFmpeg hardware encoding (NVENC on NVIDIA GPUs) if you have appropriate drivers installed.

FFmpeg (Windows recommended via Chocolatey):

```powershell
choco install ffmpeg
```

Or ensure `ffmpeg.exe` and `ffprobe.exe` are on your `PATH`.

---

## GPU Acceleration

Motion CCTV automatically detects and uses GPU acceleration when available:

### OpenCV CUDA Acceleration
- **Background subtraction (MOG2)** - runs on GPU
- **Image resizing** - GPU-accelerated
- **Color conversion and blur** - GPU-accelerated  
- **Morphological operations** - GPU-accelerated

### FFmpeg Hardware Decoding
Hardware-accelerated video decoding can be enabled with `--hwaccel-decode`:
- **NVIDIA NVDEC** - NVIDIA GPUs (via CUDA)
- **Intel Quick Sync** - Intel CPUs with iGPU
- **VA-API** - Linux with compatible hardware
- **VideoToolbox** - macOS

```bash
python motion_cctv.py /path/to/videos --hwaccel-decode
```

Note: Requires OpenCV built with hardware acceleration support.

### FFmpeg Hardware Encoding
Automatically detects and uses available GPU encoders:
- **NVIDIA NVENC** (h264_nvenc) - NVIDIA GPUs
- **Intel Quick Sync** (h264_qsv) - Intel CPUs with iGPU
- **VA-API** (h264_vaapi) - Linux with compatible hardware
- **VideoToolbox** (h264_videotoolbox) - macOS

### Disabling GPU
To force CPU-only processing:
```bash
python motion_cctv.py /path/to/videos --no-gpu
```

GPU acceleration provides significant performance improvements (typically 2-5x faster) when processing large video files.

---

## Performance Optimizations

Motion CCTV includes several optimizations to speed up processing:

### Frame Downscaling (Default: ON)
Frames are downscaled to 640px width for detection (configurable with `--downscale-width`). This:
- Reduces processing time by ~4x
- Reduces noise from compression artifacts
- Maintains detection accuracy for most CCTV footage

```bash
python motion_cctv.py /path/to/videos --downscale-width 480  # More aggressive
```

### Frame Skipping (Default: OFF)
Process every Nth frame instead of all frames:

```bash
python motion_cctv.py /path/to/videos --frame-skip 2  # Every 2nd frame (2x faster)
python motion_cctv.py /path/to/videos --frame-skip 3  # Every 3rd frame (3x faster)
```

**Trade-offs:**
- ✅ Dramatically faster processing (2-3x speedup)
- ⚠️ May miss very brief motion events
- ⚠️ Best for high FPS footage (30+ fps)

**Recommended settings:**
- 30 fps footage: `--frame-skip 2` (processes 15 fps)
- 60 fps footage: `--frame-skip 3` (processes 20 fps)

### Combined Optimizations
For maximum performance on high-quality CCTV footage:

```bash
python motion_cctv.py /path/to/videos \
  --downscale-width 640 \
  --frame-skip 2 \
  --hwaccel-decode \
  --reencode-video  # Only if re-encoding, enables GPU encoding
```

This can provide 5-10x performance improvement on systems with GPU acceleration.

---

## Usage

### Basic run (recommended defaults)

```bash
python motion_cctv.py /path/to/video_folder
```

Output will be written to:

```
<video_folder>/motion_output/
```

---

## Output Structure

```
motion_output/
├── segments.csv
├── _logs/
│   ├── video__clip_001.log
│   └── ...
└── VideoName/
    ├── VideoName_motion_001_12.345-25.678.mp4
    └── ...
```

### `segments.csv` columns

| Column            | Description                       |
|-------------------|-----------------------------------|
| source_file       | Original video filename           |
| clip_index        | Index within that video           |
| start_seconds     | Segment start time                |
| end_seconds       | Segment end time                  |
| duration_seconds  | Segment length                    |
| peak_motion_ratio | Max motion intensity during event |
| clip_path         | Relative path to output clip      |
| status            | `ok` or `failed`                  |
| error             | Error message if clip failed      |

---

## Recommended Settings (Outdoor CCTV)

```bash
python motion_cctv.py /path/to/videos   --motion-ratio 0.003   --min-motion-frames 8   --min-still-frames 18   --min-contour-area 250   --merge-gap 3.0   --min-duration 3.0   --pad 1.0
```

---

## Ignoring Noisy Regions (ROI)

If part of the frame (e.g. sky, trees) causes false motion:

```bash
--roi 0,0.2,1,0.8
```

ROI format:

```
x,y,width,height   (fractions from 0.0 to 1.0)
```

---

## Audio Handling

Many security cameras store audio as **`pcm_mulaw` inside MP4**, which breaks stream-copy.

### Default behavior
- Video only
- Audio dropped

### Keep audio (re-encoded to AAC)

```bash
--keep-audio
```

### Force full re-encode

```bash
--reencode-video
```

---

## Tuning Guide

### Too many false positives?
- Increase `--min-contour-area`
- Increase `--motion-ratio`
- Increase `--min-motion-frames`

### Missing real motion?
- Decrease `--motion-ratio`
- Decrease `--min-contour-area`

### Events split into multiple clips?
- Increase `--merge-gap`
- Increase `--min-still-frames`

---

## Concatenating Clips

The repository includes a utility script `concat_clips.py` for combining multiple video files into a single output file.

### Usage

Recursively concatenate all video files in a directory:

```bash
python concat_clips.py /path/to/videos output.mp4
```

Non-recursive (only process files in the specified directory):

```bash
python concat_clips.py /path/to/videos output.mp4 --no-recursive
```

Custom ffmpeg/ffprobe paths:

```bash
python concat_clips.py /path/to/videos output.mp4 --ffmpeg /custom/path/ffmpeg --ffprobe /custom/path/ffprobe
```

### Features

- **Recursive scanning**: Finds all video files in subdirectories by default
- **Spec preservation**: Automatically detects video specs (codec, resolution, framerate) from the first clip
- **Smart re-encoding**: Re-encodes clips that don't match the target specs to ensure smooth concatenation
- **Multiple formats**: Supports mp4, avi, mkv, mov, flv, wmv, webm, m4v, mpg, mpeg
- **Safe concatenation**: Uses FFmpeg concat demuxer with temporary files for intermediate processing

### How It Works

1. Scans the input directory for video files (recursively by default)
2. Extracts video specs (resolution, framerate, codec) from the first video file
3. For each subsequent video:
   - If specs match, uses the original file (fast, no re-encoding)
   - If specs differ, re-encodes to match the first video's specs
4. Concatenates all processed videos using FFmpeg's concat demuxer
5. Outputs a single video file with consistent encoding throughout

### Requirements

- FFmpeg and FFprobe must be installed and available on PATH
- Same requirements as `motion_cctv.py` (see [Requirements](#requirements) section above)

### Example Use Cases

**Combine motion clips from multiple cameras:**

```bash
python concat_clips.py /path/to/motion_output/Camera1/ combined_camera1.mp4
```

**Merge all clips in a folder with different resolutions:**

```bash
# The script automatically re-encodes to match the first clip
python concat_clips.py /path/to/mixed_videos/ unified_output.mp4
```

**Process only specific directory without subdirectories:**

```bash
python concat_clips.py /path/to/specific_folder/ output.mp4 --no-recursive
```

---

## Design Philosophy

- Security footage is not cinematic video
- Motion must be **persistent**, not momentary
- CSV output is **authoritative**, clips are secondary
- Failures should be logged, not fatal
- CLI tools must never appear “stuck”

---

## License

MIT License (or replace with your preferred license).

---

## Contributing

Pull requests welcome for:
- Additional detectors (optical flow, object tracking)
- Performance improvements
- Better defaults for specific camera types
- Visualization/debug modes
