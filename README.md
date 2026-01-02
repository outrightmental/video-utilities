# Motion CCTV — Motion-Only Clip Extraction for Security Footage

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

| Column | Description |
|------|------------|
| source_file | Original video filename |
| clip_index | Index within that video |
| start_seconds | Segment start time |
| end_seconds | Segment end time |
| duration_seconds | Segment length |
| peak_motion_ratio | Max motion intensity during event |
| clip_path | Relative path to output clip |
| status | `ok` or `failed` |
| error | Error message if clip failed |

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
