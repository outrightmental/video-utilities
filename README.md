# Video Utilities

[![CI](https://github.com/outrightmental/video-utilities/actions/workflows/ci.yml/badge.svg)](https://github.com/outrightmental/video-utilities/actions/workflows/ci.yml)

A collection of free, open-source command-line video utilities for processing, concatenating, and analysing video files.

---

## Table of Contents

- [Requirements](#requirements)
- [motion\_cctv — Motion-Only Clip Extraction](#motion_cctv--motion-only-clip-extraction)
- [concat\_clips — Concatenate Video Clips](#concat_clips--concatenate-video-clips)
- [shuffle\_concat\_seam — Shuffle & Concatenate with Seam Matching](#shuffle_concat_seam--shuffle--concatenate-with-seam-matching)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

### Software
- **Python 3.8+**
- **FFmpeg + FFprobe**
- **OpenCV for Python**

### Install dependencies

```bash
pip install opencv-python numpy
```

**Note:** The standard `opencv-python` package from PyPI does **not** include CUDA support.

FFmpeg (Windows recommended via Chocolatey):

```powershell
choco install ffmpeg
```

Or ensure `ffmpeg` and `ffprobe` are on your `PATH`.

---

## motion\_cctv — Motion-Only Clip Extraction

**Location:** [`motion_cctv/`](motion_cctv/)

Extract motion-only clips from security camera (or any other) footage using background subtraction and sustained-motion detection.

### Why?

Most tools use frame-difference or scene-cut detection, which fails on security/CCTV footage because lighting flickers, compression artefacts, and brief spikes produce hundreds of useless micro-clips. `motion_cctv` instead answers:

> **"Has something actually been moving in the scene for long enough to matter?"**

### Features

- Background subtraction (OpenCV MOG2) with noise suppression
- Requires **sustained motion**, not single-frame spikes
- Batch processing with optional recursive folder scanning
- Fast FFmpeg cutting (stream copy by default); handles odd audio codecs
- CSV of every detected segment with clip status/errors
- Verbose CLI progress — never appears "stuck"

### Basic Usage

```bash
python motion_cctv/motion_cctv.py /path/to/video_folder
```

Output:

```
<video_folder>/motion_output/
├── segments.csv
├── _logs/
│   └── ...
└── VideoName/
    └── VideoName_motion_001_12.345-25.678.mp4
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

### GPU Acceleration

#### OpenCV CUDA

Background subtraction, resizing, colour conversion, blur, and morphology can all run on a CUDA-enabled GPU.

To enable CUDA, build OpenCV from source with `-DWITH_CUDA=ON` or use a pre-built CUDA package. Verify with:

```bash
python -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

#### FFmpeg Hardware Decoding

```bash
python motion_cctv/motion_cctv.py /path/to/videos --hwaccel-decode
```

Supports NVIDIA NVDEC, Intel Quick Sync, VA-API (Linux), and VideoToolbox (macOS).

#### FFmpeg Hardware Encoding

Automatically detects GPU encoders: NVENC, Quick Sync, VA-API, VideoToolbox.

#### Disable GPU

```bash
python motion_cctv/motion_cctv.py /path/to/videos --no-gpu
```

### Performance Optimizations

| Optimisation | Flag | Default | Notes |
|---|---|---|---|
| Frame downscaling | `--downscale-width 640` | ON (640 px) | ~4× speedup, reduces noise |
| Frame skipping | `--frame-skip 2` | OFF | ~2-3× speedup; best for 30+ fps footage |
| HW decode | `--hwaccel-decode` | OFF | Requires driver support |

Combined example for maximum throughput:

```bash
python motion_cctv/motion_cctv.py /path/to/videos \
  --downscale-width 640 \
  --frame-skip 2 \
  --hwaccel-decode \
  --reencode-video
```

### Tuning Guide

| Problem | Adjust |
|---|---|
| Too many false positives | ↑ `--min-contour-area`, `--motion-ratio`, `--min-motion-frames` |
| Missing real motion | ↓ `--motion-ratio`, `--min-contour-area` |
| Events split into multiple clips | ↑ `--merge-gap`, `--min-still-frames` |

### ROI (Ignoring Noisy Regions)

```bash
python motion_cctv/motion_cctv.py /path/to/videos --roi 0,0.2,1,0.8
```

Format: `x,y,width,height` (fractions 0.0–1.0).

### Audio Handling

| Mode | Flag |
|---|---|
| Video only (default) | *(none)* |
| Keep audio (re-encode to AAC) | `--keep-audio` |
| Full re-encode | `--reencode-video` |

### Tests

```bash
python motion_cctv/test_e2e.py
```

Requires example footage in `example_footage/`. The test is skipped when footage is absent.

---

## shuffle\_concat\_seam — Shuffle & Concatenate with Seam Matching

**Location:** [`shuffle_concat_seam/`](shuffle_concat_seam/)

> **Note:** The functionality of this utility has been merged into [`concat_clips`](#concat_clips--concatenate-video-clips).
> Use `concat_clips --shuffle --match-seams` for the equivalent behaviour.

Shuffle video clips into a random order and concatenate them with **motion-aware seam frame matching** for smoother transitions.

### The Problem

When concatenating clips that are "almost loops", simple joining produces visible jumps at each boundary. Matching a single still frame can also cause sudden-reversal-motion seams in clips with repetitive back-and-forth motion.

### The Solution

1. Extract the last **2 consecutive frames** of the preceding clip ("needle pair").
2. Sample pairs of consecutive frames in the first N seconds of the next clip ("haystack").
3. Pick the pair with the lowest combined MSE — this captures **motion direction** and prevents sudden reversals.

### Basic Usage

```bash
python shuffle_concat_seam/shuffle_concat_seam.py /path/to/videos output.mp4
```

Automatic output naming:

```bash
python shuffle_concat_seam/shuffle_concat_seam.py --folder /path/to/videos
# → /path/to/videos.mp4
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--folder` | Input folder; output saved as `<folder>.mp4` |
| `--haystack-duration` | Seconds to search for best match (default: 1.0) |
| `--seed` | Random seed for reproducible ordering |
| `--recursive` | Search subdirectories |
| `--fps` | Output framerate (H.264 bitstream remux) |
| `--no-trim` | Skip seam matching; use full clips |
| `--ffmpeg` / `--ffprobe` | Custom executable paths |

### How It Works

1. Reads all video files from the input directory.
2. Shuffles into random order.
3. For each successive clip, finds the best-matching start frame pair via combined MSE comparison (grayscale + Gaussian blur).
4. Trims via stream copy (or re-encodes if specs differ).
5. Concatenates into a single output file.

### Tests

```bash
python -m unittest shuffle_concat_seam.test_shuffle_concat_seam -v
```

---

## concat\_clips — Concatenate Video Clips

**Location:** [`concat_clips/`](concat_clips/)

Concatenate all video files in a directory into a single output file. Clips are sorted **alphabetically by filename** by default. Optional `--shuffle` and `--match-seams` flags enable random ordering and smooth motion-aware seam transitions.

### Basic Usage

```bash
# Default: alphabetical order
python concat_clips/concat_clips.py /path/to/videos output.mp4

# Shuffle into a random order
python concat_clips/concat_clips.py /path/to/videos output.mp4 --shuffle

# Match seams between clips for smoother transitions (requires OpenCV)
python concat_clips/concat_clips.py /path/to/videos output.mp4 --match-seams

# Shuffle and match seams together
python concat_clips/concat_clips.py /path/to/videos output.mp4 --shuffle --match-seams --seed 42
```

Automatic output naming:

```bash
python concat_clips/concat_clips.py --folder /path/to/videos
# → /path/to/videos.mp4
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--shuffle` | Shuffle clips into a random order (default: alphabetical) |
| `--seed` | Random seed for reproducible shuffling (used with `--shuffle`) |
| `--match-seams` | Match seams between clips using motion-aware frame comparison (requires OpenCV) |
| `--haystack-duration` | Seconds to search for best match (used with `--match-seams`, default: 1.0) |
| `--haystack-skip` | Seconds to skip at start of each clip before searching (default: 0.0) |
| `--folder` | Input folder; output saved as `<folder>.mp4` |
| `--no-recursive` | Don't search subdirectories (default: recursive) |
| `--fps` | Output framerate (H.264 bitstream remux) |
| `--ffmpeg` / `--ffprobe` | Custom executable paths |

### Seam Matching Algorithm (`--match-seams`)

1. Extract the last **2 consecutive frames** of the preceding clip ("needle pair").
2. Sample pairs of consecutive frames in the first N seconds of the next clip ("haystack").
3. Pick the pair with the lowest combined MSE — this captures **motion direction** and prevents sudden reversals.
4. Trim the next clip to start at that best-matching frame.

### Features

- Alphabetical sort by filename (default)
- Random shuffle with optional reproducible seed (`--shuffle`, `--seed`)
- Motion-aware seam matching for smooth transitions (`--match-seams`)
- Recursive scanning across subdirectories (default)
- Auto-detects codec, resolution, and framerate from the first clip
- Smart re-encoding for clips that don't match
- Supports mp4, avi, mkv, mov, flv, wmv, webm, m4v, mpg, mpeg

### Tests

```bash
python -m unittest concat_clips.test_concat_clips -v
```

---

## Contributing

Pull requests welcome for:

- Additional detectors (optical flow, object tracking)
- Performance improvements
- Better defaults for specific camera types
- New video utilities

---

## License

MIT
