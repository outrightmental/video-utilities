# Video Utilities

[![CI](https://github.com/outrightmental/video-utilities/actions/workflows/ci.yml/badge.svg)](https://github.com/outrightmental/video-utilities/actions/workflows/ci.yml)

A collection of free, open-source command-line video utilities for processing, concatenating, and analysing video files.

---

## Table of Contents

- [Requirements](#requirements)
- [extract\_motion — Motion-Only Clip Extraction](#extract_motion--motion-only-clip-extraction)
- [concat\_clips — Concatenate Video Clips](#concat_clips--concatenate-video-clips)
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

## extract\_motion — Motion-Only Clip Extraction

**Location:** [`extract_motion/`](extract_motion/)

Extract motion-only clips from any footage using background subtraction and sustained-motion detection. Long, mostly-static recordings — security cameras, dashcams, trail cameras, timelapses, locked-off b-roll — benefit most, but nothing about the tool is tied to a particular source.

### Why?

Most tools use frame-difference or scene-cut detection, which falls apart on real-world continuous footage: lighting flickers, compression artefacts, and brief spikes produce hundreds of useless micro-clips. `extract_motion` instead answers:

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
python extract_motion/extract_motion.py /path/to/video_folder
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
python extract_motion/extract_motion.py /path/to/videos --hwaccel-decode
```

Supports NVIDIA NVDEC, Intel Quick Sync, VA-API (Linux), and VideoToolbox (macOS).

#### FFmpeg Hardware Encoding

Automatically detects GPU encoders: NVENC, Quick Sync, VA-API, VideoToolbox.

#### Disable GPU

```bash
python extract_motion/extract_motion.py /path/to/videos --no-gpu
```

### Performance Optimizations

| Optimisation | Flag | Default | Notes |
|---|---|---|---|
| Frame downscaling | `--downscale-width 640` | ON (640 px) | ~4× speedup, reduces noise |
| Frame skipping | `--frame-skip 2` | OFF | ~2-3× speedup; best for 30+ fps footage |
| HW decode | `--hwaccel-decode` | OFF | Requires driver support |

Combined example for maximum throughput:

```bash
python extract_motion/extract_motion.py /path/to/videos \
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
python extract_motion/extract_motion.py /path/to/videos --roi 0,0.2,1,0.8
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
python extract_motion/test_e2e.py
```

Requires example footage in `example_footage/`. The test is skipped when footage is absent.

---

## concat\_clips — Concatenate Video Clips

**Location:** [`concat_clips/`](concat_clips/)

Concatenate all video files in a directory into a single output file. Clips are sorted **alphabetically by filename** by default. Optional flags choose a different ordering (`--shuffle`, `--sort-by-matching-ends`, `--sort-by-intensity`), smooth the transitions between clips (`--match-seams`), and burn each clip's filename into the output for review (`--review`).

### Basic Usage

```bash
# Default: alphabetical order
python concat_clips/concat_clips.py /path/to/videos output.mp4

# Shuffle into a random order
python concat_clips/concat_clips.py /path/to/videos output.mp4 --shuffle

# Order clips so each beginning continues the previous ending (requires OpenCV)
python concat_clips/concat_clips.py /path/to/videos output.mp4 --sort-by-matching-ends

# Start that ordering from a particular clip
python concat_clips/concat_clips.py /path/to/videos output.mp4 --sort-by-matching-ends --first-clip joe123

# Match seams between clips for smoother transitions (requires OpenCV)
python concat_clips/concat_clips.py /path/to/videos output.mp4 --match-seams

# Shuffle and match seams together
python concat_clips/concat_clips.py /path/to/videos output.mp4 --shuffle --match-seams --seed 42

# Order by matching ends, then cut each junction at its best frame
python concat_clips/concat_clips.py /path/to/videos output.mp4 --sort-by-matching-ends --match-seams

# Order by how much motion each clip contains (quietest first)
python concat_clips/concat_clips.py /path/to/videos output.mp4 --sort-by-intensity

# Busiest first instead
python concat_clips/concat_clips.py /path/to/videos output.mp4 --sort-by-intensity desc

# Smooth transitions that also build in energy across the sequence
python concat_clips/concat_clips.py /path/to/videos output.mp4 --sort-by-matching-ends --sort-by-intensity

# Burn each clip's filename into the output, for reviewing which clips to delete
python concat_clips/concat_clips.py /path/to/videos output.mp4 --review
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
| `--sort-by-matching-ends` | Order clips so each beginning continues the previous ending (requires OpenCV; cannot be combined with `--shuffle`) |
| `--first-clip` | Substring selecting which clip opens the sequence (used with `--sort-by-matching-ends`, default: alphabetically first) |
| `--sort-window` | Seconds analysed at each clip boundary (used with `--sort-by-matching-ends`, default: 0.25) |
| `--sort-by-intensity` | Take each clip's overall amount of motion into account, `asc` (default) or `desc` (requires OpenCV; cannot be combined with `--shuffle`) |
| `--match-seams` | Match seams between clips using motion-aware frame comparison (requires OpenCV) |
| `--review` | Burn each clip's filename into its stretch of the output so a reviewer knows which source clips to delete before the next run (requires OpenCV; forces a re-encode of every clip) |
| `--haystack-duration` | Seconds to search for best match (used with `--match-seams`, default: 1.0) |
| `--haystack-skip` | Seconds to skip at start of each clip before searching (default: 0.0) |
| `--folder` | Input folder; output saved as `<folder>.mp4` |
| `--no-recursive` | Don't search subdirectories (default: recursive) |
| `--fps` | Output framerate (H.264 bitstream remux) |
| `--ffmpeg` / `--ffprobe` | Custom executable paths |

### Clip Ordering Algorithm (`--sort-by-matching-ends`)

Instead of trusting filenames, this derives the running order from the footage itself — each clip is followed by whichever remaining clip picks up closest to where it left off.

1. Analyse a short window (`--sort-window`, default 0.25s) at **both ends of every clip**. That is two ffmpeg passes per clip, not one per pair. Frames are downscaled and blurred, then summarized as three things: the boundary frame itself, an aggregate motion vector, and an average speed.
2. Score every possible transition A → B on three criteria:
   - **Appearance** — how alike A's last frame and B's first frame look (MSE).
   - **Direction** — whether both are moving the same way, so a leftward pan is not followed by a rightward one. Direction comes from centroid displacement (centroid of appearing pixels minus centroid of disappearing pixels), which encodes *where things moved* rather than what they look like. It is faded out when either clip's motion is too small to trust, so noise in a static shot cannot fake a direction.
   - **Speed** — whether both are moving at a comparable rate, so the cut does not jump from a fast pan to a near-freeze.
3. Starting from the first clip, greedily append whichever unused clip scores best against the current clip's end. Ties break on filename, so the result is deterministic.

The chosen order and each transition's score are printed, lower being smoother.

**Choosing the first clip.** By default the sequence opens with the alphabetically first clip. `--first-clip` takes a substring searched within each filename — `--first-clip joe123` matches `joe123-final.mp4`. Matching is case-insensitive; if several clips match, the alphabetically first one wins and the rest are listed as a warning; if none match, the run stops rather than silently guessing.

**Combining.** `--sort-by-matching-ends` cannot be used with `--shuffle` — one randomises the order, the other derives it — and attempting both is an error. It does compose with `--match-seams`: ordering decides which clips adjoin, then seam matching decides where each junction is cut.

### Clip Intensity (`--sort-by-intensity [asc|desc]`)

Each clip is sampled end to end at a fixed rate and scored on how much the picture changes between samples — a locked-off shot of an empty room scores near zero, a busy street scores high. The rate is the same for every clip, so a long clip is not made to look busier than it is by being sampled more sparsely. This costs one extra decode pass per clip, cheap next to the full re-encode every clip already gets on the way into the concatenation.

The measured value for each clip is printed alongside the chosen order.

**On its own**, it orders the clips outright: quietest first for `asc` (the default), busiest first for `desc`. Clips whose motion could not be measured go last in either direction.

**Layered onto `--sort-by-matching-ends`**, it becomes one half of the decision instead:

- The sequence opens on the quietest clip for `asc`, or the busiest for `desc`, so the arc has somewhere to go. An ascending arc that opened on the busiest clip could only ever fall. An explicit `--first-clip` still wins over this.
- At each step the remaining clips are ranked twice — once on transition smoothness, once on intensity — and the blended rank picks the winner, with smoothness deciding exact ties.

The two are combined by *rank* rather than by raw value on purpose. A transition score is in squared grey levels and can differ a hundredfold between visually distinct clips, while intensity is a single number per clip; any fixed weighting between them would be arbitrary, and in practice would leave intensity unable to affect anything at all. Ranking puts them on equal footing.

So the three ordering modes give you a choice of what to optimise:

| Flags | Optimises |
|-------|-----------|
| `--sort-by-matching-ends` | Smooth transitions only |
| `--sort-by-intensity` | The energy arc only |
| Both together | A balance of the two |

Like `--sort-by-matching-ends`, it cannot be combined with `--shuffle`, and it composes with `--match-seams`.

### Seam Matching Algorithm (`--match-seams`)

Where clip ordering decides which clips adjoin, seam matching decides where each junction is actually cut — searching **both sides** of the seam at once rather than trimming only the incoming clip.

1. For each adjacent pair of clips (A, B), extract frames from the last `--haystack-duration` seconds of A and the first `--haystack-duration` seconds of B.
2. Score every combination of a consecutive pair from A's tail against a consecutive pair from B's head on three criteria:
   - **Similarity** — the two junction frames look as alike as possible.
   - **Direction** — motion continues the same way across the cut instead of reversing.
   - **Velocity** — the cut lands during fast motion, where artefacts are hardest to notice.
3. The winning combination sets both the trim *end* of A and the trim *start* of B, with B resuming one frame past the junction so the matched frame is not shown twice.
4. Every clip is then re-encoded at its determined trim points and concatenated.

Because the cut must be frame-accurate, trimmed clips are re-encoded rather than stream-copied — stream copy can only cut at keyframes, which would defeat the search.

### Review Mode (`--review`)

Burns each clip's filename into the bottom-left corner of its own stretch of the output — white text on a semi-transparent box — so someone reviewing the footage knows exactly which source clips to delete before the next concat run.

The label is rendered with OpenCV and composited with ffmpeg's core `overlay` filter, so it works on ffmpeg builds that lack the `drawtext` filter (Homebrew's, for one). It composes with every ordering flag and with `--match-seams`. Because the label must be burned into the frames, every clip is re-encoded — including clips that would otherwise be used untouched.

### Features

- Alphabetical sort by filename (default)
- Random shuffle with optional reproducible seed (`--shuffle`, `--seed`)
- Motion-aware clip ordering from the footage itself (`--sort-by-matching-ends`, `--first-clip`)
- Ordering by how much motion each clip contains, alone or blended with the above (`--sort-by-intensity`)
- Motion-aware seam matching for smooth transitions (`--match-seams`)
- Burned-in clip-name labels for reviewing which clips to delete (`--review`)
- Recursive scanning across subdirectories (default)
- Auto-detects codec, resolution, and framerate from the first clip
- Smart re-encoding for clips that don't match
- Supports mp4, avi, mkv, mov, flv, wmv, webm, m4v, mpg, mpeg

### Tests

```bash
python -m unittest concat_clips.test_concat_clips -v
python -m unittest concat_clips.test_concat_seams -v
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
