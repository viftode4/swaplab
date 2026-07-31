# Face Identity Builder + Swapper Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the best local face identity from capture video (`identity.py`), benchmark every zero-shot swapper against it (`bench.py`), and lock the winning model in as SwapLab's default.

**Architecture:** Both tools run IN-PROCESS with facefusion's internals (detector, landmarks, ArcFace embeddings) via a shared bootstrap module `ff_session.py` — one process for hundreds of frames, no per-frame subprocess. Swap execution in bench.py shells out to `facefusion.py headless-run` exactly like swap.py does. `scan_faces.py` is refactored onto the same bootstrap.

**Tech Stack:** Python 3.12 (repo venv), facefusion internals at pinned commit `3f81a8a`, ffmpeg raw-frame pipes (pattern already in swap.py), cv2 + numpy + Pillow (all already installed as facefusion deps).

## Global Constraints

- Consent only; on-device; `faces/`, `clips/`, `jobs/`, `photos/`, `bench-targets/`, `bench-results/` never get committed (add the last two to .gitignore in Task 3).
- Commit messages: imperative sentence, repo style, NO prefixes, NO Co-Authored-By/AI trailers.
- No new Python dependencies.
- Never modify files under `facefusion/`.
- facefusion internals import pattern: must chdir into the checkout and put it on sys.path (package shadowing — repo root has a sibling dir named `facefusion`); resolve ALL user paths to absolute BEFORE chdir.
- Real swaps/scans take seconds-to-minutes; Bash verification steps use timeout 600000.
- The pinned facefusion's swapper models ALL accept pixel boost `1024x1024` (verified in `processors/modules/face_swapper/choices.py`).
- Face library persons currently: alexia, jeni, teo, vlad. Existing test media: `clips/first-test.mp4` (one person), `clips/screenrecording_*.mp4`. Vlad's capture clips may NOT have arrived yet — every QA step must work with existing media; the vlad run is the final, user-gated step.

---

### Task 1: ff_session.py bootstrap + scan_faces.py refactor

**Files:**
- Create: `ff_session.py` (repo root)
- Modify: `scan_faces.py` (use the bootstrap)

**Interfaces:**
- Produces: `ff_session.boot(config_path: str, extra_args: list[str] | None = None) -> None` — makes `import facefusion.*` work and initializes facefusion state (detector/landmarker settings from the ini, `--face-selector-order left-right`). After calling it, `from facefusion.face_creator import get_many_faces` etc. just work, and the process cwd IS the facefusion checkout. Also `ff_session.FACEFUSION: Path` and `ff_session.ROOT: Path` (absolute, resolved before any chdir). Tasks 2-3 call `boot()` then use facefusion directly.

- [ ] **Step 1: Write ff_session.py**

```python
#!/usr/bin/env python3
"""Bootstrap facefusion's internals for in-process use by repo tools.

The checkout lives at <repo>/facefusion and the real package one level
deeper, so the repo root on sys.path shadows the package. boot() chdirs
into the checkout, fixes sys.path, and feeds facefusion's own arg parser
so state defaults + facefusion-swaplab.ini apply exactly as in a real swap.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACEFUSION = ROOT / 'facefusion'
CONFIG = ROOT / 'facefusion-swaplab.ini'


def boot(config_path: str | None = None, extra_args: list[str] | None = None) -> None:
    os.chdir(FACEFUSION)
    sys.path.insert(0, str(FACEFUSION))
    sys.argv = ['facefusion.py', 'headless-run',
                '--config-path', str(config_path or CONFIG),
                '--target-path', '/dev/null/unused.jpg',
                '--output-path', '/dev/null/unused.jpg',
                '--face-selector-order', 'left-right',
                '--processors', 'face_swapper',
                *(extra_args or [])]
    from facefusion.program import create_program
    from facefusion.args import apply_args
    from facefusion import state_manager
    apply_args(vars(create_program().parse_args()), state_manager.init_item)
```

- [ ] **Step 2: Refactor scan_faces.py onto boot()**

Replace scan_faces.py's own chdir/sys.path/sys.argv/parser block with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_session

def main() -> None:
    target = str(Path(sys.argv[1]).resolve())
    config = str(Path(sys.argv[2]).resolve())
    ff_session.boot(config, ['--target-path', target])
    ...rest unchanged (read_static_image, get_many_faces, sort_and_filter_faces, JSON print)...
```

Keep the output JSON shape byte-identical: `{"width", "height", "faces": [{"index", "x", "y", "w", "h"}]}`. Note scan_faces.py is invoked by swap.py with cwd=FACEFUSION — after this refactor it no longer NEEDS that cwd (boot chdirs itself), but the callers stay unchanged.

- [ ] **Step 3: Verify the refactor end to end**

```bash
.venv/bin/python swap.py --list-faces --video faces/alexia/photo-1.jpeg | tail -1 | .venv/bin/python -m json.tool
cd facefusion && ../.venv/bin/python ../scan_faces.py ../faces/alexia/photo-1.jpeg ../facefusion-swaplab.ini | tail -1; cd ..
# and from the repo root WITHOUT cd (the new capability):
.venv/bin/python scan_faces.py faces/alexia/photo-1.jpeg facefusion-swaplab.ini | tail -1
```

Expected: all three print the same 1-face JSON, exit 0.

- [ ] **Step 4: Commit**

```bash
git add ff_session.py scan_faces.py
git commit -m "Share one facefusion bootstrap across the repo tools"
```

---

### Task 2: identity.py — build the best face directory from video

**Files:**
- Create: `identity.py` (repo root)

**Interfaces:**
- Consumes: `ff_session.boot()`; after boot: `facefusion.vision.read_static_image`, `facefusion.face_creator.get_many_faces`.
- Produces: CLI `identity.py --video V [--video V2 ...] --person NAME [--keep N] [--dry-run]`. Writes `faces/<person>/photo-<n>.jpg` crops; archives any previous photos to `faces/<person>/.old-<YYYYmmdd-HHMMSS>/`. Task 3's bench consumes `faces/<person>/` as a normal face dir (nothing new).

- [ ] **Step 1: Write identity.py**

Key decisions baked in (constants at top, calibrated in Step 3):

```python
#!/usr/bin/env python3
"""Build the best multi-angle face identity from capture video.

Usage:
    .venv/bin/python identity.py --video vlad-angles.mov --video vlad-expressions.mov \
        --person vlad [--keep 18] [--dry-run]

Scores every ~4th-per-second frame for sharpness and pose, buckets by head
angle, and keeps the sharpest frame per bucket so the identity covers the
full pose/expression space instead of 8 arbitrary frames.
Consented faces only.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_session

SAMPLE_FPS = 4
MIN_FACE_HEIGHT = 256        # px in the source frame; smaller crops upscale badly
MIN_DETECTOR_SCORE = 0.7
SHARPNESS_PERCENTILE = 40    # drop the blurriest 40% of each video
CROP_PAD = 0.6               # padding around the face box, fraction of box size
YAW_EDGES = [-0.35, -0.12, 0.12, 0.35]    # 5 yaw buckets from the nose-offset proxy
PITCH_EDGES = [0.48, 0.62]                # 3 pitch buckets from the eye-mouth proxy
DEFAULT_KEEP = 18


def fail(message: str) -> 'NoReturn':
    print(f'identity.py: {message}', file=sys.stderr)
    sys.exit(1)


def decode_frames(video: Path):
    """Yield (frame_index, HxWx3 uint8 RGB) at SAMPLE_FPS via an ffmpeg pipe."""
    import numpy as np
    probe = subprocess.run(
        [shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(video)],
        capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        fail(f'cannot read {video.name}: {probe.stderr.strip()[:200]}')
    width, height = (int(v) for v in probe.stdout.strip().split(',')[:2])
    decode = subprocess.Popen(
        [shutil.which('ffmpeg'), '-v', 'error', '-i', str(video),
         '-vf', f'fps={SAMPLE_FPS}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    index = 0
    while True:
        raw = decode.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        yield index, np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        index += 1
    decode.wait()
```

IMPORTANT pixel-format check for the implementer: before trusting `rgb24`, Read `facefusion/facefusion/vision.py`'s `read_image` and confirm what color order `color_mode='rgb'` actually produces for the detection pipeline (cv2 loads BGR; the function may or may not convert). Match the ffmpeg `-pix_fmt` (`rgb24` vs `bgr24`) to EXACTLY what `read_static_image(path)` (default args) returns, since that is what `get_many_faces` receives during real swaps. Wrong order still detects faces but shifts embeddings — it would silently degrade every downstream score.

```python
def pose_proxies(landmark_5) -> tuple[float, float]:
    """Yaw/pitch proxies from the 5-point landmarks (no extra models).

    yaw: nose x-offset from the eye midpoint, normalized by inter-eye
    distance — 0 frontal, positive when looking right.
    pitch: nose y-position within the eye-to-mouth span — ~0.55 level,
    smaller looking up, larger looking down.
    """
    import numpy as np
    points = np.asarray(landmark_5, dtype=float)
    eye_l, eye_r, nose, mouth_l, mouth_r = points
    mid_eye = (eye_l + eye_r) / 2
    mid_mouth = (mouth_l + mouth_r) / 2
    inter_eye = np.linalg.norm(eye_r - eye_l) or 1.0
    span = (mid_mouth[1] - mid_eye[1]) or 1.0
    yaw = float((nose[0] - mid_eye[0]) / inter_eye)
    pitch = float((nose[1] - mid_eye[1]) / span)
    return yaw, pitch


def bucket_of(yaw: float, pitch: float) -> tuple[int, int]:
    yaw_bucket = sum(yaw > edge for edge in YAW_EDGES)
    pitch_bucket = sum(pitch > edge for edge in PITCH_EDGES)
    return yaw_bucket, pitch_bucket


def sharpness(gray_crop) -> float:
    import cv2
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
```

Candidate collection per video (after `ff_session.boot()`):

```python
def collect(video: Path, video_tag: str) -> list[dict]:
    import cv2
    import numpy as np
    from facefusion.face_creator import get_many_faces
    candidates = []
    for index, frame in decode_frames(video):
        faces = get_many_faces([frame])
        if not faces:
            continue
        face = max(faces, key=lambda f: (f.bounding_box[3] - f.bounding_box[1]))
        x1, y1, x2, y2 = (float(v) for v in face.bounding_box)
        if (y2 - y1) < MIN_FACE_HEIGHT or face.score_set.get('detector', 0) < MIN_DETECTOR_SCORE:
            continue
        # store the padded CROP, never the full frame: 4K frames are ~25 MB
        # each and a 40s clip yields 160 of them — crops keep memory flat
        pad_x = (x2 - x1) * CROP_PAD
        pad_y = (y2 - y1) * CROP_PAD
        crop = frame[max(0, int(y1 - pad_y)):min(frame.shape[0], int(y2 + pad_y)),
                     max(0, int(x1 - pad_x)):min(frame.shape[1], int(x2 + pad_x))].copy()
        if crop.size == 0:
            continue
        face_only = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        yaw, pitch = pose_proxies(face.landmark_set.get('5'))
        candidates.append({
            'video': video_tag, 'frame': index, 'crop': crop,
            'sharp': sharpness(cv2.cvtColor(face_only, cv2.COLOR_RGB2GRAY)),
            'bucket': bucket_of(yaw, pitch), 'yaw': yaw, 'pitch': pitch,
        })
    if candidates:
        floor = np.percentile([c['sharp'] for c in candidates], SHARPNESS_PERCENTILE)
        candidates = [c for c in candidates if c['sharp'] >= floor]
    return candidates
```

(Sharpness is measured on the unpadded face region so background texture never
inflates a blurry face's score; the `.copy()` detaches the crop from the frame
buffer so the frame memory is actually released.)

Selection + writing:

```python
def select(candidates: list[dict], keep: int) -> list[dict]:
    chosen, used = [], set()
    by_bucket: dict[tuple, list] = {}
    for c in sorted(candidates, key=lambda c: -c['sharp']):
        by_bucket.setdefault(c['bucket'], []).append(c)
    for bucket, group in by_bucket.items():        # sharpest per bucket first
        chosen.append(group[0]); used.add(id(group[0]))
    rest = sorted((c for c in candidates if id(c) not in used),
                  key=lambda c: -c['sharp'])
    # round-robin across videos so clip-B expressions survive
    by_video: dict[str, list] = {}
    for c in rest:
        by_video.setdefault(c['video'], []).append(c)
    while len(chosen) < keep and any(by_video.values()):
        for group in by_video.values():
            if group and len(chosen) < keep:
                chosen.append(group.pop(0))
    return chosen[:keep]


def write_person(person_dir: Path, chosen: list[dict]) -> None:
    """Archive the old set, write the new one; restore the old set on any failure."""
    import cv2
    stamp = time.strftime('%Y%m%d-%H%M%S')
    archive = person_dir / f'.old-{stamp}'
    person_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in person_dir.iterdir()
                if p.is_file() and not p.name.startswith('.')]
    if existing:
        archive.mkdir()
        for photo in existing:
            photo.rename(archive / photo.name)
    try:
        for number, c in enumerate(sorted(chosen, key=lambda c: c['bucket']), start=1):
            bgr = cv2.cvtColor(c['crop'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(person_dir / f'photo-{number}.jpg'), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
    except BaseException:
        for photo in person_dir.glob('photo-*.jpg'):
            photo.unlink(missing_ok=True)
        if archive.is_dir():
            for photo in archive.iterdir():
                photo.rename(person_dir / photo.name)
            archive.rmdir()
        raise
```

main(): parse args (`--video` action='append' required, `--person` required, `--keep` int default DEFAULT_KEEP, `--dry-run`); resolve all video paths ABSOLUTE and verify each `is_file()` BEFORE `ff_session.boot()` (boot chdirs); early no-face guard: while collecting the first video, if the first 100 sampled frames yield zero candidates, fail('no face found in the first 100 frames of <name> — wrong clip?'). After collecting all videos: if total kept-candidates < 8 → fail with the dominant filter reason (count how many were dropped by size vs score vs sharpness and name the biggest). Print the report ALWAYS (also on --dry-run):

```
video vlad-angles.mov: 154 frames, 121 faces, 88 after filters
bucket coverage: 11/15 (missing: far-left/up, ...)
kept 18: sharpness 142-489, yaw -0.41..0.44, pitch 0.43..0.71
```

On --dry-run stop there; else `write_person`, then print `faces/<person>: 18 photos written (old set archived to .old-<stamp>)`.

- [ ] **Step 2: Verify pixel format against facefusion's reader**

Per the IMPORTANT note above: read `facefusion/facefusion/vision.py` `read_image`, decide rgb24/bgr24, and leave a one-line comment in `decode_frames` citing the exact vision.py behavior. Sanity: run a real photo through BOTH paths and compare embeddings:

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
import sys
sys.path.insert(0, '.')
import ff_session, numpy as np
ff_session.boot()
from facefusion.vision import read_static_image
from facefusion.face_creator import get_many_faces
photo = str((ff_session.ROOT / 'faces/alexia/photo-1.jpeg'))
ref = get_many_faces([read_static_image(photo)])[0].embedding_norm
import subprocess, shutil
import identity   # decode_frames on a 1-frame "video": jpg works as input to ffmpeg
frame = next(identity.decode_frames(Path(photo)))[1]
alt = get_many_faces([frame])[0].embedding_norm
print('cosine:', float(np.dot(ref, alt)))
EOF
```

Expected: cosine ≥ 0.99. If it's ~0.8 or lower, the pixel order is flipped — fix `-pix_fmt` and re-run.

- [ ] **Step 3: QA on existing media (no vlad clips needed)**

```bash
# dry run on an existing clip with one person
.venv/bin/python identity.py --video clips/first-test.mp4 --person .qa-ident --dry-run
# real run into a scratch person, inspect, then clean up
.venv/bin/python identity.py --video clips/first-test.mp4 --person .qa-ident --keep 12
ls faces/.qa-ident/ && open faces/.qa-ident/photo-1.jpg  # or Read a few crops
rm -rf faces/.qa-ident
```

Expected: dry run prints the report and creates nothing (`ls faces/ | grep qa` empty); real run writes ≤12 sharp, varied crops; report bucket coverage is plausible for the clip (a mostly-frontal clip should land mostly in center buckets — if everything lands in ONE bucket, the yaw/pitch proxies or edges are mis-scaled: print 5 sample yaw/pitch values and adjust YAW_EDGES/PITCH_EDGES). Verify archive/restore: run twice, second run must move the first set into `.old-*`; Ctrl-C mid-write (or simulate by raising) restores.

- [ ] **Step 4: Commit**

```bash
git add identity.py
git commit -m "Build face identities from video with pose-bucketed frame picks"
```

---

### Task 3: bench.py — rank the swappers on a real identity

**Files:**
- Create: `bench.py` (repo root)
- Modify: `.gitignore` (add `bench-targets/` and `bench-results/`)

**Interfaces:**
- Consumes: `ff_session.boot()`, facefusion internals (read_static_image, get_many_faces), swap-style subprocess invocation (PYTHON/venv, cwd=FACEFUSION), `faces/<person>/` directories.
- Produces: CLI `bench.py --face faces/<person> [--targets bench-targets] [--models all|comma,list]`; writes `bench-results/<stamp>/` with `ranking.json`, per-target contact sheets, and prints the ranking table. Task 4 reads the winner from the printed table (human decision).

- [ ] **Step 1: Write bench.py**

```python
#!/usr/bin/env python3
"""Benchmark every zero-shot swapper on one identity, rank by likeness.

Usage:
    .venv/bin/python bench.py --face faces/vlad [--targets bench-targets] [--models all]

Runs each model against the fixed target set (swap only, no enhancer so the
swapper itself is measured), scores ArcFace similarity to the identity's
reference embedding, and writes contact sheets for the eyeball check.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_session

MODELS = ['hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256',
          'ghost_1_256', 'ghost_2_256', 'ghost_3_256',
          'simswap_256', 'inswapper_128_fp16']
PIXEL_BOOST = '1024x1024'   # valid for every model above at the pinned commit
```

Flow in main():
1. Resolve `--face` dir (must contain photos), `--targets` dir, model list (`all` → MODELS). All paths absolute BEFORE boot().
2. Targets: every image in `--targets` (default `ROOT/'bench-targets'`). If the dir is missing or empty, SEED it: copy the first photo of every OTHER person in faces/ (skip the benched person) into `bench-targets/` and print what was seeded — cheap, varied, consent-safe targets; the user can add better ones later. Need ≥ 2 targets or fail.
3. `ff_session.boot()`; compute the reference embedding: mean of `embedding_norm` over every identity photo's most-prominent face, re-normalized (`ref = ref / np.linalg.norm(ref)`). Fail if any identity photo has no detectable face (name it).
4. For each model x target, run the swap as a subprocess (NOT via swap.py — its photo stack hardwires the enhancer):

```python
def run_swap(model: str, target: Path, out: Path, sources: list[str]) -> bool:
    command = [str(ff_session.ROOT / '.venv/bin/python'), 'facefusion.py', 'headless-run',
               '--config-path', str(ff_session.CONFIG),
               '--source-paths', *sources,
               '--target-path', str(target), '--output-path', str(out),
               '--processors', 'face_swapper',
               '--face-swapper-model', model,
               '--face-swapper-pixel-boost', PIXEL_BOOST,
               '--face-selector-mode', 'one',
               '--face-selector-order', 'left-right',
               '--execution-providers', 'coreml',
               '--execution-thread-count', '4',
               '--output-image-quality', '95']
    result = subprocess.run(command, cwd=ff_session.FACEFUSION,
                            capture_output=True, text=True)
    return result.returncode == 0 and out.is_file()
```

   A model that fails (download error, inference crash) is recorded as `"failed"` with the last 200 chars of its output, skipped, and reported at the end — not fatal. First run downloads several models (a few hundred MB each); print a heads-up line before the loop.
5. Score each successful output IN-PROCESS: `faces = get_many_faces([read_static_image(str(out))])`; most-prominent face's `embedding_norm` dot `ref` = cosine similarity; also Laplacian sharpness of the face crop (reuse the same formula as identity.py — copy the 3-line function, do NOT import identity.py, it has argparse at module scope? No — imports are safe, identity.py guards with __main__; still, copy the tiny function to keep the tools independent).
6. Ranking: per model, mean similarity across targets (failed targets excluded), printed sorted:

```
model                 mean-sim   per-target                    sharpness
hyperswap_1c_256      0.71       0.74 0.69 0.71               312
ghost_2_256           0.66       ...
(inswapper_128_fp16   FAILED: <reason>)
```

7. Contact sheet per target (Pillow): row of labelled crops — original target face, then each model's swapped face crop, 256px tall, model name + sim underneath — saved to `bench-results/<stamp>/<target-stem>.jpg`. Write `ranking.json` with the full table. Print the results dir path last.

- [ ] **Step 2: .gitignore**

Add `bench-targets/` and `bench-results/` next to the `photos/` line.

- [ ] **Step 3: QA with an existing identity**

```bash
# fast pass first: 2 models only, auto-seeded targets
.venv/bin/python bench.py --face faces/teo --models hyperswap_1a_256,ghost_2_256
```

Expected: seeds bench-targets/ from alexia/jeni/vlad photos (prints them), runs 2x3 swaps (~2-4 min + first-time model downloads), prints a 2-row ranking with plausible sims (same-ish magnitude, 0.3-0.8 range), writes contact sheets — Read one and confirm labels/crops look right. Then the full run:

```bash
.venv/bin/python bench.py --face faces/teo
```

Expected: 8 rows (or failures explicitly listed), ~10-25 min. `git status` clean (bench dirs ignored).

- [ ] **Step 4: Commit**

```bash
git add bench.py .gitignore
git commit -m "Benchmark the swapper models against a face identity"
```

---

### Task 4: Docs + the vlad run (user-gated)

**Files:**
- Modify: `README.md`
- Modify: `facefusion-swaplab.ini` (only AFTER the vlad benchmark picks a winner)

- [ ] **Step 1: README**

After the photo-mode block in `## Use`, add:

```markdown
## Best identity + picking your swapper

Build a strong multi-angle identity from a short capture video (slow head
turn + expressions, window light, 4K — see the capture checklist in
docs/superpowers/specs/2026-07-31-face-identity-design.md):

​```bash
.venv/bin/python identity.py --video vlad-angles.mov --video vlad-expressions.mov --person vlad
.venv/bin/python bench.py --face faces/vlad     # ranks every swapper on YOUR face
​```

The benchmark winner goes into `facefusion-swaplab.ini` under
`[face_swapper]` so every swap uses it by default (`--swapper-model`
still overrides per run).

Want the absolute ceiling? A personal DFM: train with DeepFaceLab on a
rented NVIDIA GPU (~1-3 days, only your own footage uploaded), export the
.dfm, drop it in `facefusion/.assets/models/custom/`, and swap with
`--processors deep_swapper`. No local training — DFL needs CUDA.
```

(Use plain fences in the actual README — the inner backticks above are escaped only for this plan document.)

- [ ] **Step 2: Commit docs**

```bash
git add README.md
git commit -m "Document identity building and the swapper benchmark"
```

- [ ] **Step 3 (USER-GATED — only when vlad's capture clips are in the iCloud inbox or provided):**

```bash
INBOX=~/Library/Mobile\ Documents/com~apple~CloudDocs/SwapLab/inbox
ls "$INBOX"   # expect vlad-angles.mov / vlad-expressions.mov (or similar)
.venv/bin/python identity.py --video "$INBOX/vlad-angles.mov" --video "$INBOX/vlad-expressions.mov" --person vlad
.venv/bin/python bench.py --face faces/vlad
```

Read the contact sheets, confirm the metric winner also looks best, then write it into the ini:

```ini
[face_swapper]
face_swapper_model = <winner>
```

Verify the default sticks: run one photo swap WITHOUT `--swapper-model` and grep the facefusion invocation/log for the winner model name. Commit:

```bash
git add facefusion-swaplab.ini
git commit -m "Default the swapper to the benchmark winner for vlad"
```

If the clips have not arrived when this task executes, stop after Step 2 and report Task 4 Step 3 as pending-on-user — do NOT fabricate a winner.
