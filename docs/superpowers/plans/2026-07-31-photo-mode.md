# SwapLab Photo Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Best-quality still-image face swap: CLI photo stack with per-face mapping (`--all-faces`, `--map`, `--list-faces`), and a webapp Clip | Photo flow with tap-to-assign numbered face boxes.

**Architecture:** An image target in `swap.py` auto-branches to a photo-tuned max-quality stack (quality tiers ignored). Per-face mapping runs chained facefusion passes in `reference` selector mode with `left-right` ordering so indices stay stable and match the UI. A standalone `scan_faces.py` reuses facefusion's own detector (pinned commit `3f81a8a`) to print numbered boxes as JSON — the same code path that later picks the reference face, so numbering always agrees. The webapp gains a `photos/` library, detection endpoint, `kind: photo` jobs, and an overlay tap UI.

**Tech Stack:** Python 3.12, FaceFusion (pinned, gitignored), FastAPI, Pillow, macOS `sips` (HEIC), vanilla-JS single-file frontend.

## Global Constraints

- Consent only, label output AI-generated, LAN/Tailscale only (from main spec — keep the existing UI copy visible).
- FaceFusion pinned at commit `3f81a8a78454089d720b8f318a12ae1702c4633b`; `scan_faces.py` may import its internals.
- No new Python dependencies. Pillow is already available (used by captions); `sips`/`ffmpeg` are host tools.
- Commit messages in repo style: imperative sentence, no `feat:` prefixes, **no Co-Authored-By trailer** (user override).
- This repo has no pytest suite; every task verifies by running real commands. `faces/` and `jobs/` are gitignored private media — never commit anything from them.
- Photo indices are ALWAYS left-to-right: every photo-mode facefusion invocation and `scan_faces.py` passes/uses `--face-selector-order left-right`.

## Verification assets (build once, reuse in every task)

QA needs a multi-face photo of consented people. Build one by compositing two existing face-library photos side by side (stays on-device, gitignored path):

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
from PIL import Image
faces = sorted(Path('faces').glob('*/photo-*'))
assert len(faces) >= 2, 'need at least two photos in faces/ — add some via the webapp first'
a, b = Image.open(faces[0]).convert('RGB'), Image.open(faces[-1]).convert('RGB')
h = 800
a = a.resize((int(a.width * h / a.height), h))
b = b.resize((int(b.width * h / b.height), h))
canvas = Image.new('RGB', (a.width + b.width, h), 'white')
canvas.paste(a, (0, 0)); canvas.paste(b, (a.width, 0))
Path('jobs').mkdir(exist_ok=True)
canvas.save('jobs/.qa-group.jpg', quality=95)
print('jobs/.qa-group.jpg', canvas.size)
EOF
```

Also note the two person names for later steps: `ls faces/` (referred to below as `<personA>` and `<personB>`).

---

### Task 1: Photo quality stack in swap.py

**Files:**
- Modify: `swap.py` (QUALITY dict area ~line 77; main() output handling ~lines 570–660)

**Interfaces:**
- Produces: `is_image(path: Path) -> bool` (suffix in IMAGE_EXTS); `PHOTO_STACK: tuple[list[str], list[str]]` shaped exactly like a `QUALITY` value; `convert_image(src: Path, dest: Path) -> None` (Pillow re-save, quality 95); `check_output_image(path: Path, target: Path) -> None` (fails unless openable and same size as target). Later tasks rely on these names.

- [ ] **Step 1: Add the helpers and the photo stack**

After the `QUALITY` dict in `swap.py`, add:

```python
def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


# a still has none of the video constraints (flicker, memory, encode time),
# so photos always run this max stack and ignore the --quality tiers:
# 1024 pixel boost (video: 512), enhancer blend 80 (video caps it at 25
# purely to damp temporal flicker), expression restorer keeps the photo's
# original expression, region masking spares the hairline
PHOTO_STACK = (['face_swapper', 'expression_restorer', 'face_enhancer'], [
    '--face-mask-types', 'box', 'region',
    '--face-mask-blur', '0.4',
    '--face-swapper-pixel-boost', '1024x1024',
    '--expression-restorer-factor', '90',
    '--face-enhancer-blend', '80',
    '--output-image-quality', '95',
    '--face-selector-order', 'left-right',
])
```

Then, next to `check_output_video`, add:

```python
def convert_image(src: Path, dest: Path) -> None:
    """Image-to-image format conversion — the video remux path would
    re-encode a picture with h264 and mangle it."""
    from PIL import Image
    try:
        Image.open(src).convert('RGB').save(dest, quality=95)
    except OSError as error:
        fail(f'could not convert {src.name} to {dest.suffix}: {error}')


def check_output_image(path: Path, target: Path) -> None:
    """Verify the result is a readable image at the target's resolution."""
    from PIL import Image
    try:
        with Image.open(path) as result, Image.open(target) as original:
            if result.size != original.size:
                fail(f'output is {result.size[0]}x{result.size[1]}, '
                     f'expected {original.size[0]}x{original.size[1]}')
    except OSError as error:
        fail(f'output {path} is not a readable image: {error}')
```

- [ ] **Step 2: Branch main() for image targets**

In `main()`, right before `processors, extra = QUALITY[args.quality]`, add:

```python
    photo = is_image(video)
```

and replace that line with:

```python
    if photo:
        if args.quality != 'good':          # 'good' is just the default
            print('photo target: --quality is ignored, photos always run '
                  'the max stack', flush=True)
        processors, extra = PHOTO_STACK
    else:
        processors, extra = QUALITY[args.quality]
```

(`photo = is_image(video)` must be computed on the resolved target — the existing `if video.suffix.lower() not in {...}` guard above it can now read `if not is_image(video):` instead of repeating the extension set; make that swap in both places it occurs in `main()`.)

In the output-handling block near the end of `main()` (the `else:` branch where `work_out != out` remuxes via ffmpeg), route photos to Pillow instead:

```python
        if work_out != out:
            if photo:
                convert_image(work_out, out)
                work_out.unlink(missing_ok=True)
            else:
                ... existing ffmpeg remux code unchanged ...
```

And replace the final validation call:

```python
    if photo:
        check_output_image(out, video)
    else:
        check_output_video(out)
```

(`check_output_video` currently runs unconditionally at `swap.py:657`; the existing `if not args.no_stabilize and video.suffix... ` line can also become `if not args.no_stabilize and not photo:`.)

- [ ] **Step 3: Verify on a real photo**

```bash
.venv/bin/python swap.py --video jobs/.qa-group.jpg \
    --face "faces/$(ls faces | head -1)" --out jobs/.qa-single.png
.venv/bin/python -c "
from PIL import Image
a = Image.open('jobs/.qa-group.jpg'); b = Image.open('jobs/.qa-single.png')
assert a.size == b.size, (a.size, b.size)
print('OK', b.size, b.format)"
```

Expected: swap.py prints the photo-stack notice path (no dedupe/stabilize lines), exits 0, the assert passes, format PNG (converted from facefusion's jpg output by `convert_image`). Open `jobs/.qa-single.png` and eyeball that exactly one face (the most prominent) was swapped and looks sharper than a `--quality best` video frame.

- [ ] **Step 4: Commit**

```bash
git add swap.py
git commit -m "Run photos through a stills-tuned max-quality stack"
```

---

### Task 2: scan_faces.py + swap.py --list-faces

**Files:**
- Create: `scan_faces.py` (repo root, next to swap.py)
- Modify: `swap.py` (argparse + early exit in main())

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `scan_faces.py <image> <config-ini>` prints to stdout ONE JSON object: `{"width": int, "height": int, "faces": [{"index": int, "x": float, "y": float, "w": float, "h": float}, ...]}` — pixel coordinates, faces ordered left-to-right, `index` counting from 0. Exit 1 with `{"error": "..."}` on unreadable image. `swap.py --list-faces --video photo.jpg` proxies exactly that JSON (no `--face`/`--out` required). Task 4's webapp calls `swap.py --list-faces`.

- [ ] **Step 1: Write scan_faces.py**

```python
#!/usr/bin/env python3
"""Print the faces facefusion detects in an image, as JSON, left to right.

Run from inside the facefusion checkout with its venv:
    cd facefusion && ../.venv/bin/python ../scan_faces.py photo.jpg ../facefusion-swaplab.ini

Uses facefusion's own detector (pinned commit) so the numbering agrees
exactly with what --reference-face-position selects during a swap.
"""

import json
import sys


def main() -> None:
    target, config = sys.argv[1], sys.argv[2]
    # feed facefusion's own arg parser so state defaults + our ini apply,
    # exactly as they do during a real swap
    sys.argv = ['facefusion.py', 'headless-run', '--config-path', config,
                '--target-path', target, '--output-path', '/dev/null/unused.jpg',
                '--face-selector-order', 'left-right',
                '--processors', 'face_swapper']
    from facefusion.program import create_program
    from facefusion.args import apply_args
    from facefusion import state_manager
    apply_args(vars(create_program().parse_args()), state_manager.init_item)

    from facefusion.vision import read_static_image
    from facefusion.face_creator import get_many_faces
    from facefusion.face_selector import sort_and_filter_faces

    frame = read_static_image(target)   # same color mode as the swap pipeline
    if frame is None:
        print(json.dumps({'error': f'could not read image: {target}'}))
        sys.exit(1)
    faces = sort_and_filter_faces([], get_many_faces([frame]))
    height, width = frame.shape[:2]
    boxes = []
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = (float(v) for v in face.bounding_box)
        boxes.append({'index': index, 'x': x1, 'y': y1,
                      'w': x2 - x1, 'h': y2 - y1})
    print(json.dumps({'width': width, 'height': height, 'faces': boxes}))


if __name__ == '__main__':
    main()
```

Note: if `face.bounding_box` raises AttributeError at runtime, check the field name on the `Face` type in `facefusion/facefusion/types.py` (the sort helpers in `face_selector.py` reveal it, e.g. `get_bounding_box_left`) and adjust — do not guess a second name without reading that file.

- [ ] **Step 2: Run it directly to verify detection**

```bash
cd facefusion && ../.venv/bin/python ../scan_faces.py \
    ../jobs/.qa-group.jpg ../facefusion-swaplab.ini; cd ..
```

Expected: exit 0, one JSON line with `"faces"` of length 2, index 0 having smaller `x` than index 1 (left-to-right). First run may download the face classifier models — that's fine, it's one-time.

- [ ] **Step 3: Add --list-faces to swap.py**

Argparse: `parser.add_argument('--list-faces', action='store_true', help='detect faces in the target image, print JSON boxes, and exit')`. Also relax the required args: change `--out` to `required=False` and add after parsing:

```python
    if args.list_faces:
        if not is_image(video):
            fail('--list-faces works on images')
        result = subprocess.run(
            [str(PYTHON), str(ROOT / 'scan_faces.py'), str(video), str(CONFIG)],
            cwd=FACEFUSION, capture_output=True, text=True)
        if result.returncode != 0:
            fail(f'face scan failed: {(result.stdout or result.stderr).strip()[-300:]}')
        print(result.stdout.strip().splitlines()[-1])
        return
    if not args.out:
        fail('--out is required')
```

Place this right after the `if not video.is_file(): fail(...)` check, before the `--face`/`--audio` validation (a scan needs neither). Keep `out = Path(args.out)...` working when `args.out` is None: move that line below this block.

- [ ] **Step 4: Verify the proxy**

```bash
.venv/bin/python swap.py --list-faces --video jobs/.qa-group.jpg | .venv/bin/python -m json.tool
```

Expected: pretty-printed JSON, 2 faces, exit 0. Also verify the error path: `.venv/bin/python swap.py --list-faces --video clips/$(ls clips | head -1)` fails with `--list-faces works on images`.

- [ ] **Step 5: Commit**

```bash
git add scan_faces.py swap.py
git commit -m "Add a face scanner that numbers faces left to right"
```

---

### Task 3: --all-faces and --map in swap.py

**Files:**
- Modify: `swap.py` (argparse; the facefusion-invocation section of main(), currently the single `command = [...]` build at ~line 600)

**Interfaces:**
- Consumes: `PHOTO_STACK`, `is_image`, `convert_image`, `check_output_image` (Task 1); `scan_faces.py` JSON shape via `--list-faces` machinery (Task 2).
- Produces: CLI `--all-faces` (with `--face`, photos only) and repeatable `--map N=person-photo-or-dir` (photos only, exclusive with `--face`/`--all-faces`/`--audio`/`--edit` — mapping passes carry no audio/edit). Task 4's webapp worker invokes these.

- [ ] **Step 1: Add the flags and validation**

Argparse additions:

```python
    parser.add_argument('--all-faces', action='store_true',
                        help='swap every face in a photo to the --face person')
    parser.add_argument('--map', action='append', metavar='N=PERSON',
                        help='photo face #N (left to right, from --list-faces) '
                             'becomes this person — photo or directory (repeatable)')
```

Validation, next to the existing `--face` checks in `main()`:

```python
    mappings: list[tuple[int, Path]] = []
    for pair in args.map or []:
        index_raw, _, person_raw = pair.partition('=')
        try:
            index = int(index_raw)
        except ValueError:
            fail(f'--map {pair!r}: face number must be an integer')
        person = Path(person_raw).expanduser().resolve()
        if not (person.is_file() or person.is_dir()):
            fail(f'--map face photo not found: {person}')
        if any(index == seen for seen, _ in mappings):
            fail(f'--map face #{index} given twice')
        mappings.append((index, person))
    if mappings and (face is not None or args.all_faces):
        fail('--map replaces --face/--all-faces — use one or the other')
    if (mappings or args.all_faces) and not is_image(video):
        fail('--all-faces and --map work on photos (image targets) only')
    if args.all_faces and face is None:
        fail('--all-faces needs --face to say who everyone becomes')
    if mappings and (audio is not None or edits):
        fail('--map cannot be combined with --audio or --edit')
```

Update the "nothing to do" guard to accept mappings: `if face is None and audio is None and not edits and not mappings:`.

CRITICAL: further down, main() has a lip-sync-only guard `if face is None: processors, extra = [], []` — that would silently strip the swap processors from `--map` jobs (which pass no `--face`). Change it to `if face is None and not mappings:`.

- [ ] **Step 2: Restructure the invocation into passes**

Extract the current command build + run (the `command = [...]` block through the chunked-or-single run and the `work_out.is_file()` check) into:

```python
def run_facefusion(target: Path, output: Path, processors: list[str],
                   sources: list[str], extra: list[str], cpu: bool) -> None:
    command = [
        str(PYTHON), 'facefusion.py', 'headless-run',
        '--config-path', str(CONFIG),
        *(['--source-paths', *sources] if sources else []),
        '--target-path', str(target),
        '--output-path', str(output),
        '--processors', *processors,
        '--execution-providers', 'cpu' if cpu else 'coreml',
        '--video-memory-strategy', 'moderate',
        '--execution-thread-count', '4',
        *extra,
    ]
    frame_count = video_frames(target) if not is_image(target) else 0
    if frame_count > CHUNK_FRAMES * 1.5:
        run_in_chunks(command, output, frame_count)
    else:
        result = subprocess.run(command, cwd=FACEFUSION)
        if result.returncode != 0:
            fail(f'facefusion exited with {result.returncode} '
                 '(no face in photo/video and codec issues are the usual causes)')
    if not output.is_file():
        fail('facefusion reported success but produced no output file')
```

Note `--face-selector-mode` moves OUT of the shared command and into per-mode `extra` (video keeps `['--face-selector-mode', 'one']` appended where `QUALITY` extras are assembled; comments about thread count/selector mode move with their lines).

The calling code in `main()` becomes:

```python
    selector = ['--face-selector-mode', 'one']
    if args.all_faces:
        selector = ['--face-selector-mode', 'many']
    if mappings:
        count_json = json.loads(subprocess.run(
            [str(PYTHON), str(ROOT / 'scan_faces.py'), str(video), str(CONFIG)],
            cwd=FACEFUSION, capture_output=True, text=True).stdout.splitlines()[-1])
        found = len(count_json.get('faces', []))
        for index, _ in mappings:
            if not 0 <= index < found:
                fail(f'--map face #{index}: the photo has {found} face(s), '
                     f'numbered 0..{found - 1} left to right')
        current, temps = video, []
        for pass_number, (index, person) in enumerate(mappings):
            last = pass_number == len(mappings) - 1
            step_out = work_out if last else \
                work_out.with_name(f'.{work_out.stem}.pass{pass_number}{work_out.suffix}')
            pass_sources = [str(p) for p in face_photos(person)]
            try:
                run_facefusion(current, step_out, processors, pass_sources,
                               [*extra, '--face-selector-mode', 'reference',
                                '--reference-face-position', str(index)],
                               args.cpu)
            except SystemExit:
                for temp in temps:
                    temp.unlink(missing_ok=True)
                print(f'swap.py: face #{index} ({person.name}) was the pass '
                      'that failed', file=sys.stderr)
                raise
            if not last:
                temps.append(step_out)
            current = step_out
        for temp in temps:
            temp.unlink(missing_ok=True)
    else:
        run_facefusion(video, work_out, processors, sources,
                       [*extra, *selector], args.cpu)
```

(`import json` at top of swap.py. `processors`/`extra` here are the PHOTO_STACK values from Task 1; `sources` stays as built today for the non-map path.)

- [ ] **Step 3: Verify all three photo modes end to end**

```bash
P1=faces/$(ls faces | head -1); P2=faces/$(ls faces | tail -1)
# biggest face (unchanged default)
.venv/bin/python swap.py --video jobs/.qa-group.jpg --face "$P1" --out jobs/.qa-one.jpg
# everyone
.venv/bin/python swap.py --video jobs/.qa-group.jpg --face "$P1" --all-faces --out jobs/.qa-all.jpg
# per-face mapping, both directions
.venv/bin/python swap.py --video jobs/.qa-group.jpg \
    --map "0=$P2" --map "1=$P1" --out jobs/.qa-map.jpg
# error paths
.venv/bin/python swap.py --video jobs/.qa-group.jpg --map "5=$P1" --out /tmp/x.jpg; echo "exit=$?"
.venv/bin/python swap.py --video jobs/.qa-group.jpg --all-faces --out /tmp/x.jpg; echo "exit=$?"
```

Expected: three output photos exist; `.qa-all.jpg` has BOTH faces as person A; `.qa-map.jpg` has left face as person B and right face as person A (eyeball them); `--map 5=` fails naming `numbered 0..1`; `--all-faces` without `--face` fails with its message; no stray `.pass0` temp files: `ls jobs/.qa-map* jobs/.*pass* 2>/dev/null`.

Also confirm video regression: run one short existing clip through `--quality fast` and check it still completes (`ls clips/` for a small one).

- [ ] **Step 4: Commit**

```bash
git add swap.py
git commit -m "Swap every face or map each face to a chosen person in photos"
```

---

### Task 4: Webapp backend photo support

**Files:**
- Modify: `webapp/app.py`

**Interfaces:**
- Consumes: `swap.py --list-faces` JSON (Task 2), `--all-faces`/`--map` flags (Task 3).
- Produces (Task 5's frontend calls these):
  - `GET /api/photos` → `[{"name": str, "file": str}]`, newest first
  - `POST /api/photos` (multipart `photo`) → `{"name", "file", "width", "height", "faces": [...]}` — stores in `photos/`, HEIC converted to JPEG via `sips`, EXIF baked, detection run immediately; **400 "no faces found in that photo"** when zero faces
  - `GET /api/photos/{file_name}` → the full-size image (for the tap UI)
  - `GET /api/photos/{file_name}/faces` → same JSON shape as scan_faces (cached)
  - `POST /api/jobs` gains `photo_name: str = Form('')` and `mapping: str = Form('')`; mapping is JSON: `{"all": "<person>"}` or `{"<index>": "<person>", ...}`; photo jobs write `kind: "photo"` into job.json
  - `GET /api/jobs/{id}/result` and `/preview` serve `result.jpg`/`result.png` with image media types for photo jobs

- [ ] **Step 1: Photo library storage + detection**

Add near the top constants: `PHOTOS = ROOT / 'photos'`; `HEIC_EXTS = {'.heic', '.heif'}`. Then:

```python
def sips_to_jpeg(src: Path, dest: Path) -> bool:
    """HEIC → JPEG with macOS's own converter; stays on-device."""
    result = subprocess.run(
        ['sips', '-s', 'format', 'jpeg', '-s', 'formatOptions', '95',
         str(src), '--out', str(dest)], capture_output=True)
    return result.returncode == 0 and dest.is_file()


def scan_photo(photo: Path) -> dict:
    """Detect faces via swap.py --list-faces; cache beside the photo."""
    cache = photo.with_name(f'.{photo.stem}.faces.json')
    if cache.is_file() and cache.stat().st_mtime >= photo.stat().st_mtime:
        return json.loads(cache.read_text())
    result = subprocess.run(
        [str(PYTHON), str(SWAP), '--list-faces', '--video', str(photo)],
        capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise HTTPException(500, 'face detection failed — check the server log')
    data = json.loads(result.stdout.strip().splitlines()[-1])
    cache.write_text(json.dumps(data))
    return data


def list_photos() -> list[dict]:
    photos = []
    if PHOTOS.is_dir():
        entries = sorted(PHOTOS.iterdir(), key=lambda p: p.stat().st_mtime,
                         reverse=True)
        for entry in entries:
            if entry.suffix.lower() in IMAGE_EXTS and not entry.name.startswith('.'):
                photos.append({'name': entry.stem, 'file': entry.name})
    return photos
```

- [ ] **Step 2: Photo endpoints**

```python
@app.get('/api/photos')
def api_photos() -> list[dict]:
    return list_photos()


@app.post('/api/photos')
async def api_add_photo(photo: UploadFile = File(...)) -> dict:
    ext = Path(photo.filename or '').suffix.lower()
    if ext not in IMAGE_EXTS | HEIC_EXTS:
        raise HTTPException(400, f'photo must be one of {sorted(IMAGE_EXTS | HEIC_EXTS)}')
    base = slug(Path(photo.filename or 'photo').stem)
    PHOTOS.mkdir(exist_ok=True)
    dest = PHOTOS / f'{base}{ext}'
    if dest.exists():
        dest = PHOTOS / f'{base}-{secrets.token_hex(2)}{ext}'
    await save_upload(photo, dest)
    if ext in HEIC_EXTS:
        jpeg = dest.with_suffix('.jpg')
        if not sips_to_jpeg(dest, jpeg):
            dest.unlink(missing_ok=True)
            raise HTTPException(400, 'could not convert that HEIC — try exporting as JPEG')
        dest.unlink(missing_ok=True)
        dest = jpeg
    normalize_photo(dest)
    scan = scan_photo(dest)
    if not scan.get('faces'):
        dest.unlink(missing_ok=True)
        raise HTTPException(400, 'no faces found in that photo')
    return {'name': dest.stem, 'file': dest.name, **scan}


def photo_path(file_name: str) -> Path:
    photo = (PHOTOS / file_name).resolve()
    if photo.parent != PHOTOS.resolve() or not photo.is_file():
        raise HTTPException(404)
    return photo


@app.get('/api/photos/{file_name}')
def api_photo(file_name: str) -> FileResponse:
    return FileResponse(photo_path(file_name))


@app.get('/api/photos/{file_name}/faces')
def api_photo_faces(file_name: str) -> dict:
    return scan_photo(photo_path(file_name))
```

Add `PHOTOS.mkdir(exist_ok=True)` in `start_worker()` next to the other mkdirs.

- [ ] **Step 3: Photo jobs — creation**

In `api_create_job`, add parameters `photo_name: str = Form('')` and `mapping: str = Form('')`. Before the existing video/clip branch, insert the photo branch (a photo job takes this path and skips clip handling entirely):

```python
    if photo_name:
        source = photo_path(photo_name)     # 404s on nonsense
        try:
            plan = json.loads(mapping) if mapping else {}
        except ValueError:
            raise HTTPException(400, 'mapping must be a JSON object')
        if not isinstance(plan, dict) or not plan:
            raise HTTPException(400, 'assign at least one face')
        scan = scan_photo(source)
        found = len(scan['faces'])
        persons = {}
        for key, person_name in plan.items():
            if key != 'all':
                try:
                    index = int(key)
                except ValueError:
                    raise HTTPException(400, f'bad face number: {key}')
                if not 0 <= index < found:
                    raise HTTPException(400, f'face #{key} not found — the photo has {found}')
            person = FACES / str(person_name)
            photos = person_photos(person) if person.is_dir() else []
            if not photos:
                raise HTTPException(400, f'unknown face: {person_name}')
            persons[key] = person
        if 'all' in plan and len(plan) > 1:
            raise HTTPException(400, 'use "all" alone, or number the faces')

        job_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}'
        path = JOBS / job_id
        path.mkdir(parents=True)
        shutil.copy(source, path / f'input{source.suffix.lower()}')
        for key, person in persons.items():
            slot = path / ('face' if key == 'all' else f'face-{key}')
            slot.mkdir()
            for photo_file in person_photos(person):
                shutil.copy(photo_file, slot / photo_file.name)
        write_job(path, {
            'id': job_id, 'kind': 'photo', 'status': 'queued',
            'quality': 'best',
            'face': ', '.join(f'{k}→{plan[k]}' for k in sorted(plan)),
            'video': photo_name, 'audio': None, 'captions': False,
            'edit': None, 'screen_recording': False,
            'created': time.time(), 'error': None,
        })
        wake.set()
        return {'id': job_id}
```

Fix the guard line above — write it plainly (the one-liner above is a placeholder trap; use this):

```python
            photos = person_photos(person) if person.is_dir() else []
            if not photos:
                raise HTTPException(400, f'unknown face: {person_name}')
```

- [ ] **Step 4: Photo jobs — worker + result endpoints**

In `run_job`, branch by kind at the top (before the current `video/face` lookup):

```python
    if data.get('kind') == 'photo':
        photo_file = next((f for f in path.iterdir() if f.stem == 'input'), None)
        if not photo_file:
            data.update(status='failed', error='job folder is missing input files')
            write_job(path, data)
            return
        result_file = path / f'result{photo_file.suffix.lower()}'
        command = [str(PYTHON), str(SWAP), '--video', str(photo_file),
                   '--out', str(result_file)]
        all_dir = path / 'face'
        if all_dir.is_dir():
            command += ['--face', str(all_dir), '--all-faces']
        else:
            for slot in sorted(path.glob('face-*')):
                command += ['--map', f'{slot.name.split("-", 1)[1]}={slot}']
    else:
        ... existing video command build, result_file = path / 'result.mp4' ...
```

Unify the tail: the existing success check `(path / 'result.mp4').is_file()` becomes `result_file.is_file()`, and `make_preview(path)` runs only for video jobs (`if data.get('kind') != 'photo': make_preview(path)`).

Result endpoints — replace the body of `api_job_result`:

```python
@app.get('/api/jobs/{job_id}/result')
def api_job_result(job_id: str) -> FileResponse:
    path = job_dir(job_id)
    for name, media in (('result.mp4', 'video/mp4'), ('result.jpg', 'image/jpeg'),
                        ('result.jpeg', 'image/jpeg'), ('result.png', 'image/png'),
                        ('result.webp', 'image/webp')):
        result = path / name
        if result.is_file():
            return FileResponse(result, media_type=media,
                                filename=f'swap-{job_id}{result.suffix}')
    raise HTTPException(404, 'no result yet')
```

(`api_job_preview` already falls through to `api_job_result` when `preview.mp4` is missing — photo jobs never create one, so no change needed there.)

- [ ] **Step 5: Verify the API end to end with curl**

Start the app (`.venv/bin/python webapp/app.py --port 8878` in background), then:

```bash
curl -sf -F "photo=@jobs/.qa-group.jpg" http://localhost:8878/api/photos | .venv/bin/python -m json.tool
# → name/file + 2 faces. Note the "file" value as $F and two person names.
curl -sf http://localhost:8878/api/photos/$F/faces | .venv/bin/python -m json.tool
curl -sf -F "photo_name=$F" -F 'mapping={"0": "<personA>", "1": "<personB>"}' \
     http://localhost:8878/api/jobs
# poll api/jobs until status done, then:
curl -sfI http://localhost:8878/api/jobs/<id>/result | grep -i content-type   # image/jpeg
# error paths:
curl -s -F "photo_name=$F" -F 'mapping={"9": "<personA>"}' http://localhost:8878/api/jobs   # 400 face #9
curl -s -F "photo_name=$F" -F 'mapping={}' http://localhost:8878/api/jobs                   # 400 assign at least one
```

Also queue one video job through the old form fields to confirm no regression, then stop the server.

- [ ] **Step 6: Commit**

```bash
git add webapp/app.py
git commit -m "Teach the webapp photo jobs with per-face mapping"
```

---

### Task 5: Webapp frontend — Photo mode

**Files:**
- Modify: `webapp/static/index.html`

**Interfaces:**
- Consumes: all Task 4 endpoints, exactly as specified there.
- Produces: user-facing Clip | Photo flow.

Design (keep the existing visual language — chips, `--go` accent, step labels):

- [ ] **Step 1: Mode toggle + photo library**

Step 2's header becomes a toggle. Replace `<div class="step"><b>2</b>onto which clip</div>` with:

```html
<div class="step"><b>2</b>onto which
  <button class="chip mini modechip sel" data-mode="clip">clip</button>
  <button class="chip mini modechip" data-mode="photo">photo</button>
</div>
```

Below the existing `<div class="faces clips" id="clips"></div>` add:

```html
<div class="faces clips" id="photos" hidden></div>
<div id="photoStage" hidden>
  <div class="stagewrap"><img id="stageImg" alt=""></div>
  <div class="faceinfo" id="mapInfo"></div>
</div>
```

CSS additions:

```css
.modechip { margin-left: 8px; text-transform: none; letter-spacing: 0; }
.stagewrap { position: relative; margin-top: 10px; }
.stagewrap img { width: 100%; border-radius: 6px; display: block; }
.facebox {
  position: absolute; border: 2px solid var(--go); border-radius: 6px;
  background: none; color: var(--go); font-weight: 800; font-size: 14px;
  display: grid; place-items: start start; padding: 2px 6px;
}
.facebox.sel { background: rgba(200, 255, 46, .18); }
.facebox.mapped { border-style: solid; }
.facebox i { font-style: normal; background: #101203cc; border-radius: 4px;
             padding: 1px 6px; }
```

JS state additions: `state.mode = 'clip'; state.photo = null; state.photoScan = null; state.mapping = {}; state.selBox = null;`

Mode toggle handler:

```js
for (const chip of document.querySelectorAll('.modechip')) {
  chip.onclick = () => {
    state.mode = chip.dataset.mode;
    for (const c of document.querySelectorAll('.modechip'))
      c.classList.toggle('sel', c === chip);
    $('clips').hidden = state.mode !== 'clip';
    $('photos').hidden = state.mode !== 'photo';
    $('photoStage').hidden = state.mode !== 'photo' || !state.photo;
    document.querySelector('.quality').parentNode.style.display = '';  // see step 3
    paintMode(); ready();
  };
}
```

`loadPhotos()` mirrors `loadClips()`: fetch `api/photos`, tiles use `api/photos/${file}` as `img src` (the photo is its own thumbnail), an upload tile with `accept="image/*,.heic,.heif"` calling `uploadPhoto(file)` — an XHR clone of `uploadClip` posting to `api/photos`, and on success calling `selectPhoto(saved)`.

- [ ] **Step 2: Face boxes + tap-to-assign**

```js
async function selectPhoto(p) {
  state.photo = p.file;
  state.photoScan = p.faces ? p : await fetch(`api/photos/${encodeURIComponent(p.file)}/faces`)
      .then(r => r.json()).then(scan => ({...p, ...scan}));
  state.mapping = {}; state.selBox = null;
  $('photoStage').hidden = false;
  $('stageImg').src = `api/photos/${encodeURIComponent(p.file)}`;
  paintBoxes(); loadPhotos(); ready();
}

function paintBoxes() {
  const wrap = document.querySelector('.stagewrap');
  for (const old of wrap.querySelectorAll('.facebox')) old.remove();
  const scan = state.photoScan;
  if (!scan) return;
  for (const f of scan.faces) {
    const b = document.createElement('button');
    b.className = 'facebox' + (state.selBox === f.index ? ' sel' : '')
                + (state.mapping[f.index] ? ' mapped' : '');
    b.style.left = `${100 * f.x / scan.width}%`;
    b.style.top = `${100 * f.y / scan.height}%`;
    b.style.width = `${100 * f.w / scan.width}%`;
    b.style.height = `${100 * f.h / scan.height}%`;
    b.innerHTML = `<i>${f.index + 1}${state.mapping[f.index] ? ' → ' + state.mapping[f.index] : ''}</i>`;
    b.onclick = () => { state.selBox = state.selBox === f.index ? null : f.index;
                        delete state.mapping.all; paintBoxes(); paintMode(); };
    wrap.appendChild(b);
  }
  paintMode();
}
```

Assignment hooks into the EXISTING face tiles: in `loadFaces()`, change the tile `onclick` to:

```js
    b.onclick = () => {
      if (state.mode === 'photo' && state.photo != null && state.selBox != null) {
        state.mapping[state.selBox] = f.name;
        state.selBox = null;
        paintBoxes();
      }
      state.face = f.name; state.newFace = null; loadFaces(); ready();
    };
```

`paintMode()` writes instructions into `#mapInfo`:

```js
function paintMode() {
  if (state.mode !== 'photo' || !state.photoScan) return;
  const n = Object.keys(state.mapping).filter(k => k !== 'all').length;
  $('mapInfo').innerHTML = state.mapping.all
    ? `everyone becomes <b>${state.mapping.all}</b> `
    : state.selBox != null
      ? `now tap a face up top to assign face ${state.selBox + 1} `
      : n ? `${n} face${n > 1 ? 's' : ''} assigned — tap SWAP IT `
          : `tap a numbered face, then tap whose face goes in `;
  const all = document.createElement('button');
  all.className = 'chip mini'; all.textContent = state.mapping.all ? 'undo everyone' : '👥 everyone';
  all.onclick = () => {
    if (state.mapping.all) { delete state.mapping.all; }
    else if (state.face) { state.mapping = { all: state.face }; state.selBox = null; }
    else { report('photo mode', 'pick whose face goes in first (step 1)'); return; }
    paintBoxes();
  };
  $('mapInfo').appendChild(all);
  ready();
}
```

- [ ] **Step 3: Submit + queue rendering + hide video-only controls**

In photo mode hide quality + extras (photos are always max quality; captions/voice/screen-recording are video-only). Add `class="videoonly"` (alongside existing classes) to: the step-3 label div, `#quality`, the step-4 label div, and both `.extras` divs — but NOT `#editor`, which has its own chip-toggled `hidden` that the loop would clobber. Then in the mode toggle handler:

```js
    const photoMode = state.mode === 'photo';
    for (const el of document.querySelectorAll('.videoonly')) el.hidden = photoMode;
    if (photoMode) $('editor').hidden = true;   // chip reopens it in clip mode
```

`ready()` becomes mode-aware:

```js
function ready() {
  $('go').disabled = state.mode === 'photo'
    ? !(state.photo && (state.mapping.all || Object.keys(state.mapping).length))
    : !((state.video || state.clip) && state.face);
}
```

In `$('go').onclick`, branch the FormData build:

```js
  const fd = new FormData();
  if (state.mode === 'photo') {
    fd.append('photo_name', state.photo);
    fd.append('mapping', JSON.stringify(state.mapping));
  } else {
    ... existing video/clip/quality/face/audio/captions/screen/edit appends ...
  }
```

After a successful photo submit, clear `state.mapping = {}` (keep the photo selected for a rerun with different people).

Queue strip: done photo jobs get an inline thumbnail instead of the video save/HD pair. In `poll()`:

```js
    const link = j.status === 'done'
      ? (j.kind === 'photo'
          ? `<a href="api/jobs/${j.id}/result" download>save</a>`
          : `<a href="api/jobs/${j.id}/preview" download>save</a>
             <a class="hd" href="api/jobs/${j.id}/result" download>HD</a>`)
      : '';
```

and for done photo jobs prepend a small preview image into the row: `div.innerHTML = (j.kind === 'photo' && j.status === 'done' ? `<img class="jobthumb" src="api/jobs/${j.id}/result">` : spin) + ...` with CSS `.jobthumb { width: 44px; height: 44px; object-fit: cover; border-radius: 6px; flex-shrink: 0; }` (replaces the spinner slot, which is empty for done jobs anyway).

- [ ] **Step 4: Verify in a real browser**

Start the app, open it on the Mac (`http://localhost:8877`), and walk the QA list:

1. Toggle to Photo — clips hide, quality/extras hide, photo library shows.
2. Upload `jobs/.qa-group.jpg` — numbered boxes 1 and 2 appear over the right faces (left face is 1).
3. Tap box 1 → tap person B → box shows `1 → <personB>`; tap box 2 → person A.
4. SWAP IT → job appears with `0→<personB>, 1→<personA> → <photo>`, finishes, thumbnail + save work, saved image opens.
5. Everyone: reselect photo, tap 👥 everyone with a person selected, submit, verify both faces swapped.
6. Toggle back to Clip — old flow intact (queue a fast video job).
7. Upload a HEIC (`sips -s format heic jobs/.qa-group.jpg --out /tmp/qa.heic` to make one) — converts, detects, swaps.
8. Upload a no-face image (any screenshot without faces) — clean red error "no faces found in that photo", no job created.

From the phone (same LAN), repeat 2–4 once to confirm touch targets work.

- [ ] **Step 5: Commit**

```bash
git add webapp/static/index.html
git commit -m "Add a photo mode with tap-to-assign face mapping"
```

---

### Task 6: Docs + final QA sweep

**Files:**
- Modify: `README.md` (Use + Webapp sections)

- [ ] **Step 1: Document the photo mode**

In README `## Use`, after the existing quality line, add:

```markdown
Photos work too — an image target always runs the max-quality stills stack
(`--quality` is ignored):

```bash
.venv/bin/python swap.py --video group.jpg --face vlad.jpg --out swapped.jpg
# --all-faces            everyone in the photo becomes --face
# --map 0=ana --map 2=vlad   face #N (left to right) becomes that person
# --list-faces           print the numbered face boxes as JSON
```
```

In the Webapp section, mention the Photo mode: pick or upload a picture (HEIC fine), tap a numbered face, tap whose face goes in, or 👥 everyone.

- [ ] **Step 2: Full QA matrix re-run**

Re-run the spec's QA list end to end on a fresh server start: single-face photo, group per-face map, all-faces, HEIC, PNG in/PNG out (`--out x.png`), resolution equality (Task 1's assert), video regression (one fast clip job through the webapp). Fix anything that fails before committing.

- [ ] **Step 3: Clean up QA artifacts and commit**

```bash
rm -f jobs/.qa-*.jpg jobs/.qa-*.png /tmp/qa.heic
git add README.md
git commit -m "Document the photo mode"
```
