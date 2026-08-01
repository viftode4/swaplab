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
RETRY_PAUSE = 5             # seconds to let the contending workload finish
RETRY_LABEL = 'transient CoreML failure'
# CoreML reports contention as a failed prediction, not as a busy signal
TRANSIENT_MARKERS = ('Unable to compute the prediction', 'CoreML', 'ML Program')
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}   # same set swap.py's IMAGE_EXTS uses


def fail(message: str) -> 'NoReturn':
    print(f'bench.py: {message}', file=sys.stderr)
    sys.exit(1)


def sharpness(gray_crop) -> float:
    import cv2
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def most_prominent(faces):
    # same convention as identity.py: tallest bounding box wins
    return max(faces, key=lambda f: f.bounding_box[3] - f.bounding_box[1])


def crop_from_bbox(frame, bounding_box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bounding_box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def seed_targets(face_dir: Path, targets_dir: Path) -> list[Path]:
    """Copy the first photo of every OTHER person in faces/ into targets_dir."""
    faces_root = face_dir.parent
    seeded = []
    for other in sorted(faces_root.iterdir()):
        if not other.is_dir() or other.name == face_dir.name or other.name.startswith('.'):
            continue
        photos = sorted(p for p in other.iterdir() if p.is_file() and not p.name.startswith('.'))
        if not photos:
            continue
        first = photos[0]
        dest = targets_dir / f'{other.name}{first.suffix}'
        shutil.copy(first, dest)
        seeded.append(dest)
    return seeded


def reference_embedding(face_dir: Path):
    import numpy as np
    from facefusion.vision import read_static_image
    from facefusion.face_creator import get_many_faces

    photos = sorted(p for p in face_dir.iterdir() if p.is_file() and not p.name.startswith('.'))
    if not photos:
        fail(f'no photos in {face_dir}')
    embeddings = []
    for photo in photos:
        frame = read_static_image(str(photo))
        detected = get_many_faces([frame])
        if not detected:
            fail(f'no face detected in {photo}')
        embeddings.append(most_prominent(detected).embedding_norm)
    ref = np.mean(embeddings, axis=0)
    return ref / np.linalg.norm(ref)


def is_transient(tail: str) -> bool:
    """A contended CoreML run looks like a broken model. Tell them apart."""
    return any(marker in (tail or '') for marker in TRANSIENT_MARKERS)


def run_swap(model: str, target: Path, out: Path, sources: list[str]) -> tuple[bool, str]:
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
    ok = result.returncode == 0 and out.is_file()
    tail = ((result.stdout or '') + (result.stderr or ''))[-200:].strip()
    return ok, tail


def score_output(out: Path, ref):
    import numpy as np
    from facefusion.vision import read_static_image
    from facefusion.face_creator import get_many_faces

    frame = read_static_image(str(out))
    detected = get_many_faces([frame])
    if not detected:
        return None
    face = most_prominent(detected)
    sim = float(np.dot(face.embedding_norm, ref))
    crop = crop_from_bbox(frame, face.bounding_box)
    sharp = sharpness(_gray(crop)) if crop is not None else 0.0
    return sim, sharp, crop


def _gray(bgr_crop):
    import cv2
    return cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)


def labeled_cell(bgr_crop, label: str, height: int = 256):
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw

    if bgr_crop is not None and bgr_crop.size:
        crop_height, crop_width = bgr_crop.shape[:2]
        scale = height / crop_height
        resized = cv2.resize(bgr_crop, (max(1, round(crop_width * scale)), height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.zeros((height, height, 3), dtype='uint8')

    face_img = Image.fromarray(rgb)
    caption_height = 34
    canvas = Image.new('RGB', (face_img.width, height + caption_height), 'white')
    canvas.paste(face_img, (0, 0))
    ImageDraw.Draw(canvas).multiline_text((4, height + 4), label, fill='black')
    return canvas


def build_contact_sheet(cells: list[tuple[str, 'np.ndarray | None', 'float | None']]):
    from PIL import Image

    tiles = []
    for label, crop, sim in cells:
        caption = label if sim is None else f'{label}\n{sim:.2f}'
        tiles.append(labeled_cell(crop, caption))
    total_width = sum(tile.width for tile in tiles)
    max_height = max(tile.height for tile in tiles)
    sheet = Image.new('RGB', (total_width, max_height), 'white')
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark swapper models against a face identity.')
    parser.add_argument('--face', required=True, help='faces/<person> directory (the identity).')
    parser.add_argument('--targets', default=None, help='Target images dir (default: ROOT/bench-targets).')
    parser.add_argument('--models', default='all', help='"all" or a comma-separated model list.')
    args = parser.parse_args()

    # resolve + verify BEFORE boot() (boot chdirs into the facefusion checkout)
    face_dir = Path(args.face).resolve()
    if not face_dir.is_dir():
        fail(f'face dir not found: {face_dir}')
    targets_dir = Path(args.targets).resolve() if args.targets else (ff_session.ROOT / 'bench-targets')
    models = MODELS if args.models == 'all' else [m.strip() for m in args.models.split(',')]
    for model in models:
        if model not in MODELS:
            fail(f'unknown model {model!r} — pick from {", ".join(MODELS)}')

    targets_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(p for p in targets_dir.iterdir() if p.is_file() and not p.name.startswith('.'))
    if not targets:
        seeded = seed_targets(face_dir, targets_dir)
        if seeded:
            print(f'seeded {len(seeded)} targets into {targets_dir}: ' +
                  ', '.join(p.name for p in seeded))
        targets = sorted(p for p in targets_dir.iterdir() if p.is_file() and not p.name.startswith('.'))
    if len(targets) < 2:
        fail(f'need >= 2 targets in {targets_dir}, found {len(targets)}')

    sources = [str(p) for p in sorted(face_dir.iterdir())
               if p.is_file() and not p.name.startswith('.')
               and p.suffix.lower() in PHOTO_EXTENSIONS]
    if not sources:
        fail(f'no photos in {face_dir}')

    ff_session.boot()

    from facefusion.vision import read_static_image
    from facefusion.face_creator import get_many_faces

    ref = reference_embedding(face_dir)

    target_crops = {}
    for target in targets:
        frame = read_static_image(str(target))
        detected = get_many_faces([frame])
        target_crops[target.stem] = crop_from_bbox(frame, most_prominent(detected).bounding_box) \
            if detected else None

    stamp = time.strftime('%Y%m%d-%H%M%S')
    results_dir = ff_session.ROOT / 'bench-results' / stamp
    swaps_dir = results_dir / 'swaps'
    results_dir.mkdir(parents=True, exist_ok=True)

    print('first run downloads several models (a few hundred MB each) — this can take a while')

    table = {}
    crops = {}
    for model in models:
        model_dir = swaps_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        per_target = {}
        sharp_values = []
        failed = None
        for target in targets:
            out = model_dir / target.name
            ok, tail = run_swap(model, target, out, sources)
            if not ok and is_transient(tail):
                # CoreML fails the whole prediction when something else on the
                # machine is competing for ANE/GPU ("Unable to compute the
                # prediction using ML Program"). Dropping the model on a first
                # failure silently removes it from the ranking, which is how a
                # contended run crowned inswapper while hyperswap_1b/1c — the
                # two that held facial structure best — vanished entirely.
                print(f'  {model} on {target.name}: {RETRY_LABEL}, retrying',
                      file=sys.stderr, flush=True)
                time.sleep(RETRY_PAUSE)
                ok, tail = run_swap(model, target, out, sources)
            if not ok:
                failed = tail or 'subprocess produced no output'
                break
            scored = score_output(out, ref)
            if scored is None:
                per_target[target.stem] = None
                continue
            sim, sharp, crop = scored
            per_target[target.stem] = sim
            sharp_values.append(sharp)
            crops[(model, target.stem)] = crop
        table[model] = {'failed': failed} if failed else \
            {'per_target': per_target, 'sharpness': sharp_values}

    def mean_sim(row):
        sims = [v for v in row['per_target'].values() if v is not None]
        return sum(sims) / len(sims) if sims else 0.0

    ok_rows = sorted(((m, r) for m, r in table.items() if 'failed' not in r),
                      key=lambda kv: -mean_sim(kv[1]))
    failed_rows = [(m, r) for m, r in table.items() if 'failed' in r]

    if len(ok_rows) < 2:
        for model, row in failed_rows:
            print(f"{model}: FAILED: {row['failed']}", file=sys.stderr)
        fail('fewer than 2 models produced results — no ranking')

    print()
    print('note: mean-sim uses ArcFace, which inswapper directly optimizes — check the contact sheets before crowning it')
    print(f"{'model':<22}{'mean-sim':<11}{'per-target':<30}{'sharpness'}")
    for model, row in ok_rows:
        per_target_str = ' '.join(f'{v:.2f}' if v is not None else '-'
                                   for v in row['per_target'].values())
        mean_sharp = sum(row['sharpness']) / len(row['sharpness']) if row['sharpness'] else 0.0
        print(f"{model:<22}{mean_sim(row):<11.2f}{per_target_str:<30}{mean_sharp:.0f}")
    for model, row in failed_rows:
        print(f"({model:<21}FAILED: {row['failed']})")

    for target in targets:
        cells = [('original', target_crops[target.stem], None)]
        for model, row in ok_rows:
            crop = crops.get((model, target.stem))
            if crop is None:
                continue
            cells.append((model, crop, row['per_target'].get(target.stem)))
        sheet = build_contact_sheet(cells)
        sheet.save(results_dir / f'{target.stem}.jpg', quality=90)

    (results_dir / 'ranking.json').write_text(json.dumps({
        'face': face_dir.name,
        'targets': [t.stem for t in targets],
        'models': table,
    }, indent=2))

    print(results_dir)


if __name__ == '__main__':
    main()
