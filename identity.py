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
YAW_NAMES = ['far-left', 'left', 'center', 'right', 'far-right']
PITCH_NAMES = ['up', 'level', 'down']
DEFAULT_KEEP = 18
MIN_TOTAL_CANDIDATES = 8
EARLY_GUARD_FRAMES = 100


def fail(message: str) -> 'NoReturn':
    print(f'identity.py: {message}', file=sys.stderr)
    sys.exit(1)


def decode_frames(video: Path):
    """Yield (frame_index, HxWx3 uint8 BGR) at SAMPLE_FPS via an ffmpeg pipe.

    facefusion.vision.read_image()'s default color_mode='rgb' is a misnomer:
    it only swaps to cv2.IMREAD_UNCHANGED for 'rgba'; for 'rgb' (the default
    read_static_image uses) it stays cv2.IMREAD_COLOR with no BGR->RGB
    conversion, so cv2.imread's native BGR order is what get_many_faces sees
    on every real swap. Decode bgr24 here to match exactly (see task-2
    pixel-format check, cosine >= 0.99 against read_static_image).
    """
    import numpy as np
    probe = subprocess.run(
        [shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(video)],
        capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        fail(f'cannot read {video.name}: {probe.stderr.strip()[:200]}')
    width, height = (int(v) for v in probe.stdout.strip().split(',')[:2])
    # ffprobe's stream=width,height reports CODED dims, but ffmpeg's decode
    # below applies the display-matrix rotation, so a rotation=+-90 clip
    # emits height x width frames — swap dims here or the reshape shears.
    rotation_probe = subprocess.run(
        [shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream_side_data=rotation', '-of', 'csv=p=0', str(video)],
        capture_output=True, text=True)
    rotation_text = rotation_probe.stdout.strip().splitlines()[0].strip() \
        if rotation_probe.stdout.strip() else ''
    try:
        rotation = int(float(rotation_text)) if rotation_text else 0
    except ValueError:
        rotation = 0
    if abs(rotation) % 180 == 90:
        width, height = height, width
    decode = subprocess.Popen(
        [shutil.which('ffmpeg'), '-v', 'error', '-i', str(video),
         '-vf', f'fps={SAMPLE_FPS}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
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


def bucket_name(bucket: tuple[int, int]) -> str:
    yaw_bucket, pitch_bucket = bucket
    return f'{YAW_NAMES[yaw_bucket]}/{PITCH_NAMES[pitch_bucket]}'


def sharpness(gray_crop) -> float:
    import cv2
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def collect(video: Path, video_tag: str, early_guard: bool = False) -> tuple[list[dict], dict]:
    import cv2
    import numpy as np
    from facefusion.face_creator import get_many_faces
    candidates = []
    frames = faces = dropped_size = dropped_score = 0
    for index, frame in decode_frames(video):
        frames += 1
        detected = get_many_faces([frame])
        if detected:
            faces += 1
            face = max(detected, key=lambda f: (f.bounding_box[3] - f.bounding_box[1]))
            x1, y1, x2, y2 = (float(v) for v in face.bounding_box)
            if (y2 - y1) < MIN_FACE_HEIGHT:
                dropped_size += 1
            elif face.score_set.get('detector', 0) < MIN_DETECTOR_SCORE:
                dropped_score += 1
            else:
                # store the padded CROP, never the full frame: 4K frames are ~25 MB
                # each and a 40s clip yields 160 of them — crops keep memory flat
                pad_x = (x2 - x1) * CROP_PAD
                pad_y = (y2 - y1) * CROP_PAD
                crop = frame[max(0, int(y1 - pad_y)):min(frame.shape[0], int(y2 + pad_y)),
                             max(0, int(x1 - pad_x)):min(frame.shape[1], int(x2 + pad_x))].copy()
                if crop.size != 0:
                    face_only = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                    yaw, pitch = pose_proxies(face.landmark_set.get('5'))
                    candidates.append({
                        'video': video_tag, 'frame': index, 'crop': crop,
                        'sharp': sharpness(cv2.cvtColor(face_only, cv2.COLOR_BGR2GRAY)),
                        'bucket': bucket_of(yaw, pitch), 'yaw': yaw, 'pitch': pitch,
                    })
        if early_guard and frames == EARLY_GUARD_FRAMES and not candidates:
            fail(f'no face found in the first {EARLY_GUARD_FRAMES} frames of {video.name} — wrong clip?')

    dropped_sharpness = 0
    if candidates:
        floor = np.percentile([c['sharp'] for c in candidates], SHARPNESS_PERCENTILE)
        kept = [c for c in candidates if c['sharp'] >= floor]
        dropped_sharpness = len(candidates) - len(kept)
        candidates = kept
    stats = {
        'name': video.name, 'frames': frames, 'faces': faces,
        'after_filters': len(candidates), 'dropped_size': dropped_size,
        'dropped_score': dropped_score, 'dropped_sharpness': dropped_sharpness,
    }
    return candidates, stats


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


def write_person(person_dir: Path, chosen: list[dict]) -> 'Path | None':
    """Write the new set into a staging dir, archive the old set, then swap in.

    New crops are written to a sibling `.new-<stamp>/` dir first, so the
    person dir never contains a half-written JPEG — the window where the
    person dir is touched shrinks to pure renames. On ANY failure, staged
    and partial files are removed and the old set is fully restored.
    Returns the archive dir if an old set existed, else None.
    """
    import cv2
    stamp = time.strftime('%Y%m%d-%H%M%S')
    staging = person_dir / f'.new-{stamp}'
    archive = person_dir / f'.old-{stamp}'
    person_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    existing = []
    archived = False
    try:
        for number, c in enumerate(sorted(chosen, key=lambda c: c['bucket']), start=1):
            # crop is already BGR (decode_frames emits bgr24), which is what
            # cv2.imwrite expects — no color conversion needed here.
            cv2.imwrite(str(staging / f'photo-{number}.jpg'), c['crop'],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        existing = [p for p in person_dir.iterdir()
                    if p.is_file() and not p.name.startswith('.')]
        if existing:
            archive.mkdir(exist_ok=True)
            for photo in existing:
                photo.rename(archive / photo.name)
        archived = True   # existing files (if any) are now fully in `archive`
        for photo in staging.iterdir():
            photo.rename(person_dir / photo.name)
        staging.rmdir()
    except BaseException:
        for photo in staging.iterdir():
            photo.unlink(missing_ok=True)
        if staging.is_dir():
            staging.rmdir()
        if archived:
            # only reachable via the swap-in loop above, so every photo-*.jpg
            # left in person_dir at this point is a freshly renamed new file —
            # originals (if any) already moved to `archive` in full
            for photo in person_dir.glob('photo-*.jpg'):
                photo.unlink(missing_ok=True)
        if archive.is_dir():
            for photo in archive.iterdir():
                photo.rename(person_dir / photo.name)
            archive.rmdir()
        raise
    return archive if existing else None


def build_report(video_stats: list[dict], all_candidates: list[dict], chosen: list[dict]) -> str:
    lines = []
    for vs in video_stats:
        lines.append(f"video {vs['name']}: {vs['frames']} frames, {vs['faces']} faces, "
                     f"{vs['after_filters']} after filters")

    all_buckets = [(y, p) for y in range(len(YAW_NAMES)) for p in range(len(PITCH_NAMES))]
    covered = {c['bucket'] for c in all_candidates}
    missing = [bucket_name(b) for b in all_buckets if b not in covered]
    coverage_line = f'bucket coverage: {len(covered)}/{len(all_buckets)}'
    if missing:
        coverage_line += f" (missing: {', '.join(missing)})"
    lines.append(coverage_line)

    if chosen:
        sharps = [c['sharp'] for c in chosen]
        yaws = [c['yaw'] for c in chosen]
        pitches = [c['pitch'] for c in chosen]
        lines.append(
            f'kept {len(chosen)}: sharpness {min(sharps):.0f}-{max(sharps):.0f}, '
            f'yaw {min(yaws):.2f}..{max(yaws):.2f}, pitch {min(pitches):.2f}..{max(pitches):.2f}'
        )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a face identity from capture video.')
    parser.add_argument('--video', action='append', required=True, dest='videos',
                        help='Path to a capture video; repeatable.')
    parser.add_argument('--person', required=True, help='Person name (faces/<person>/).')
    parser.add_argument('--keep', type=int, default=DEFAULT_KEEP)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.keep < 1:
        fail('--keep must be at least 1')

    # resolve + verify BEFORE boot() (boot chdirs into the facefusion checkout)
    videos = []
    for raw in args.videos:
        path = Path(raw).resolve()
        if not path.is_file():
            fail(f'video not found: {path}')
        videos.append(path)

    ff_session.boot()

    all_candidates = []
    video_stats = []
    for video in videos:
        candidates, stats = collect(video, video.name, early_guard=True)
        all_candidates.extend(candidates)
        video_stats.append(stats)

    total_kept = sum(vs['after_filters'] for vs in video_stats)
    chosen = select(all_candidates, args.keep) if total_kept >= MIN_TOTAL_CANDIDATES else []

    print(build_report(video_stats, all_candidates, chosen))

    if total_kept < MIN_TOTAL_CANDIDATES:
        dropped = {
            'size': sum(vs['dropped_size'] for vs in video_stats),
            'score': sum(vs['dropped_score'] for vs in video_stats),
            'sharpness': sum(vs['dropped_sharpness'] for vs in video_stats),
        }
        reason = max(dropped, key=dropped.get)
        fail(f'only {total_kept} usable face candidates across all videos (need >= '
             f'{MIN_TOTAL_CANDIDATES}); dominant filter: {reason} ({dropped[reason]} dropped; '
             f'size={dropped["size"]} score={dropped["score"]} sharpness={dropped["sharpness"]})')

    if args.dry_run:
        return

    person_dir = ff_session.ROOT / 'faces' / args.person
    archive = write_person(person_dir, chosen)
    suffix = f' (old set archived to {archive.name})' if archive else ''
    print(f'faces/{args.person}: {len(chosen)} photos written{suffix}')


if __name__ == '__main__':
    main()
