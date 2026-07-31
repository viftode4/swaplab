#!/usr/bin/env python3
"""SwapLab — headless face-swap wrapper around FaceFusion.

Usage:
    .venv/bin/python swap.py --video clip.mp4 --face vlad.jpg --out result.mp4
    .venv/bin/python swap.py ... --quality fast|good|best   (default: good)
    .venv/bin/python swap.py ... --audio voice.m4a          (lip-sync to track)
    .venv/bin/python swap.py ... --captions                 (burn auto-subtitles)

--audio without --face lip-syncs the original face. Consented faces and
voices only. Label output as AI-generated when posting.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACEFUSION = ROOT / 'facefusion'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
CONFIG = ROOT / 'facefusion-swaplab.ini'

# face editor control -> facefusion flag suffix (all sliders run -1.0..1.0)
EDIT_CONTROLS = {
    'smile': 'mouth-smile',
    'pout': 'mouth-pout',
    'grim': 'mouth-grim',
    'purse': 'mouth-purse',
    'lips': 'lip-open-ratio',
    'mouth-x': 'mouth-position-horizontal',
    'mouth-y': 'mouth-position-vertical',
    'eyes': 'eye-open-ratio',
    'brows': 'eyebrow-direction',
    'gaze-x': 'eye-gaze-horizontal',
    'gaze-y': 'eye-gaze-vertical',
    'pitch': 'head-pitch',
    'yaw': 'head-yaw',
    'roll': 'head-roll',
}

# named shorthands -> control values (usable from the CLI via --puppet)
PUPPETS = {
    'grin': {'smile': 1.0},
    'shocked': {'eyes': 1.0, 'lips': 0.7, 'brows': 1.0},
    'grumpy': {'grim': 0.7, 'smile': -0.6, 'brows': -1.0},
    'side-eye': {'gaze-x': 1.0},
    'sleepy': {'eyes': -0.7},
    'pout': {'pout': 1.0},
}


def parse_edits(pairs: list[str] | None, puppet: str | None) -> dict[str, float]:
    edits = dict(PUPPETS[puppet]) if puppet else {}
    for pair in pairs or []:
        control, _, raw = pair.partition('=')
        if control not in EDIT_CONTROLS:
            fail(f'unknown edit control {control!r} — pick from {", ".join(EDIT_CONTROLS)}')
        try:
            value = float(raw)
        except ValueError:
            fail(f'edit {control}: {raw!r} is not a number')
        if not -1.0 <= value <= 1.0:
            fail(f'edit {control}: {value} outside -1.0..1.0')
        edits[control] = value
    return edits

# no xseg occlusion masking in any tier: xseg_2 erases the whole swap, and
# xseg_1 reads an open, motion-blurred mouth as an "occluder" and erases the
# swap around it — the decision flips frame to frame, so the face oscillates
# between swapped and original (verified on single frames: same stack with
# occlusion off swaps cleanly). Region masking below covers the hairline and
# most objects held in front of the face, without that failure mode.

# quality tier -> (processors, extra facefusion args)
QUALITY = {
    'fast': (['face_swapper'], []),
    'good': (['face_swapper', 'face_enhancer'],
             ['--face-enhancer-blend', '25']),
    # fidelity stack: high-res swap, restore the original's expressions,
    # enhance at half blend so skin keeps the source footage's texture
    'best': (['face_swapper', 'expression_restorer', 'face_enhancer'], [
        '--output-video-encoder', 'h264_videotoolbox',
        # region masking swaps only parsed face regions, so the hairline
        # and anything above it stay untouched; softer mask edge to blend
        '--face-mask-types', 'box', 'region',
        '--face-mask-blur', '0.4',
        '--face-swapper-pixel-boost', '512x512',
        '--expression-restorer-factor', '90',
        # measured face-region jitter over the source: swap alone +0.52,
        # enhancer at 25 +0.66, at 50 +0.89. The enhancer sharpens a paused
        # frame but reinvents skin detail every frame, which reads as flicker
        '--face-enhancer-blend', '25',
        '--output-video-quality', '95',
    ]),
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


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


def face_photos(path: Path) -> list[Path]:
    """A face is one photo or a directory of photos (averaged identity)."""
    if path.is_dir():
        photos = sorted(p for p in path.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('.'))
        if not photos:
            fail(f'no photos inside face directory {path}')
        return photos
    return [path]


def fail(message: str) -> 'NoReturn':
    print(f'swap.py: {message}', file=sys.stderr)
    sys.exit(1)


# phone screen recordings run 1290x2796; at that size the four-stage best
# pipeline gets OOM-killed, and memory scales with frame area, not frame rate.
# Frame rate is never blind-halved: fps=30 on a 60fps recording of ~30fps
# content lands out of phase with the content cadence (measured: 15 doubled +
# 15 dropped content frames in 9s), which is the judder an earlier attempt hit.
# Duplicate-aware dedupe below keeps every unique content frame instead.
MAX_LONG_SIDE = 1920

# Apple silicon has a dedicated encode engine: several times faster than
# libx264 and it leaves the CPU free for the actual inference work
VIDEO_ENCODE = ['-c:v', 'h264_videotoolbox', '-q:v', '65']


def video_stats(path: Path) -> tuple[int, int, float]:
    ffprobe = shutil.which('ffprobe')
    probe = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,r_frame_rate', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True)
    width, height, rate = probe.stdout.strip().split(',')[:3]
    num, _, den = rate.partition('/')
    fps = float(num) / float(den or 1)
    return int(width), int(height), fps


def content_box(video: Path) -> tuple[int, int, int, int] | None:
    """Find the moving region of a screen recording.

    Phone UI (status bar, buttons, captions) is static; the played video is
    not. Pixels whose brightness varies over time therefore mark the actual
    content, and cropping to it makes the face far bigger for the detector.
    """
    width, height, _ = video_stats(video)
    small_w, small_h = 160, int(160 * height / width) // 2 * 2
    raw = subprocess.run(
        [shutil.which('ffmpeg'), '-v', 'error', '-i', str(video),
         '-vf', f'scale={small_w}:{small_h},fps=4', '-frames:v', '40',
         '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
        capture_output=True).stdout
    import numpy as np
    frames = np.frombuffer(raw, dtype=np.uint8)
    count = len(frames) // (small_w * small_h)
    if count < 8:
        return None
    frames = frames[:count * small_w * small_h].reshape(count, small_h, small_w).astype(float)

    motion = frames.std(axis=0)
    moving = motion > max(3.0, motion.max() * 0.15)
    rows = np.where(moving.any(axis=1))[0]
    cols = np.where(moving.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None

    scale_x, scale_y = width / small_w, height / small_h
    left, right = int(cols[0] * scale_x), int((cols[-1] + 1) * scale_x)
    top, bottom = int(rows[0] * scale_y), int((rows[-1] + 1) * scale_y)
    box_w, box_h = right - left, bottom - top
    if box_w * box_h > 0.92 * width * height:
        return None                      # nothing meaningful to crop away
    if box_w < width * 0.25 or box_h < height * 0.25:
        return None                      # too aggressive to trust
    return (box_w // 2 * 2, box_h // 2 * 2, left // 2 * 2, top // 2 * 2)


def normalize_target(video: Path, work_dir: Path, crop_content: bool = False) -> Path:
    """Crop a screen recording to its content; cap resolution for memory."""
    width, height, fps = video_stats(video)
    filters = []

    if crop_content:
        box = content_box(video)
        if box:
            box_w, box_h, left, top = box
            filters.append(f'crop={box_w}:{box_h}:{left}:{top}')
            print(f'screen recording: cropping to the moving area '
                  f'{box_w}x{box_h} at {left},{top}', flush=True)
            width, height = box_w, box_h
        else:
            print('screen recording: no static border found, keeping the '
                  'full frame', flush=True)

    if max(width, height) > MAX_LONG_SIDE:
        filters.append(f'scale={MAX_LONG_SIDE}:{MAX_LONG_SIDE}'
                       ':force_original_aspect_ratio=decrease:force_divisible_by=2')
    elif crop_content and filters and max(width, height) < MAX_LONG_SIDE:
        # a cropped region is small; upscaling gives the detector more to work
        # with and the swapper a larger face crop to paste back
        filters.append(f'scale={MAX_LONG_SIDE}:{MAX_LONG_SIDE}'
                       ':force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos')

    if not filters:
        return video

    scaled = work_dir / f'.{video.stem}.normalized{video.suffix.lower()}'
    print(f'preparing {width}x{height}@{fps:.0f}', flush=True)
    result = subprocess.run(
        [shutil.which('ffmpeg'), '-y', '-v', 'error', '-i', str(video),
         '-vf', ','.join(filters), *VIDEO_ENCODE, '-c:a', 'copy', str(scaled)],
        capture_output=True, text=True)
    if result.returncode != 0 or not scaled.is_file():
        fail(f'could not normalize the clip:\n{result.stderr.strip()}')
    return scaled


def video_duration(path: Path) -> float:
    probe = subprocess.run(
        [shutil.which('ffprobe'), '-v', 'error', '-show_entries',
         'format=duration', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True)
    return float(probe.stdout.strip())


# a 60fps screen recording of ~30fps content duplicates every frame; the
# swapper regenerates the face per frame, so the two copies come out visibly
# different and the face strobes at 30Hz against a frozen background
# (measured: face region shifted up to 31 gray levels between frozen frames)
DUPLICATE_DIFF = 0.35     # mean gray delta below this = same content frame
DEDUPE_WORTH_IT = 0.35    # dedupe once this share of frames is duplicated


def duplicate_runs(video: Path) -> tuple[list[int], list[int]] | None:
    """Group frames into runs of identical content, streaming, tiny memory.

    Returns (first frame index of each run, run lengths) when enough of the
    clip is duplicated to cause swap strobing, else None.
    """
    import numpy as np
    decode = subprocess.Popen(
        [shutil.which('ffmpeg'), '-v', 'error', '-i', str(video),
         '-fps_mode', 'passthrough', '-vf', 'scale=128:128',
         '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    kept, runs, prev, index = [], [], None, 0
    frame_bytes = 128 * 128
    while True:
        raw = decode.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        if prev is None or np.abs(frame - prev).mean() > DUPLICATE_DIFF:
            kept.append(index)
            runs.append(1)
        else:
            runs[-1] += 1
        prev = frame
        index += 1
    decode.wait()
    if index == 0 or 1 - len(kept) / index < DEDUPE_WORTH_IT:
        return None
    return kept, runs


def raw_frame_pipe(video: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [shutil.which('ffmpeg'), '-v', 'error', '-i', str(video),
         '-fps_mode', 'passthrough', '-f', 'rawvideo',
         '-pix_fmt', 'yuv420p', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def raw_encoder(dest: Path, width: int, height: int, rate: float,
                audio_from: Path, filters: str | None = None) -> subprocess.Popen:
    command = [shutil.which('ffmpeg'), '-y', '-v', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
               '-s', f'{width}x{height}', '-r', f'{rate:.6f}', '-i', '-',
               '-i', str(audio_from), '-map', '0:v', '-map', '1:a?',
               *(['-vf', filters] if filters else []),
               *VIDEO_ENCODE, '-c:a', 'copy', str(dest)]
    return subprocess.Popen(command, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)


def write_unique_frames(video: Path, kept: list[int], total: int) -> Path:
    """Re-encode only the first frame of each duplicate run (CFR)."""
    width, height = video_size(video)
    dest = video.with_name(f'.{video.stem}.unique{video.suffix}')
    rate = len(kept) / video_duration(video)
    decode = raw_frame_pipe(video)
    encode = raw_encoder(dest, width, height, rate, audio_from=video)
    frame_bytes = width * height * 3 // 2
    keep = set(kept)
    try:
        for index in range(total):
            raw = decode.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            if index in keep:
                encode.stdin.write(raw)
        encode.stdin.close()
    except BrokenPipeError:
        fail('the unique-frame encoder died mid-stream')
    decode.wait()
    if encode.wait() != 0 or not dest.is_file():
        fail('could not extract the unique frames')
    return dest


def restore_timing(swapped: Path, runs: list[int], reference: Path,
                   out: Path, denoise: bool) -> None:
    """Copy each swapped frame back over its original duplicate slots.

    The duplicates are bit-identical, so frozen content can no longer
    shimmer, and the timeline matches the source recording exactly.
    """
    width, height = video_size(swapped)
    rate = sum(runs) / video_duration(reference)
    tmp = out.with_name(f'.{out.stem}.timed{out.suffix}')
    decode = raw_frame_pipe(swapped)
    encode = raw_encoder(tmp, width, height, rate, audio_from=reference,
                         filters='hqdn3d=2:1:20:20' if denoise else None)
    frame_bytes = width * height * 3 // 2
    copied = 0
    try:
        for length in runs:
            raw = decode.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            for _ in range(length):
                encode.stdin.write(raw)
            copied += 1
        encode.stdin.close()
    except BrokenPipeError:
        fail('the timing encoder died mid-stream')
    decode.wait()
    encode_ok = encode.wait() == 0 and tmp.is_file()
    if copied != len(runs):
        tmp.unlink(missing_ok=True)
        fail(f'swapped clip holds {copied} frames, expected {len(runs)} — '
             'the swap dropped frames')
    if not encode_ok:
        fail('could not rebuild the original frame timing')
    tmp.replace(out)
    if denoise:
        print('stabilized (temporal denoise)', flush=True)
    print(f'restored timing: {copied} unique frames back over '
          f'{sum(runs)}', flush=True)


def video_size(path: Path) -> tuple[int, int]:
    ffprobe = shutil.which('ffprobe')
    probe = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True)
    width, height = probe.stdout.strip().split(',')[:2]
    return int(width), int(height)


CAPTION_FONTS = [
    '/System/Library/Fonts/Supplemental/Arial Black.ttf',
    '/System/Library/Fonts/Supplemental/Impact.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
]


def caption_card(text: str, width: int, height: int, dest: Path) -> None:
    """Rasterize one caption as a transparent full-frame PNG (Pillow)."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = next((f for f in CAPTION_FONTS if Path(f).is_file()), None)
    size = max(24, height // 14)
    font = (ImageFont.truetype(font_path, size) if font_path
            else ImageFont.load_default(size))
    card = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)
    stroke = max(2, size // 10)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    x = (width - (box[2] - box[0])) / 2 - box[0]
    y = height * 0.78 - (box[3] - box[1]) / 2 - box[1]
    draw.text((x, y), text, font=font, fill='white',
              stroke_width=stroke, stroke_fill='black')
    card.save(dest)


def burn_captions(out: Path) -> None:
    """Transcribe speech locally and burn big word-group subtitles in."""
    from faster_whisper import WhisperModel

    print('captions: transcribing (local whisper)…', flush=True)
    model = WhisperModel('small', device='cpu', compute_type='int8')
    segments, _ = model.transcribe(str(out), word_timestamps=True)
    words = [word for segment in segments for word in segment.words or []]
    if not words:
        print('captions: no speech found, skipping')
        return

    # this ffmpeg build has no libass/drawtext — rasterize each caption as a
    # transparent PNG and stack time-gated overlay filters instead
    width, height = video_size(out)
    cards = []  # (png path, start, end)
    for i in range(0, len(words), 3):
        group = words[i:i + 3]
        text = ' '.join(w.word.strip() for w in group).upper().replace('\n', ' ')
        png = out.with_name(f'.{out.stem}.cap{i}.png')
        caption_card(text, width, height, png)
        cards.append((png, group[0].start, group[-1].end))

    ffmpeg = shutil.which('ffmpeg')
    tmp = out.with_name(f'.{out.stem}.captioned{out.suffix}')
    inputs, chain, current = [], [], '0:v'
    for index, (png, start, end) in enumerate(cards):
        inputs += ['-i', str(png)]
        label = f'v{index}'
        chain.append(f"[{current}][{index + 1}:v]overlay="
                     f"enable='between(t,{start:.3f},{end:.3f})'[{label}]")
        current = label
    burn = subprocess.run(
        [ffmpeg, '-y', '-v', 'error', '-i', str(out), *inputs,
         '-filter_complex', ';'.join(chain), '-map', f'[{current}]',
         '-map', '0:a?', *VIDEO_ENCODE, '-c:a', 'copy', str(tmp)],
        capture_output=True, text=True)
    for png, _, _ in cards:
        png.unlink(missing_ok=True)
    if burn.returncode != 0 or not tmp.is_file():
        fail(f'caption burn failed:\n{burn.stderr.strip()}')
    tmp.replace(out)
    print(f'captions: burned {len(cards)} lines')


# facefusion's memory use grows as it works — frame times on a long clip drift
# from 2s to 10s and macOS eventually kills it. Each chunk is a fresh process,
# so whatever leaked is reclaimed between them.
CHUNK_FRAMES = 150


def video_frames(path: Path) -> int:
    probe = subprocess.run(
        [shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
         '-count_packets', '-show_entries', 'stream=nb_read_packets',
         '-of', 'csv=p=0', str(path)], capture_output=True, text=True)
    try:
        return int(probe.stdout.strip())
    except ValueError:
        return 0


def run_in_chunks(command: list[str], work_out: Path, frame_count: int) -> None:
    output_at = command.index('--output-path') + 1
    total = -(-frame_count // CHUNK_FRAMES)
    chunks = []
    for index, start in enumerate(range(0, frame_count, CHUNK_FRAMES)):
        end = min(start + CHUNK_FRAMES, frame_count)
        piece = work_out.with_name(f'.{work_out.stem}.part{index}{work_out.suffix}')
        print(f'chunk {index + 1}/{total} (frames {start}-{end})', flush=True)
        chunk_command = list(command)
        chunk_command[output_at] = str(piece)
        chunk_command += ['--trim-frame-start', str(start),
                          '--trim-frame-end', str(end)]
        result = subprocess.run(chunk_command, cwd=FACEFUSION)
        if result.returncode != 0 or not piece.is_file():
            for done in chunks:
                done.unlink(missing_ok=True)
            fail(f'facefusion exited with {result.returncode} on chunk '
                 f'{index + 1} of {total}')
        chunks.append(piece)

    listing = work_out.with_name(f'.{work_out.stem}.chunks.txt')
    listing.write_text(''.join(f"file '{p.name}'\n" for p in chunks))
    concat = subprocess.run(
        [shutil.which('ffmpeg'), '-y', '-v', 'error', '-f', 'concat',
         '-safe', '0', '-i', str(listing), '-c', 'copy', str(work_out)],
        capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    for piece in chunks:
        piece.unlink(missing_ok=True)
    if concat.returncode != 0:
        fail(f'could not join the rendered chunks:\n{concat.stderr.strip()}')


def stabilize(out: Path) -> None:
    """Damp the frame-to-frame shimmer a per-frame swapper leaves behind.

    Each output frame is generated independently, so skin detail differs
    slightly every frame and reads as flicker. hqdn3d's temporal term
    averages a pixel with its own past only while the picture there is
    steady, so moving edges stay sharp. Measured on a real render: face
    jitter 8.74 -> 6.92, spatial detail 4.11 -> 3.76.
    """
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return
    tmp = out.with_name(f'.{out.stem}.stable{out.suffix}')
    result = subprocess.run(
        [ffmpeg, '-y', '-v', 'error', '-i', str(out),
         '-vf', 'hqdn3d=2:1:20:20', *VIDEO_ENCODE, '-c:a', 'copy', str(tmp)],
        capture_output=True, text=True)
    if result.returncode == 0 and tmp.is_file():
        tmp.replace(out)
        print('stabilized (temporal denoise)', flush=True)
    else:
        tmp.unlink(missing_ok=True)


def check_output_video(path: Path) -> None:
    """Verify the result is a playable video; fail loudly otherwise."""
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return
    probe = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name,duration', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        fail(f'output {path} is not a playable video:\n{probe.stderr.strip()}')


def convert_image(src: Path, dest: Path) -> None:
    """Image-to-image format conversion — the video remux path would
    re-encode a picture with h264 and mangle it."""
    from PIL import Image
    try:
        Image.open(src).convert('RGB').save(dest, quality=95)
    except OSError as error:
        fail(f'could not convert {src.name} to {dest.suffix}: {error}')


def check_output_image(path: Path, target: Path) -> None:
    """Verify the result is a readable image at the target's resolution.

    Facefusion rounds image output dimensions to even numbers
    (vision.normalize_resolution), so an odd-sized target legitimately
    comes back 1px short on that axis — tolerate exactly that."""
    from PIL import Image
    try:
        with Image.open(path) as result, Image.open(target) as original:
            if any(abs(r - o) > 1 for r, o in zip(result.size, original.size)):
                fail(f'output is {result.size[0]}x{result.size[1]}, '
                     f'expected {original.size[0]}x{original.size[1]}')
    except OSError as error:
        fail(f'output {path} is not a readable image: {error}')


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
        # measured on this Mac: 62s at 1 thread, 57s at 4, 56s at 8 for the
        # same 60 frames — 4 takes nearly all of the win at less memory
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--video', required=True, help='target video (the clip)')
    parser.add_argument('--face', help='source face photo, or a directory of '
                        'photos of the same person (averaged, stronger identity)')
    parser.add_argument('--swapper-model', help='override face swapper model '
                        '(e.g. hyperswap_1a_256, inswapper_128_fp16, ghost_3_256)')
    parser.add_argument('--enhancer-model', help='override face enhancer model '
                        '(e.g. gfpgan_1.4, codeformer)')
    parser.add_argument('--audio', help='voice/music track to lip-sync the face to')
    parser.add_argument('--captions', action='store_true',
                        help='transcribe speech locally and burn subtitles in')
    parser.add_argument('--screen-recording', action='store_true',
                        help='crop away the static phone UI and zoom the content')
    parser.add_argument('--no-stabilize', action='store_true',
                        help='skip the anti-flicker pass (keeps maximum detail)')
    parser.add_argument('--edit', action='append', metavar='CONTROL=VALUE',
                        help=f'face editor slider, -1.0..1.0 (repeatable): {", ".join(EDIT_CONTROLS)}')
    parser.add_argument('--puppet', choices=PUPPETS,
                        help='named shorthand that pre-fills --edit values')
    parser.add_argument('--list-faces', action='store_true',
                        help='detect faces in the target image, print JSON boxes, and exit')
    parser.add_argument('--all-faces', action='store_true',
                        help='swap every face in a photo to the --face person')
    parser.add_argument('--map', action='append', metavar='N=PERSON',
                        help='photo face #N (left to right, from --list-faces) '
                             'becomes this person — photo or directory (repeatable)')
    parser.add_argument('--out', required=False, help='output video path')
    parser.add_argument('--quality', choices=QUALITY, default='good')
    parser.add_argument('--cpu', action='store_true', help='force CPU (skip CoreML)')
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    face = Path(args.face).expanduser().resolve() if args.face else None
    audio = Path(args.audio).expanduser().resolve() if args.audio else None

    if not video.is_file():
        fail(f'video not found: {video}')
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
    out = Path(args.out).expanduser().resolve()
    edits = parse_edits(args.edit, args.puppet)
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
    if face is None and audio is None and not edits and not mappings:
        fail('nothing to do — pass --face, --audio and/or --edit')
    if face is not None and not (face.is_file() or face.is_dir()):
        fail(f'face photo not found: {face}')
    if audio is not None and not audio.is_file():
        fail(f'audio track not found: {audio}')
    if not PYTHON.is_file():
        fail('venv missing — run the setup in README.md first')
    out.parent.mkdir(parents=True, exist_ok=True)

    reference, runs = None, None
    if not is_image(video):
        video = normalize_target(video, out.parent, crop_content=args.screen_recording)
        reference = video
        duplicated = duplicate_runs(video)
        if duplicated:
            kept, runs = duplicated
            total = sum(runs)
            print(f'dedupe: {len(kept)} unique content frames of {total} '
                  f'({1 - len(kept) / total:.0%} duplicated) — swapping '
                  'uniques only', flush=True)
            video = write_unique_frames(video, kept, total)

    # facefusion insists the output extension match the target's; swap into a
    # sibling temp file with the right extension, remux to the asked-for name
    work_out = out
    if out.suffix.lower() != video.suffix.lower():
        work_out = out.with_name(f'.{out.stem}.work{video.suffix.lower()}')

    photo = is_image(video)
    if photo:
        if args.quality != 'good':          # 'good' is just the default
            print('photo target: --quality is ignored, photos always run '
                  'the max stack', flush=True)
        processors, extra = PHOTO_STACK
    else:
        processors, extra = QUALITY[args.quality]
    sources = []
    if face is None and not mappings:
        processors, extra = [], []          # lip-sync only, no swap
    elif face is not None:
        sources.extend(str(p) for p in face_photos(face))
    if args.swapper_model:
        extra = [*extra, '--face-swapper-model', args.swapper_model]
    if args.enhancer_model:
        extra = [*extra, '--face-enhancer-model', args.enhancer_model]
    if edits:
        # edit before the syncer (sync owns the mouth) and the enhancer
        slot = processors.index('face_enhancer') if 'face_enhancer' in processors else len(processors)
        processors = [*processors[:slot], 'face_editor', *processors[slot:]]
        for control, value in edits.items():
            extra = [*extra, f'--face-editor-{EDIT_CONTROLS[control]}', str(value)]
    if audio is not None:
        # sync before the enhancer so the generated mouth gets polished too
        slot = processors.index('face_enhancer') if 'face_enhancer' in processors else len(processors)
        processors = [*processors[:slot], 'lip_syncer', *processors[slot:]]
        sources.append(str(audio))

    # 'one' swaps the most prominent face every frame; 'reference' mode
    # dropped frames whenever the actor turned away from the reference pose
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
                       [*selector, *extra], args.cpu)

    if runs:
        restore_timing(work_out, runs, reference, out,
                       denoise=not args.no_stabilize)
        if work_out != out:
            work_out.unlink(missing_ok=True)
    else:
        if work_out != out:
            if photo:
                convert_image(work_out, out)
                work_out.unlink(missing_ok=True)
            else:
                ffmpeg = shutil.which('ffmpeg') or fail('ffmpeg is required to remux output')
                remux = subprocess.run(
                    [ffmpeg, '-y', '-v', 'error', '-i', str(work_out), '-c', 'copy', str(out)],
                    capture_output=True, text=True)
                if remux.returncode != 0:  # container mismatch — re-encode instead
                    remux = subprocess.run(
                        [ffmpeg, '-y', '-v', 'error', '-i', str(work_out),
                         *VIDEO_ENCODE, '-c:a', 'aac', str(out)],
                        capture_output=True, text=True)
                work_out.unlink(missing_ok=True)
                if remux.returncode != 0 or not out.is_file():
                    fail(f'could not convert output to {out.suffix}:\n{remux.stderr.strip()}')
        if not args.no_stabilize and not photo:
            stabilize(out)

    if args.captions:
        burn_captions(out)
    if photo:
        check_output_image(out, video)
    else:
        check_output_video(out)
    if reference is not None and reference.is_file():
        from check import analyze, report
        report(analyze(reference, out))
    print(f'done: {out}')


if __name__ == '__main__':
    main()
