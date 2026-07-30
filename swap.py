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
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACEFUSION = ROOT / 'facefusion'
PYTHON = ROOT / '.venv' / 'bin' / 'python'

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

# occlusion masking keeps hair/hands/props in front of the face on top of
# the swap instead of being painted over (xseg segmenter, per-frame cost)
# xseg_1 specifically: the 'many' ensemble includes xseg_2, which produces an
# empty mask here and silently erases the whole swap (verified frame by frame)
OCCLUSION = ['--face-mask-types', 'box', 'occlusion',
             '--face-occluder-model', 'xseg_1']

# quality tier -> (processors, extra facefusion args)
QUALITY = {
    'fast': (['face_swapper'], []),
    'good': (['face_swapper', 'face_enhancer'],
             [*OCCLUSION, '--face-enhancer-blend', '25']),
    # fidelity stack: high-res swap, restore the original's expressions,
    # enhance at half blend so skin keeps the source footage's texture
    'best': (['face_swapper', 'expression_restorer', 'face_enhancer'], [
        '--output-video-encoder', 'h264_videotoolbox',
        # region masking swaps only parsed face regions, so the hairline
        # and anything above it stay untouched; softer mask edge to blend
        '--face-mask-types', 'box', 'occlusion', 'region',
        '--face-occluder-model', 'xseg_1',
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
# Frame rate is left alone: halving it to 30 made motion judder, which reads
# as flicker and cost far more than the enhancer ever did.
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


def normalize_target(video: Path, work_dir: Path) -> Path:
    """Cap resolution so big phone clips fit in memory; keep the frame rate."""
    width, height, fps = video_stats(video)
    if max(width, height) <= MAX_LONG_SIDE:
        return video

    scaled = work_dir / f'.{video.stem}.normalized{video.suffix.lower()}'
    filters = [f'scale={MAX_LONG_SIDE}:{MAX_LONG_SIDE}'
               ':force_original_aspect_ratio=decrease:force_divisible_by=2']
    print(f'normalizing {width}x{height}@{fps:.0f} -> '
          f'{"/".join(filters)} (keeps the render inside memory)', flush=True)
    result = subprocess.run(
        [shutil.which('ffmpeg'), '-y', '-v', 'error', '-i', str(video),
         '-vf', ','.join(filters), *VIDEO_ENCODE, '-c:a', 'copy', str(scaled)],
        capture_output=True, text=True)
    if result.returncode != 0 or not scaled.is_file():
        fail(f'could not normalize the clip:\n{result.stderr.strip()}')
    return scaled


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
    parser.add_argument('--edit', action='append', metavar='CONTROL=VALUE',
                        help=f'face editor slider, -1.0..1.0 (repeatable): {", ".join(EDIT_CONTROLS)}')
    parser.add_argument('--puppet', choices=PUPPETS,
                        help='named shorthand that pre-fills --edit values')
    parser.add_argument('--out', required=True, help='output video path')
    parser.add_argument('--quality', choices=QUALITY, default='good')
    parser.add_argument('--cpu', action='store_true', help='force CPU (skip CoreML)')
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    face = Path(args.face).expanduser().resolve() if args.face else None
    audio = Path(args.audio).expanduser().resolve() if args.audio else None
    out = Path(args.out).expanduser().resolve()

    if not video.is_file():
        fail(f'video not found: {video}')
    edits = parse_edits(args.edit, args.puppet)
    if face is None and audio is None and not edits:
        fail('nothing to do — pass --face, --audio and/or --edit')
    if face is not None and not (face.is_file() or face.is_dir()):
        fail(f'face photo not found: {face}')
    if audio is not None and not audio.is_file():
        fail(f'audio track not found: {audio}')
    if not PYTHON.is_file():
        fail('venv missing — run the setup in README.md first')
    out.parent.mkdir(parents=True, exist_ok=True)

    if video.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
        video = normalize_target(video, out.parent)

    # facefusion insists the output extension match the target's; swap into a
    # sibling temp file with the right extension, remux to the asked-for name
    work_out = out
    if out.suffix.lower() != video.suffix.lower():
        work_out = out.with_name(f'.{out.stem}.work{video.suffix.lower()}')

    processors, extra = QUALITY[args.quality]
    sources = []
    if face is None:
        processors, extra = [], []          # lip-sync only, no swap
    else:
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

    command = [
        str(PYTHON), 'facefusion.py', 'headless-run',
        *(['--source-paths', *sources] if sources else []),
        '--target-path', str(video),
        '--output-path', str(work_out),
        '--processors', *processors,
        '--execution-providers', 'cpu' if args.cpu else 'coreml',
        '--video-memory-strategy', 'moderate',
        # measured on this Mac: 62s at 1 thread, 57s at 4, 56s at 8 for the
        # same 60 frames — 4 takes nearly all of the win at less memory
        '--execution-thread-count', '4',
        # 'one' swaps the most prominent face every frame; 'reference' mode
        # dropped frames whenever the actor turned away from the reference pose
        '--face-selector-mode', 'one',
        *extra,
    ]
    result = subprocess.run(command, cwd=FACEFUSION)
    if result.returncode != 0:
        fail(f'facefusion exited with {result.returncode} '
             '(no face in photo/video and codec issues are the usual causes)')
    if not work_out.is_file():
        fail('facefusion reported success but produced no output file')

    if work_out != out:
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

    if args.captions:
        burn_captions(out)
    check_output_video(out)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
