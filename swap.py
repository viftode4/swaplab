#!/usr/bin/env python3
"""SwapLab — headless face-swap wrapper around FaceFusion.

Usage:
    .venv/bin/python swap.py --video clip.mp4 --face vlad.jpg --out result.mp4
    .venv/bin/python swap.py ... --quality fast|good|best   (default: good)

Consented faces only. Label output as AI-generated when posting.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACEFUSION = ROOT / 'facefusion'
PYTHON = ROOT / '.venv' / 'bin' / 'python'

# quality tier -> (processors, extra facefusion args)
QUALITY = {
    'fast': (['face_swapper'], []),
    'good': (['face_swapper', 'face_enhancer'], []),
    'best': (['face_swapper', 'face_enhancer'], [
        '--output-video-quality', '95',
        '--output-video-preset', 'slower',
    ]),
}


def fail(message: str) -> 'NoReturn':
    print(f'swap.py: {message}', file=sys.stderr)
    sys.exit(1)


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
    parser.add_argument('--face', required=True, help='source face photo (a consented friend)')
    parser.add_argument('--out', required=True, help='output video path')
    parser.add_argument('--quality', choices=QUALITY, default='good')
    parser.add_argument('--cpu', action='store_true', help='force CPU (skip CoreML)')
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    face = Path(args.face).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not video.is_file():
        fail(f'video not found: {video}')
    if not face.is_file():
        fail(f'face photo not found: {face}')
    if not PYTHON.is_file():
        fail('venv missing — run the setup in README.md first')
    out.parent.mkdir(parents=True, exist_ok=True)

    processors, extra = QUALITY[args.quality]
    command = [
        str(PYTHON), 'facefusion.py', 'headless-run',
        '--source-paths', str(face),
        '--target-path', str(video),
        '--output-path', str(out),
        '--processors', *processors,
        '--execution-providers', 'cpu' if args.cpu else 'coreml',
        # keep the swap on one person: match the reference face across frames
        '--face-selector-mode', 'reference',
        *extra,
    ]
    result = subprocess.run(command, cwd=FACEFUSION)
    if result.returncode != 0:
        fail(f'facefusion exited with {result.returncode} '
             '(no face in photo/video and codec issues are the usual causes)')
    if not out.is_file():
        fail('facefusion reported success but produced no output file')
    check_output_video(out)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
