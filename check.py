#!/usr/bin/env python3
"""SwapLab render QA — measure the flicker a viewer would actually see.

Usage:
    .venv/bin/python check.py --target clip.mp4 --result swapped.mp4

Compares the swap target against the finished render:
  * duplicate cadence — share of target frames that are frozen copies of the
    previous one (60fps screen recordings of 30fps content run ~50%)
  * shimmer — how much the result changes across those frozen pairs, measured
    in the busiest 96x96 block (the face); the eye reads any change against a
    frozen background as flicker
  * dropouts — frames where the swapped face collapses back to the target's
    (detection missed, the original face shows for an instant)

swap.py runs this automatically after every render.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROBE_W, PROBE_H = 180, 344
STATIC_DIFF = 0.35      # mean gray delta below this = duplicated frame
SCENE_DIFF = 20.0       # above this = scene cut, not shimmer
SHIMMER_WARN = 3.0      # face-block p90 above this is visible flicker
DURATION_WARN = 0.1     # seconds; catches chunk-boundary A/V padding


def gray_frames(path: Path) -> np.ndarray:
    """Decode every frame (no CFR resampling) as small grayscale."""
    raw = subprocess.run(
        [shutil.which('ffmpeg'), '-v', 'error', '-i', str(path),
         '-fps_mode', 'passthrough', '-vf', f'scale={PROBE_W}:{PROBE_H}',
         '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
        capture_output=True).stdout
    count = len(raw) // (PROBE_W * PROBE_H)
    return (np.frombuffer(raw[:count * PROBE_W * PROBE_H], dtype=np.uint8)
            .reshape(count, PROBE_H, PROBE_W).astype(np.int16))


def duration(path: Path) -> float:
    probe = subprocess.run(
        [shutil.which('ffprobe'), '-v', 'error', '-show_entries',
         'format=duration', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True)
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return 0.0


def face_block_diff(delta: np.ndarray, block: int = 96) -> float:
    """Mean of |delta| in the busiest block x block window (the face)."""
    windows = (np.lib.stride_tricks
               .sliding_window_view(delta, (block, block))[::8, ::8]
               .mean(axis=(2, 3)))
    return float(windows.max())


def analyze(target: Path, result: Path) -> dict:
    inp, out = gray_frames(target), gray_frames(result)
    count = min(len(inp), len(out))
    input_diff = np.abs(np.diff(inp[:count], axis=0)).mean(axis=(1, 2))
    frozen = input_diff < STATIC_DIFF

    shimmer = []
    for i in np.where(frozen)[0]:
        delta = np.abs(out[i + 1] - out[i]).astype(float)
        if delta.mean() < SCENE_DIFF:
            shimmer.append(face_block_diff(delta))
    shimmer = np.array(shimmer)

    # how strongly each frame is swapped: the busiest block of |result-target|
    # is the swapped face; a collapse against neighbouring frames means the
    # detector missed and the original face passed through (measured on a real
    # render: swapped frames sit at 20-35, missed frames at 3-5)
    swap_delta = np.array([
        face_block_diff(np.abs(out[i] - inp[i]).astype(float))
        for i in range(count)])
    # scene cuts (story transitions) legitimately have no face for a moment
    cuts = np.where(input_diff > SCENE_DIFF)[0]
    near_cut = np.zeros(count, dtype=bool)
    for c in cuts:
        near_cut[max(0, c - 3):c + 5] = True
    candidates = []
    for i in range(count):
        window = np.median(swap_delta[max(0, i - 15):i + 15])
        if window > 8.0 and swap_delta[i] < 0.4 * window and not near_cut[i]:
            candidates.append(i)

    # A face moving behind a foreground person makes the swapped-pixel area
    # taper down and back up over several frames. The old neighbourhood-only
    # test called that a dropout even though tracked face boxes were still
    # changed. A real detector miss has a sharp entry and recovery: keep only
    # candidate runs whose low point collapses against both immediate sides.
    dropouts = []
    position = 0
    while position < len(candidates):
        start = candidates[position]
        end = start
        while (position + 1 < len(candidates)
               and candidates[position + 1] == end + 1):
            position += 1
            end += 1
        low = np.median(swap_delta[start:end + 1])
        before = np.median(swap_delta[max(0, start - 3):start])
        after = np.median(swap_delta[end + 1:min(count, end + 4)])
        if before and after and low < 0.45 * before and low < 0.45 * after:
            dropouts.extend(range(start, end + 1))
        position += 1

    return {
        'target_frames': len(inp),
        'result_frames': len(out),
        'target_duration': duration(target),
        'result_duration': duration(result),
        'duplicate_ratio': float(frozen.mean()) if count > 1 else 0.0,
        'shimmer_pairs': len(shimmer),
        'shimmer_p50': float(np.median(shimmer)) if len(shimmer) else 0.0,
        'shimmer_p90': float(np.percentile(shimmer, 90)) if len(shimmer) else 0.0,
        'shimmer_max': float(shimmer.max()) if len(shimmer) else 0.0,
        'dropout_frames': dropouts,
    }


def report(stats: dict) -> bool:
    """Print the QA lines; return True when the render looks clean."""
    clean = True
    print(f"check: target {stats['target_frames']} frames, "
          f"result {stats['result_frames']} frames, "
          f"{stats['duplicate_ratio']:.0%} duplicated in target")
    if stats['target_frames'] != stats['result_frames']:
        print('check: WARN frame count changed — timing will drift')
        clean = False
    duration_delta = abs(stats['target_duration'] - stats['result_duration'])
    print(f"check: target {stats['target_duration']:.3f}s, "
          f"result {stats['result_duration']:.3f}s "
          f"(delta {duration_delta:.3f}s)")
    if duration_delta > DURATION_WARN:
        print('check: WARN duration changed — audio/video timing will drift')
        clean = False
    if stats['shimmer_pairs']:
        print(f"check: flicker score (face change on frozen frames) "
              f"p50={stats['shimmer_p50']:.2f} p90={stats['shimmer_p90']:.2f} "
              f"max={stats['shimmer_max']:.2f} over {stats['shimmer_pairs']} pairs")
        if stats['shimmer_p90'] > SHIMMER_WARN:
            print(f'check: WARN visible flicker — face shifts '
                  f'{stats["shimmer_p90"]:.1f} gray levels on frozen frames '
                  f'(threshold {SHIMMER_WARN})')
            clean = False
    else:
        print('check: no duplicated frames in target — flicker score n/a')
    # the clip's lead-in/out often has no face; only mid-clip collapses matter
    interior = [f for f in stats['dropout_frames']
                if 30 < f < stats['result_frames'] - 30]
    if interior:
        print(f'check: WARN swap dropped out on {len(interior)} frames '
              f'(first at {interior[0]})')
        clean = False
    if clean:
        print('check: clean')
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--target', required=True, help='clip the swap consumed')
    parser.add_argument('--result', required=True, help='finished render')
    args = parser.parse_args()
    target, result = Path(args.target), Path(args.result)
    for path in (target, result):
        if not path.is_file():
            print(f'check: {path} not found', file=sys.stderr)
            sys.exit(1)
    report(analyze(target, result))


if __name__ == '__main__':
    main()
