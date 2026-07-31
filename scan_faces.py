#!/usr/bin/env python3
"""Print the faces facefusion detects in an image, as JSON, left to right.

Run from inside the facefusion checkout with its venv:
    cd facefusion && ../.venv/bin/python ../scan_faces.py photo.jpg ../facefusion-swaplab.ini

Uses facefusion's own detector (pinned commit) so the numbering agrees
exactly with what --reference-face-position selects during a swap.
"""

import json
import os
import sys


def main() -> None:
    target, config = sys.argv[1], sys.argv[2]
    # this script lives at the repo root, one level above the facefusion
    # checkout, so its own directory lands first on sys.path — and that
    # directory contains a *sibling* folder also named "facefusion" (the
    # checkout), which shadows the real package one level deeper inside it.
    # Prepend cwd (the facefusion checkout, per the docstring's usage) so
    # the real package resolves first, exactly as running facefusion.py
    # directly already does.
    sys.path.insert(0, os.getcwd())
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
