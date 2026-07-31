#!/usr/bin/env python3
"""Print the faces facefusion detects in an image, as JSON, left to right.

Run from the repo root with its venv:
    .venv/bin/python scan_faces.py photo.jpg facefusion-swaplab.ini

Uses facefusion's own detector (pinned commit) so the numbering agrees
exactly with what --reference-face-position selects during a swap.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_session


def main() -> None:
    target = str(Path(sys.argv[1]).resolve())
    config = str(Path(sys.argv[2]).resolve())
    ff_session.boot(config, ['--target-path', target])

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
