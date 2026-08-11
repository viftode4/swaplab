#!/usr/bin/env python3
"""Score how much a rendered face looks like a person's real photos.

    score_face.py <person-dir> <candidate.png> [candidate2.png ...]

Prints one "<path>\t<cosine>" line per candidate, best first. Cosine is
between the candidate's face embedding and the mean embedding of the
person's reference photos, so it is comparable across candidates of the
same person but not between different people.

Runs under comfyui/.venv (insightface + antelopev2 live there).
"""

import sys
from pathlib import Path

import cv2
import numpy
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parent
INSIGHTFACE_DIR = ROOT / 'comfyui' / 'models' / 'insightface'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def largest_face(app: FaceAnalysis, path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    # SCRFD needs to see the whole head with room around it; a tight face
    # crop (what the refine stage renders) detects as no face at all, which
    # silently scored every candidate -1 and made best-of-N a coin flip
    border = max(image.shape[:2]) // 3
    image = cv2.copyMakeBorder(image, border, border, border, border,
                               cv2.BORDER_REPLICATE)
    faces = app.get(image)
    if not faces:
        return None
    biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return biggest.normed_embedding


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    person = Path(sys.argv[1])
    candidates = [Path(p) for p in sys.argv[2:]]

    app = FaceAnalysis(name='antelopev2', root=str(INSIGHTFACE_DIR),
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    references = [person] if person.is_file() else sorted(
        p for p in person.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('.'))
    embeddings = [e for p in references if (e := largest_face(app, p)) is not None]
    if not embeddings:
        print('score_face.py: no face found in the reference photos', file=sys.stderr)
        sys.exit(1)
    target = numpy.mean(embeddings, axis=0)
    target /= numpy.linalg.norm(target)

    scored = []
    for candidate in candidates:
        embedding = largest_face(app, candidate)
        # a render with no detectable face is a failed render, not a tie
        scored.append((float(numpy.dot(target, embedding)) if embedding is not None
                       else -1.0, candidate))
    for score, candidate in sorted(scored, reverse=True):
        print(f'{candidate}\t{score:.4f}')


if __name__ == '__main__':
    main()
