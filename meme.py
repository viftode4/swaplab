#!/usr/bin/env python3
"""Identity refinement on top of the photoreal swap (local SDXL + InstantID).

swap.py's swappers are photoreal and leave everything else in the photo
untouched, but their likeness is limited by a 128-256px embedding. Redrawing
the whole head from noise fixes likeness and ruins everything else: hats and
hair get reinvented and the result reads as an illustration.

So this runs in two stages: swap.py first for a photoreal base, then a light
diffusion pass over the face only — low denoise, edge-locked, per-person LoRA
when one is trained — which pushes the identity without repainting the scene.

    meme.py --photo party.jpg --face faces/teo --out teo.jpg
    meme.py --photo party.jpg --face faces/vlad --index 1 --likeness 80 --out v.jpg
    meme.py --photo party.jpg --face faces/teo --whole-head --out big.jpg

--whole-head opts back into the redraw-everything behaviour, for when the
head's shape and proportions have to change and losing the hat is fine.

Runs everything locally: ComfyUI (started on demand, kept warm) on MPS,
insightface on CPU. First run after boot pays ~1 min of model loading.
"""

import argparse
import io
import json
import mimetypes
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
COMFY = ROOT / 'comfyui'
COMFY_PYTHON = COMFY / '.venv' / 'bin' / 'python'
COMFY_URL = 'http://127.0.0.1:8188'
COMFY_LOG = COMFY / '.server.log'
CONFIG = ROOT / 'facefusion-swaplab.ini'
FACEFUSION = ROOT / 'facefusion'
PYTHON = ROOT / '.venv' / 'bin' / 'python'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
CHECKPOINT = 'RealVisXL_V4.0.safetensors'
CONTROLNET = 'instantid-controlnet.safetensors'
CANNY_CONTROLNET = 'canny-sdxl.safetensors'
INSTANTID = 'ip-adapter.bin'
# refine mode: the mask hugs the face itself, so hats, hair and background
# are never inside the region the sampler is allowed to touch
# tight crop: the sampler renders 1024px whatever the crop covers, so every
# unit of margin is resolution taken off the face itself
CROP_SCALE = 1.32
MASK_W, MASK_H = 1.05, 1.2
MASK_UP = 0.06
FEATHER = 20
DENOISE = 0.34              # enough to move identity, too little to restyle
CANNY_STRENGTH = 0.75       # hold the swapped face's own structure
CFG = 3.5                   # high cfg is what makes SDXL look illustrated

# whole-head mode: wide box so the sampler can change head shape and hairline
WHOLE_HEAD = {
    'crop_scale': 2.4, 'mask_w': 1.7, 'mask_h': 2.1, 'mask_up': 0.35,
    'feather': 31, 'denoise': 0.85, 'canny': 0.45, 'cfg': 5.0,
}


def fail(message: str) -> 'NoReturn':
    print(f'meme.py: {message}', file=sys.stderr)
    sys.exit(1)


def api(path: str, payload: dict | None = None, timeout: int = 30):
    request = urllib.request.Request(
        COMFY_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'} if payload is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # the body carries why the graph was rejected; the status line alone
        # ("400: Bad Request") sends you looking in the wrong place
        detail = error.read().decode(errors='replace')[:500]
        fail(f'comfyui rejected {path}: {detail}')


def upload(name: str, data: bytes) -> None:
    """Multipart upload into ComfyUI's input folder (stdlib only)."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(name)[0] or 'image/png'
    body = (f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f'Content-Type: {content_type}\r\n\r\n').encode() \
        + data + f'\r\n--{boundary}\r\n' \
                 f'Content-Disposition: form-data; name="overwrite"\r\n\r\n' \
                 f'true\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(
        f'{COMFY_URL}/upload/image', data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def ensure_server() -> None:
    try:
        api('/system_stats')
        return
    except (urllib.error.URLError, OSError):
        pass
    if not COMFY_PYTHON.is_file():
        fail('comfyui venv missing — run the meme-mode setup first')
    print('starting comfyui…', flush=True)
    with COMFY_LOG.open('w') as log:
        subprocess.Popen(
            [str(COMFY_PYTHON), 'main.py', '--listen', '127.0.0.1', '--port', '8188'],
            cwd=COMFY, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        time.sleep(2)
        try:
            api('/system_stats')
            return
        except (urllib.error.URLError, OSError):
            continue
    fail(f'comfyui did not come up — see {COMFY_LOG}')


def scan_faces(photo: Path) -> list[dict]:
    result = subprocess.run(
        [str(PYTHON), str(ROOT / 'scan_faces.py'), str(photo.resolve()), str(CONFIG)],
        cwd=FACEFUSION, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f'face scan failed: {(result.stdout or result.stderr).strip()[-300:]}')
    return json.loads(result.stdout.splitlines()[-1]).get('faces', [])


def reference_photos(face_dir: Path) -> list[Path]:
    """Every photo of the person — their embeddings are averaged."""
    if face_dir.is_file():
        return [face_dir]
    photos = sorted(p for p in face_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('.'))
    if not photos:
        fail(f'no photos inside face directory {face_dir}')
    return photos


def head_mask(size: int, box: tuple[float, float, float, float],
              tuning: dict) -> Image.Image:
    """Feathered ellipse over the region the sampler may repaint."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2 - tuning['mask_up'] * h
    rx, ry = w * tuning['mask_w'] / 2, h * tuning['mask_h'] / 2
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(tuning['feather']))


def swap_first(photo: Path, face: Path, index: int, work: Path) -> Path:
    """Photoreal base: swap.py's photo stack, everything else untouched."""
    out = work / f'swapped{photo.suffix.lower()}'
    command = [str(PYTHON), str(ROOT / 'swap.py'),
               '--video', str(photo), '--out', str(out),
               '--map', f'{index}={face}']
    print('swapping (photoreal base)…', flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0 or not out.is_file():
        detail = (result.stdout + result.stderr).strip().splitlines()
        reason = next((line for line in reversed(detail)
                       if line.startswith('swap.py: ')), detail[-1] if detail else '?')
        fail(f'base swap failed: {reason[:300]}')
    return out


def pick_best(candidates: list[Image.Image], face: Path,
              work: Path) -> tuple[int, list[float]]:
    """Rank renders by how close their face is to the person's real photos.

    Seeds vary the result more than any single knob here, so rendering a few
    and keeping the closest one is the cheapest quality there is. Scoring runs
    in comfyui's venv, where insightface lives.
    """
    if len(candidates) == 1:
        return 0, [0.0]
    paths = []
    for index, image in enumerate(candidates):
        path = work / f'try-{index}.png'
        image.save(path)
        paths.append(path)
    result = subprocess.run(
        [str(COMFY_PYTHON), str(ROOT / 'score_face.py'), str(face),
         *(str(p) for p in paths)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print('scoring failed, keeping the first render', flush=True)
        return 0, [0.0] * len(candidates)
    scores = {}
    for line in result.stdout.strip().splitlines():
        path, _, score = line.partition('\t')
        if score:
            scores[Path(path).name] = float(score)
    ordered = [scores.get(f'try-{i}.png', -1.0) for i in range(len(candidates))]
    return max(range(len(ordered)), key=lambda i: ordered[i]), ordered


def restore_texture(result: Image.Image, original: Image.Image,
                    amount: float = 0.55) -> Image.Image:
    """Put the camera's own skin texture back over the generated face.

    Diffusion output is characteristically too clean — that is what reads as
    plastic. Split both images into low frequencies (lighting, shape, which
    must come from the render) and high frequencies (pores, stubble, grain,
    which the camera actually recorded), then carry the original's high
    frequencies across. Nothing is hallucinated: the detail is the photo's.
    """
    import numpy

    radius = max(2, result.width // 220)
    new = numpy.asarray(result, dtype=numpy.float32)
    old = numpy.asarray(original.resize(result.size), dtype=numpy.float32)
    new_low = numpy.asarray(
        result.filter(ImageFilter.GaussianBlur(radius)), dtype=numpy.float32)
    old_low = numpy.asarray(
        original.resize(result.size).filter(ImageFilter.GaussianBlur(radius)),
        dtype=numpy.float32)
    # keep the render's own detail too — a straight swap of high frequencies
    # re-imposes the original person's features through the texture
    detail = (new - new_low) * (1 - amount) + (old - old_low) * amount
    return Image.fromarray(
        numpy.clip(new_low + detail, 0, 255).astype(numpy.uint8))


def match_light(result: Image.Image, original: Image.Image,
                mask: Image.Image) -> Image.Image:
    """Nudge the redrawn head into the scene's light.

    The sampler invents its own exposure, which reads as a paste-on in dark
    shots. Match mean/std per channel over the masked region to the original
    — halfway only, so the new head keeps its own shading.
    """
    import numpy

    weights = (numpy.asarray(mask, dtype=numpy.float32) / 255)[..., None]
    if weights.sum() < 1:
        return result
    new = numpy.asarray(result, dtype=numpy.float32)
    old = numpy.asarray(original.resize(result.size), dtype=numpy.float32)

    def stats(pixels):
        mean = (pixels * weights).sum((0, 1)) / weights.sum()
        variance = (((pixels - mean) ** 2) * weights).sum((0, 1)) / weights.sum()
        return mean, numpy.sqrt(variance) + 1e-6

    new_mean, new_std = stats(new)
    old_mean, old_std = stats(old)
    matched = (new - new_mean) / new_std * old_std + old_mean
    blended = new * 0.5 + matched * 0.5
    return Image.fromarray(numpy.clip(blended, 0, 255).astype(numpy.uint8))


def build_graph(crop_name: str, mask_name: str, ref_names: list[str],
                weight: float, seed: int, prompt: str,
                lora: str | None = None, tuning: dict | None = None) -> dict:
    tuning = tuning or {'denoise': DENOISE, 'canny': CANNY_STRENGTH, 'cfg': CFG}
    graph = {
        '1': {'class_type': 'CheckpointLoaderSimple',
              'inputs': {'ckpt_name': CHECKPOINT}},
        '2': {'class_type': 'InstantIDModelLoader',
              'inputs': {'instantid_file': INSTANTID}},
        '3': {'class_type': 'InstantIDFaceAnalysis', 'inputs': {'provider': 'CPU'}},
        '4': {'class_type': 'ControlNetLoader',
              'inputs': {'control_net_name': CONTROLNET}},
        '10': {'class_type': 'LoadImage', 'inputs': {'image': crop_name}},
        '11': {'class_type': 'LoadImageMask',
               'inputs': {'image': mask_name, 'channel': 'red'}},
        '12': {'class_type': 'CLIPTextEncode',
               'inputs': {'clip': ['1', 1], 'text': prompt}},
        '13': {'class_type': 'CLIPTextEncode',
               'inputs': {'clip': ['1', 1],
                          # ^ both rewired through the LoRA below when trained
                          # press-photo identities drag a stock-photo prior in
                          # with them — the watermark terms are load-bearing
                          'text': 'watermark, text, letters, words, logo, '
                                  'caption, stamp, stock photo, cartoon, '
                                  'painting, illustration, blurry, deformed, '
                                  'disfigured, low quality'}},
        '14': {'class_type': 'VAEEncode',
               'inputs': {'pixels': ['10', 0], 'vae': ['1', 2]}},
        '15': {'class_type': 'SetLatentNoiseMask',
               'inputs': {'samples': ['14', 0], 'mask': ['11', 0]}},
    }
    # identity references: load each, then chain into one batch. Node keys are
    # named, not numbered — a numeric scheme collided with the fixed nodes once
    # a face had enough photos (18 refs overwrote the sampler and save nodes).
    previous = None
    for index, name in enumerate(ref_names):
        node = f'ref{index}'
        graph[node] = {'class_type': 'LoadImage', 'inputs': {'image': name}}
        if previous is None:
            previous = [node, 0]
        else:
            batch = f'refbatch{index}'
            graph[batch] = {'class_type': 'ImageBatch',
                            'inputs': {'image1': previous, 'image2': [node, 0]}}
            previous = [batch, 0]
    model_source, clip_source = ['1', 0], ['1', 1]
    if lora:
        graph['5'] = {'class_type': 'LoraLoader', 'inputs': {
            'model': ['1', 0], 'clip': ['1', 1], 'lora_name': lora,
            'strength_model': 0.9, 'strength_clip': 0.9}}
        model_source, clip_source = ['5', 0], ['5', 1]
        graph['12']['inputs']['clip'] = clip_source
        graph['13']['inputs']['clip'] = clip_source
    graph['40'] = {'class_type': 'ApplyInstantID', 'inputs': {
        'instantid': ['2', 0], 'insightface': ['3', 0], 'control_net': ['4', 0],
        'image': previous, 'model': model_source,
        'positive': ['12', 0], 'negative': ['13', 0],
        'weight': round(weight, 2), 'start_at': 0.0, 'end_at': 1.0,
        'image_kps': ['10', 0], 'mask': ['11', 0],
    }}
    # edge conditioning from the original crop pins pose, expression and
    # framing — only the identity should change, not the scene or the face's
    # attitude. Soft strength, released at 70%, so identity can still reshape.
    graph['50'] = {'class_type': 'ControlNetLoader',
                   'inputs': {'control_net_name': CANNY_CONTROLNET}}
    graph['51'] = {'class_type': 'Canny', 'inputs': {
        'image': ['10', 0], 'low_threshold': 0.3, 'high_threshold': 0.65}}
    graph['52'] = {'class_type': 'ControlNetApplyAdvanced', 'inputs': {
        'positive': ['40', 1], 'negative': ['40', 2],
        'control_net': ['50', 0], 'image': ['51', 0],
        'strength': tuning['canny'], 'start_percent': 0.0, 'end_percent': 0.9,
        'vae': ['1', 2],
    }}
    graph['41'] = {'class_type': 'KSampler', 'inputs': {
        'model': ['40', 0], 'positive': ['52', 0], 'negative': ['52', 1],
        'latent_image': ['15', 0], 'seed': seed, 'steps': 30,
        'cfg': tuning['cfg'], 'sampler_name': 'dpmpp_2m',
        'scheduler': 'karras', 'denoise': tuning['denoise'],
    }}
    graph['42'] = {'class_type': 'VAEDecode',
                   'inputs': {'samples': ['41', 0], 'vae': ['1', 2]}}
    graph['43'] = {'class_type': 'SaveImage',
                   'inputs': {'images': ['42', 0], 'filename_prefix': 'meme'}}
    return graph


def run_graph(graph: dict) -> Image.Image:
    queued = api('/prompt', {'prompt': graph, 'client_id': uuid.uuid4().hex})
    if 'prompt_id' not in queued:
        fail(f'comfyui rejected the workflow: {json.dumps(queued)[:400]}')
    prompt_id = queued['prompt_id']
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(2)
        history = api(f'/history/{prompt_id}').get(prompt_id)
        if not history:
            continue
        status = history.get('status', {})
        if status.get('status_str') == 'error':
            messages = [m for kind, m in status.get('messages', [])
                        if kind == 'execution_error']
            detail = messages[-1].get('exception_message', '?') if messages else '?'
            fail(f'render failed: {detail[:300]}')
        outputs = history.get('outputs', {})
        for node in outputs.values():
            for image in node.get('images', []):
                query = urllib.parse.urlencode(image)
                with urllib.request.urlopen(f'{COMFY_URL}/view?{query}', timeout=60) as r:
                    return Image.open(io.BytesIO(r.read())).convert('RGB')
    fail('render timed out after 15 minutes')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--photo', required=True, help='target photo')
    parser.add_argument('--face', required=True,
                        help='face photo, or a directory of photos of the person')
    parser.add_argument('--index', type=int, default=0,
                        help='which face to replace, left to right (default 0)')
    parser.add_argument('--likeness', type=int, default=50, metavar='0..100',
                        help='identity strength (default 50)')
    parser.add_argument('--prompt', default='candid photograph of a person, '
                        'natural skin texture, film grain, sharp focus',
                        help='what the repainted region should be')
    parser.add_argument('--whole-head', action='store_true',
                        help='redraw head shape and hairline too — changes '
                             'proportions, but loses hats and hairstyles')
    parser.add_argument('--no-swap', action='store_true',
                        help='skip the photoreal swap stage (diffusion only)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tries', type=int, default=3, metavar='N',
                        help='render N seeds and keep the one that scores '
                             'closest to the real face (default 3; --seed '
                             'pins a single try)')
    parser.add_argument('--texture', type=float, default=0.55, metavar='0..1',
                        help='how much of the original photo\'s skin texture '
                             'to carry over the render (default 0.55)')
    parser.add_argument('--no-texture', action='store_true',
                        help='skip the texture restore pass')
    parser.add_argument('--out', required=True, help='output image path')
    args = parser.parse_args()

    photo = Path(args.photo).expanduser().resolve()
    face = Path(args.face).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not photo.is_file():
        fail(f'photo not found: {photo}')
    if photo.suffix.lower() not in IMAGE_EXTS:
        fail(f'photo must be one of {sorted(IMAGE_EXTS)}')
    if not face.exists():
        fail(f'face not found: {face}')
    if not 0 <= args.likeness <= 100:
        fail(f'--likeness {args.likeness} outside 0..100')
    out.parent.mkdir(parents=True, exist_ok=True)

    tuning = WHOLE_HEAD if args.whole_head else {
        'crop_scale': CROP_SCALE, 'mask_w': MASK_W, 'mask_h': MASK_H,
        'mask_up': MASK_UP, 'feather': FEATHER, 'denoise': DENOISE,
        'canny': CANNY_STRENGTH, 'cfg': CFG,
    }

    faces = scan_faces(photo)
    if not faces:
        fail('no face found in the photo')
    if not 0 <= args.index < len(faces):
        fail(f'--index {args.index}: the photo has {len(faces)} face(s), '
             f'numbered 0..{len(faces) - 1} left to right')

    # stage 1: photoreal swap. Everything the diffusion pass must not touch —
    # the hat, the hair, the light — is already correct after this, and the
    # face it draws over is the right person's shape to begin with.
    work = Path(tempfile.mkdtemp(prefix='meme-'))
    try:
        base = photo
        if not (args.no_swap or args.whole_head):
            base = swap_first(photo, face, args.index, work)
            rescanned = scan_faces(base)
            if len(rescanned) == len(faces):
                faces = rescanned          # box may shift a little after swap
        box = faces[args.index]

        original = Image.open(base).convert('RGB')
        side = int(tuning['crop_scale'] * max(box['w'], box['h']))
        side = min(side, min(original.size))
        cx, cy = box['x'] + box['w'] / 2, box['y'] + box['h'] / 2
        left = int(min(max(cx - side / 2, 0), original.width - side))
        top = int(min(max(cy - side / 2, 0), original.height - side))
        crop = original.crop((left, top, left + side, top + side))
        scale = 1024 / side
        crop_1024 = crop.resize((1024, 1024), Image.LANCZOS)
        # texture comes from the untouched camera file, not from the swapped
        # base — facefusion's enhancer has already smoothed that one
        camera = Image.open(photo).convert('RGB').crop(
            (left, top, left + side, top + side)).resize((1024, 1024), Image.LANCZOS)
        mask = head_mask(1024, ((box['x'] - left) * scale, (box['y'] - top) * scale,
                                box['w'] * scale, box['h'] * scale), tuning)
        render(args, face, out, original, crop_1024, camera, mask,
               left, top, side, tuning, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def render(args, face: Path, out: Path, original: Image.Image,
           crop_1024: Image.Image, camera: Image.Image, mask: Image.Image,
           left: int, top: int, side: int, tuning: dict, work: Path) -> None:
    ensure_server()
    stamp = uuid.uuid4().hex[:8]
    buffer = io.BytesIO(); crop_1024.save(buffer, 'PNG')
    upload(f'meme-{stamp}-crop.png', buffer.getvalue())
    buffer = io.BytesIO(); mask.convert('RGB').save(buffer, 'PNG')
    upload(f'meme-{stamp}-mask.png', buffer.getvalue())
    ref_names = []
    for index, ref in enumerate(reference_photos(face)):
        name = f'meme-{stamp}-ref{index}{ref.suffix.lower()}'
        upload(name, ref.read_bytes())
        ref_names.append(name)

    weight = 0.4 + args.likeness / 100 * 0.9   # 0.4 subtle .. 1.3 unmistakable
    # a trained per-person LoRA beats embedding averaging — use it when it
    # exists (train_face.py --person <name> creates it)
    lora = None
    prompt = args.prompt
    lora_file = COMFY / 'models' / 'loras' / f'{face.name}.safetensors'
    if face.is_dir() and lora_file.is_file():
        lora = lora_file.name
        prompt = f'photo of ohwx person, {args.prompt}'
    tries = 1 if args.seed is not None else max(1, args.tries)
    seeds = [args.seed] if args.seed is not None else \
        [random.randrange(2 ** 32) for _ in range(tries)]
    print(f'refining: face #{args.index}, likeness {args.likeness} '
          f'(weight {weight:.2f}), denoise {tuning["denoise"]}, '
          f'{tries} seed{"s" if tries > 1 else ""}'
          + (f', lora {lora}' if lora else ''), flush=True)

    candidates = []
    for number, seed in enumerate(seeds, start=1):
        if tries > 1:
            print(f'  try {number} of {tries} (seed {seed})…', flush=True)
        candidates.append(run_graph(build_graph(
            f'meme-{stamp}-crop.png', f'meme-{stamp}-mask.png', ref_names,
            weight, seed, prompt, lora, tuning)))
    best, scores = pick_best(candidates, face, work)
    if tries > 1:
        ranked = ', '.join(f'{s:.3f}' for s in scores)
        print(f'  scores: {ranked} — keeping try {best + 1}', flush=True)
    result = candidates[best]

    # paste the repainted region back into the full-resolution photo
    result = match_light(result, crop_1024, mask)
    if not args.no_texture:
        result = restore_texture(result, camera, args.texture)
    result = result.resize((side, side), Image.LANCZOS)
    mask_full = mask.resize((side, side), Image.LANCZOS)
    original.paste(result, (left, top), mask_full)
    if out.suffix.lower() in {'.jpg', '.jpeg'}:
        original.save(out, quality=95)
    else:
        original.save(out)
    print(f'done: {out}', flush=True)


if __name__ == '__main__':
    main()
