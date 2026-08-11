#!/usr/bin/env python3
"""SwapLab webapp — phone-friendly upload UI over swap.py (Phase 2).

Run:
    .venv/bin/python webapp/app.py            # binds 0.0.0.0:8877 (LAN/Tailscale)
    .venv/bin/python webapp/app.py --port 9000 --host 100.x.y.z

Jobs are folders on disk (jobs/<id>/job.json); a crashed worker leaves the
job re-runnable. One worker thread = one CoreML job at a time.
"""

import argparse
import io
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / 'faces'
CLIPS = ROOT / 'clips'
PHOTOS = ROOT / 'photos'
# drop a clip here from the phone's Files app and it appears in the library
INBOX = (Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs'
         / 'SwapLab' / 'inbox')
THUMBS = CLIPS / '.thumbs'
JOBS = ROOT / 'jobs'
SWAP = ROOT / 'swap.py'
IDENTITY = ROOT / 'identity.py'
MEME = ROOT / 'meme.py'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
INDEX = Path(__file__).resolve().parent / 'static' / 'index.html'

VIDEO_EXTS = {'.mp4', '.mov', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
HEIC_EXTS = {'.heic', '.heif'}
AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.ogg', '.opus', '.flac'}
QUALITIES = {'fast', 'good', 'best'}
# Every swapper the pinned facefusion ships, ordered by visual family: near
# neighbours differ subtly, so a real difference in the picker stands out.
# ArcFace ranking disagrees with the eye here (it crowned inswapper while
# ghost_1/ghost_2 visibly feminized the face), which is the whole reason a
# photo renders all of them and you choose.
SWAPPER_MODELS = ['inswapper_128_fp16',
                  'hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256',
                  'ghost_1_256', 'ghost_2_256', 'ghost_3_256',
                  'simswap_256']
PHOTO_THUMB_WIDTH = 400     # the picker strip; the full file waits for a save
RETRY_PAUSE = 5
# CoreML reports contention as a failed prediction, not as a busy signal
TRANSIENT_MARKERS = ('Unable to compute the prediction', 'CoreML', 'ML Program')
EDIT_CONTROLS = {'smile', 'pout', 'grim', 'purse', 'lips', 'mouth-x', 'mouth-y',
                 'eyes', 'brows', 'gaze-x', 'gaze-y', 'pitch', 'yaw', 'roll'}

app = FastAPI(title='SwapLab')
wake = threading.Event()
# one render at a time, previews included: concurrent CoreML work fails with
# "Unable to compute the prediction" — that is how a contended benchmark run
# silently lost hyperswap_1b and 1c
render_lock = threading.Lock()


def slug(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '-', name).strip('-').lower()
    return cleaned or 'face'


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r'[a-z0-9-]+', job_id):
        raise HTTPException(400, 'bad job id')
    return JOBS / job_id


def read_job(path: Path) -> dict | None:
    try:
        return json.loads((path / 'job.json').read_text())
    except (OSError, ValueError):
        return None


def write_job(path: Path, data: dict) -> None:
    tmp = path / 'job.json.tmp'
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path / 'job.json')


def person_photos(person: Path) -> list[Path]:
    return sorted(p for p in person.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('.'))


def list_faces() -> list[dict]:
    """Each face is a directory of photos of one consented person."""
    faces = []
    if FACES.is_dir():
        for entry in sorted(FACES.iterdir()):
            if entry.is_dir() and not entry.name.startswith('.'):
                photos = person_photos(entry)
                if photos:
                    faces.append({'name': entry.name, 'count': len(photos)})
    return faces


def migrate_flat_faces() -> None:
    """Old layout was one photo per person as faces/<name>.<ext>."""
    if not FACES.is_dir():
        return
    for entry in list(FACES.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            person = FACES / entry.stem
            person.mkdir(exist_ok=True)
            entry.rename(person / f'photo-1{entry.suffix.lower()}')


async def save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open('wb') as handle:
        while chunk := await upload.read(1 << 20):
            handle.write(chunk)


def normalize_photo(path: Path) -> None:
    """Bake EXIF rotation into the pixels. Browsers honor the tag but the
    face detector reads raw pixels — a sideways face detects as no face."""
    from PIL import Image, ImageOps
    try:
        img = Image.open(path)
        fixed = ImageOps.exif_transpose(img)
        if fixed is not img:
            fixed.save(path, quality=95)
    except OSError:
        pass


def sips_to_jpeg(src: Path, dest: Path) -> bool:
    """HEIC → JPEG with macOS's own converter; stays on-device."""
    result = subprocess.run(
        ['sips', '-s', 'format', 'jpeg', '-s', 'formatOptions', '95',
         str(src), '--out', str(dest)], capture_output=True)
    return result.returncode == 0 and dest.is_file()


def scan_photo(photo: Path) -> dict:
    """Detect faces via swap.py --list-faces; cache beside the photo."""
    cache = photo.with_name(f'.{photo.stem}.faces.json')
    if cache.is_file() and cache.stat().st_mtime >= photo.stat().st_mtime:
        return json.loads(cache.read_text())
    result = subprocess.run(
        [str(PYTHON), str(SWAP), '--list-faces', '--video', str(photo)],
        capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise HTTPException(500, 'face detection failed — check the server log')
    data = json.loads(result.stdout.strip().splitlines()[-1])
    cache.write_text(json.dumps(data))
    return data


def list_photos() -> list[dict]:
    photos = []
    if PHOTOS.is_dir():
        entries = sorted(PHOTOS.iterdir(), key=lambda p: p.stat().st_mtime,
                         reverse=True)
        for entry in entries:
            if entry.suffix.lower() in IMAGE_EXTS and not entry.name.startswith('.'):
                photos.append({'name': entry.stem, 'file': entry.name})
    return photos


def list_clips() -> list[dict]:
    clips = []
    if CLIPS.is_dir():
        entries = sorted(CLIPS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for entry in entries:
            if entry.suffix.lower() in VIDEO_EXTS and not entry.name.startswith('.'):
                clips.append({'name': entry.stem, 'file': entry.name})
    return clips


def make_thumb(clip: Path) -> None:
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return
    THUMBS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, '-y', '-v', 'error', '-ss', '0.5', '-i', str(clip),
         '-frames:v', '1', '-vf', 'scale=320:-2',
         str(THUMBS / f'{clip.stem}.jpg')],
        capture_output=True)


async def store_clip(video: UploadFile) -> Path:
    """Save an uploaded video into the reusable clip library."""
    ext = Path(video.filename or '').suffix.lower()
    base = slug(Path(video.filename or 'clip').stem)
    dest = CLIPS / f'{base}{ext}'
    if dest.exists():
        dest = CLIPS / f'{base}-{secrets.token_hex(2)}{ext}'
    CLIPS.mkdir(exist_ok=True)
    await save_upload(video, dest)
    make_thumb(dest)
    return dest


@app.get('/')
def index() -> FileResponse:
    # phones cache aggressively; the page is tiny, always revalidate
    return FileResponse(INDEX, media_type='text/html',
                        headers={'Cache-Control': 'no-cache, must-revalidate'})


@app.get('/api/faces')
def api_faces() -> list[dict]:
    return list_faces()


@app.post('/api/clienterror')
async def api_client_error(report: dict) -> dict:
    """Phone-side failures are invisible from here; log them."""
    print(f'CLIENT-ERROR {json.dumps(report)[:800]}', flush=True)
    return {'logged': True}


def person_dir(name: str) -> Path:
    target = (FACES / name).resolve()
    if target.parent != FACES.resolve():
        raise HTTPException(404)
    return target


@app.get('/api/faces/{name}/thumb')
def api_face_thumb(name: str) -> FileResponse:
    person = person_dir(name)
    photos = person_photos(person) if person.is_dir() else []
    if not photos:
        raise HTTPException(404)
    return FileResponse(photos[0])


@app.post('/api/faces')
async def api_add_face(name: str = Form(...), photo: UploadFile = File(...)) -> dict:
    ext = Path(photo.filename or '').suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(400, 'a face needs a photo or a short video of them')
    person = FACES / slug(name)
    # no mkdir yet: a failed identity build would leave an empty person behind.
    # identity.py creates the dir itself once it has crops to write.
    count = len(person_photos(person)) if person.is_dir() else 0

    if ext in VIDEO_EXTS:
        # A capture video REBUILDS the identity through identity.py (pose
        # buckets, sharpness ranking, size/score floors) rather than grabbing
        # evenly spaced stills. It runs the detector over every sampled frame,
        # so it belongs in the job queue, not in this request.
        job_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}'
        path = JOBS / job_id
        path.mkdir(parents=True)
        # the source clip stays in the job folder: rebuilding with different
        # settings never needs a re-upload from the phone
        await save_upload(photo, path / f'input{ext}')
        write_job(path, {
            'id': job_id,
            'kind': 'identity',
            'status': 'queued',
            'quality': 'identity',
            'person': person.name,
            'face': person.name,
            'video': f'identity ← {photo.filename}',
            'created': time.time(),
            'error': None,
        })
        wake.set()
        return {'name': person.name, 'count': count, 'added': 0, 'job': job_id}

    person.mkdir(parents=True, exist_ok=True)
    dest = person / f'photo-{count + 1}{ext}'
    await save_upload(photo, dest)
    normalize_photo(dest)
    return {'name': person.name, 'count': count + 1, 'added': 1}


@app.delete('/api/faces/{name}')
def api_delete_face(name: str) -> dict:
    person = person_dir(name)
    if not person.is_dir():
        raise HTTPException(404)
    shutil.rmtree(person)
    return {'deleted': name}


@app.get('/api/clips')
def api_clips() -> list[dict]:
    return list_clips()


@app.post('/api/clips')
async def api_add_clip(video: UploadFile = File(...)) -> dict:
    ext = Path(video.filename or '').suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f'video must be one of {sorted(VIDEO_EXTS)}')
    clip = await store_clip(video)
    return {'name': clip.stem, 'file': clip.name}


@app.get('/api/clips/{file_name}/thumb')
def api_clip_thumb(file_name: str) -> FileResponse:
    thumb = (THUMBS / f'{Path(file_name).stem}.jpg').resolve()
    if thumb.parent != THUMBS.resolve() or not thumb.is_file():
        raise HTTPException(404)
    return FileResponse(thumb, media_type='image/jpeg')


@app.get('/api/photos')
def api_photos() -> list[dict]:
    return list_photos()


@app.post('/api/photos')
async def api_add_photo(photo: UploadFile = File(...)) -> dict:
    ext = Path(photo.filename or '').suffix.lower()
    if ext not in IMAGE_EXTS | HEIC_EXTS:
        raise HTTPException(400, f'photo must be one of {sorted(IMAGE_EXTS | HEIC_EXTS)}')
    base = slug(Path(photo.filename or 'photo').stem)
    PHOTOS.mkdir(exist_ok=True)
    dest = PHOTOS / f'{base}{ext}'
    if dest.exists():
        dest = PHOTOS / f'{base}-{secrets.token_hex(2)}{ext}'
    await save_upload(photo, dest)
    if ext in HEIC_EXTS:
        jpeg = dest.with_suffix('.jpg')
        if jpeg.exists():
            jpeg = PHOTOS / f'{base}-{secrets.token_hex(2)}.jpg'
        if not await run_in_threadpool(sips_to_jpeg, dest, jpeg):
            dest.unlink(missing_ok=True)
            raise HTTPException(400, 'could not convert that HEIC — try exporting as JPEG')
        dest.unlink(missing_ok=True)
        dest = jpeg
    normalize_photo(dest)
    scan = await run_in_threadpool(scan_photo, dest)
    if not scan.get('faces'):
        dest.unlink(missing_ok=True)
        raise HTTPException(400, 'no faces found in that photo')
    return {'name': dest.stem, 'file': dest.name, **scan}


def photo_path(file_name: str) -> Path:
    photo = (PHOTOS / file_name).resolve()
    if photo.parent != PHOTOS.resolve() or not photo.is_file():
        raise HTTPException(404)
    return photo


@app.get('/api/photos/{file_name}')
def api_photo(file_name: str) -> FileResponse:
    return FileResponse(photo_path(file_name))


@app.get('/api/photos/{file_name}/faces')
def api_photo_faces(file_name: str) -> dict:
    return scan_photo(photo_path(file_name))


def resolve_plan(source: Path, mapping: str) -> tuple[dict, dict]:
    """Validate a photo mapping and resolve its people to face dirs."""
    try:
        plan = json.loads(mapping) if mapping else {}
    except ValueError:
        raise HTTPException(400, 'mapping must be a JSON object')
    if not isinstance(plan, dict) or not plan:
        raise HTTPException(400, 'assign at least one face')
    found = len(scan_photo(source)['faces'])
    persons = {}
    for key, person_name in plan.items():
        if key != 'all':
            try:
                index = int(key)
            except ValueError:
                raise HTTPException(400, f'bad face number: {key}')
            if not 0 <= index < found:
                raise HTTPException(400, f'face #{key} not found — the photo has {found}')
        try:
            person = person_dir(str(person_name))
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            raise HTTPException(400, f'unknown face: {person_name}')
        photos = person_photos(person) if person.is_dir() else []
        if not photos:
            raise HTTPException(400, f'unknown face: {person_name}')
        persons[key] = person
    if 'all' in plan and len(plan) > 1:
        raise HTTPException(400, 'use "all" alone, or number the faces')
    return plan, persons


def stage_photo(source: Path, path: Path, persons: dict) -> None:
    """Lay out a job folder: the target photo plus one face dir per mapping."""
    path.mkdir(parents=True)
    shutil.copy(source, path / f'input{source.suffix.lower()}')
    for key, person in persons.items():
        slot = path / ('face' if key == 'all' else f'face-{key}')
        slot.mkdir()
        for photo_file in person_photos(person):
            shutil.copy(photo_file, slot / photo_file.name)


def render_preview(source: Path, mapping: str, likeness: int) -> bytes:
    """Swap-only render at the asked likeness, downscaled for the phone.

    Throwaway by design: a temp folder, never a queue entry, deleted on the
    way out. Waits on render_lock so it never fights a real render for CoreML.
    """
    _, persons = resolve_plan(source, mapping)
    path = JOBS / f'.preview-{secrets.token_hex(4)}'
    try:
        stage_photo(source, path, persons)
        photo_file = next(f for f in path.iterdir() if f.stem == 'input')
        out = path / f'result{photo_file.suffix.lower()}'
        command = photo_command(path, photo_file, out) + ['--preview']
        if likeness != 50:
            command += ['--likeness', str(likeness)]
        with render_lock:
            result = subprocess.run(
                command, cwd=ROOT, timeout=600,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                capture_output=True, text=True)
        if result.returncode != 0 or not out.is_file():
            raise HTTPException(500, failure_reason(result.stdout + result.stderr))
        from PIL import Image
        img = Image.open(out)
        img.thumbnail((1000, 4000))
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, 'JPEG', quality=82)
        return buffer.getvalue()
    finally:
        shutil.rmtree(path, ignore_errors=True)


@app.post('/api/preview')
async def api_preview(photo_name: str = Form(''), mapping: str = Form(''),
                      likeness: int = Form(50)) -> Response:
    if not 0 <= likeness <= 100:
        raise HTTPException(400, 'likeness must be 0..100')
    source = photo_path(photo_name)         # 404s on nonsense
    image = await run_in_threadpool(render_preview, source, mapping, likeness)
    return Response(content=image, media_type='image/jpeg')


@app.post('/api/jobs')
async def api_create_job(
    video: UploadFile | None = File(None),
    clip_name: str = Form(''),
    quality: str = Form('good'),
    face_name: str = Form(''),
    face_photo: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    captions: bool = Form(False),
    edit: str = Form(''),
    screen_recording: bool = Form(False),
    all_faces: bool = Form(False),
    photo_name: str = Form(''),
    mapping: str = Form(''),
    likeness: int = Form(50),
    meme: bool = Form(False),
) -> dict:
    if photo_name:
        if not 0 <= likeness <= 100:
            raise HTTPException(400, 'likeness must be 0..100')
        source = photo_path(photo_name)     # 404s on nonsense
        plan, persons = await run_in_threadpool(resolve_plan, source, mapping)

        job_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}'
        path = JOBS / job_id
        stage_photo(source, path, persons)
        if meme:
            # whole-head diffusion render: one pass per face, no model sweep
            found = len(scan_photo(source)['faces'])
            indices = list(range(found)) if 'all' in plan \
                else sorted(int(k) for k in plan)
            write_job(path, {
                'id': job_id, 'kind': 'meme', 'status': 'queued',
                'quality': 'meme', 'likeness': likeness,
                'heads': len(indices), 'done_heads': 0, 'indices': indices,
                'face': ', '.join(
                    (f'{int(k) + 1}→{plan[k]}' if k != 'all' else f'everyone→{plan[k]}')
                    for k in sorted(plan, key=lambda k: (k != 'all', int(k) if k != 'all' else -1))
                ),
                'video': photo_name, 'audio': None, 'captions': False,
                'edit': None, 'screen_recording': False,
                'created': time.time(), 'error': None,
            })
            wake.set()
            return {'id': job_id}
        write_job(path, {
            'id': job_id, 'kind': 'photo', 'status': 'queued',
            'quality': 'best', 'likeness': likeness,
            'models': SWAPPER_MODELS, 'done_models': 0, 'results': [],
            'face': ', '.join(
                (f'{int(k) + 1}→{plan[k]}' if k != 'all' else f'everyone→{plan[k]}')
                for k in sorted(plan, key=lambda k: (k != 'all', int(k) if k != 'all' else -1))
            ),
            'video': photo_name, 'audio': None, 'captions': False,
            'edit': None, 'screen_recording': False,
            'created': time.time(), 'error': None,
        })
        wake.set()
        return {'id': job_id}

    if quality not in QUALITIES:
        raise HTTPException(400, f'quality must be one of {sorted(QUALITIES)}')
    edits = {}
    if edit:
        try:
            edits = {k: float(v) for k, v in json.loads(edit).items()}
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(400, 'edit must be a JSON object of control: value')
        bad = [k for k in edits if k not in EDIT_CONTROLS]
        if bad or any(not -1.0 <= v <= 1.0 for v in edits.values()):
            raise HTTPException(400, f'bad edit controls: {bad or "value outside -1..1"}')

    if video is not None and video.filename:
        video_ext = Path(video.filename).suffix.lower()
        if video_ext not in VIDEO_EXTS:
            raise HTTPException(400, f'video must be one of {sorted(VIDEO_EXTS)}')
        clip = await store_clip(video)
        video_label = video.filename
    elif clip_name:
        matches = [c for c in list_clips() if c['name'] == clip_name]
        if not matches:
            raise HTTPException(400, f'unknown clip: {clip_name}')
        clip = CLIPS / matches[0]['file']
        video_label = clip_name
    else:
        raise HTTPException(400, 'pick a clip or upload a video')

    job_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}'
    path = JOBS / job_id
    path.mkdir(parents=True)

    shutil.copy(clip, path / f'input{clip.suffix.lower()}')

    if face_photo is not None and face_photo.filename:
        face_ext = Path(face_photo.filename).suffix.lower()
        if face_ext not in IMAGE_EXTS:
            shutil.rmtree(path)
            raise HTTPException(400, f'face photo must be one of {sorted(IMAGE_EXTS)}')
        await save_upload(face_photo, path / f'face{face_ext}')
        normalize_photo(path / f'face{face_ext}')
        face_label = 'uploaded photo'
    elif face_name:
        try:
            person = person_dir(face_name)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            shutil.rmtree(path)
            raise HTTPException(400, f'unknown face: {face_name}')
        photos = person_photos(person) if person.is_dir() else []
        if not photos:
            shutil.rmtree(path)
            raise HTTPException(400, f'unknown face: {face_name}')
        (path / 'face').mkdir()
        for photo in photos:
            shutil.copy(photo, path / 'face' / photo.name)
        face_label = face_name
    else:
        shutil.rmtree(path)
        raise HTTPException(400, 'pick a face or upload a face photo')

    audio_label = None
    if audio is not None and audio.filename:
        audio_ext = Path(audio.filename).suffix.lower()
        if audio_ext not in AUDIO_EXTS:
            shutil.rmtree(path)
            raise HTTPException(400, f'audio must be one of {sorted(AUDIO_EXTS)}')
        await save_upload(audio, path / f'audio{audio_ext}')
        audio_label = audio.filename

    write_job(path, {
        'id': job_id,
        'status': 'queued',
        'quality': quality,
        'face': face_label,
        'video': video_label,
        'audio': audio_label,
        'captions': captions,
        'edit': edits or None,
        'screen_recording': screen_recording,
        'all_faces': all_faces,
        'created': time.time(),
        'error': None,
    })
    wake.set()
    return {'id': job_id}


PROGRESS_RE = re.compile(rb'(downloading|analysing|extracting|processing|merging):\s*(\d+)%')
STAGE_LABEL = {'downloading': 'fetching models', 'analysing': 'analysing',
               'extracting': 'reading frames', 'processing': 'swapping',
               'merging': 'writing video'}


def job_progress(path: Path) -> dict | None:
    """Last progress marker from swap.log (facefusion prints tqdm lines)."""
    log = path / 'swap.log'
    try:
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 8192))
            found = PROGRESS_RE.findall(handle.read())
        if found:
            stage, pct = found[-1]
            return {'stage': STAGE_LABEL.get(stage.decode(), stage.decode()),
                    'pct': int(pct)}
    except OSError:
        pass
    return None


@app.get('/api/jobs')
def api_jobs() -> list[dict]:
    jobs = []
    if JOBS.is_dir():
        for entry in JOBS.iterdir():
            data = read_job(entry)
            if data:
                if data.get('status') == 'running':
                    data['progress'] = job_progress(entry)
                jobs.append(data)
    jobs.sort(key=lambda j: j.get('created', 0), reverse=True)
    queued = sorted((j for j in jobs if j.get('status') == 'queued'),
                    key=lambda j: j.get('created', 0))
    for position, job in enumerate(queued, start=1):
        job['place'] = position
    return jobs[:30]


@app.get('/api/jobs/{job_id}')
def api_job(job_id: str) -> dict:
    data = read_job(job_dir(job_id))
    if not data:
        raise HTTPException(404)
    return data


RESULT_MEDIA = {'.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.webp': 'image/webp'}


def result_stems(job_id: str, model: str | None) -> list[str]:
    """A photo job holds one result per swapper; everything else holds one."""
    if model:
        if model not in SWAPPER_MODELS:
            raise HTTPException(404, 'unknown model')
        return [f'result-{model}']
    # no model asked for: the first that rendered, so old links still resolve
    data = read_job(JOBS / job_id) or {}
    picked = [f"result-{r['model']}" for r in (data.get('results') or [])
              if r.get('ok') and r.get('model')]
    return ['result', *picked]


@app.get('/api/jobs/{job_id}/result')
def api_job_result(job_id: str, model: str | None = None) -> FileResponse:
    path = job_dir(job_id)
    for stem in result_stems(job_id, model):
        for suffix, media in RESULT_MEDIA.items():
            result = path / f'{stem}{suffix}'
            if result.is_file():
                return FileResponse(result, media_type=media,
                                    filename=f'swap-{job_id}{result.suffix}')
    raise HTTPException(404, 'no result yet')


@app.get('/api/jobs/{job_id}/thumb')
def api_job_thumb(job_id: str, model: str | None = None) -> FileResponse:
    """Picker-sized copy, so eight results cost a phone one small strip."""
    path = job_dir(job_id)
    for stem in result_stems(job_id, model):
        thumb = path / f'{stem}-thumb.jpg'
        if thumb.is_file():
            return FileResponse(thumb, media_type='image/jpeg')
    return api_job_result(job_id, model)


@app.get('/api/jobs/{job_id}/preview')
def api_job_preview(job_id: str) -> FileResponse:
    """Mobile-data-friendly copy; falls back to the full result."""
    path = job_dir(job_id)
    preview = path / 'preview.mp4'
    if preview.is_file():
        return FileResponse(preview, media_type='video/mp4',
                            filename=f'swap-{job_id}.mp4')
    return api_job_result(job_id)


def make_preview(path: Path) -> None:
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return
    subprocess.run(
        [ffmpeg, '-y', '-v', 'error', '-i', str(path / 'result.mp4'),
         '-vf', 'scale=720:960:force_original_aspect_ratio=decrease:force_divisible_by=2',
         '-c:v', 'h264_videotoolbox', '-q:v', '45',
         '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart',
         str(path / 'preview.mp4')],
        capture_output=True)


def is_transient(text: str) -> bool:
    """A contended CoreML run looks exactly like a broken model. Tell them apart."""
    return any(marker in (text or '') for marker in TRANSIENT_MARKERS)


def failure_reason(text: str) -> str:
    """Our tools fail with a "<tool>.py: <reason>" line; prefer it over the
    last line, which is whatever progress output happened to come last."""
    lines = (text or '').strip().splitlines()
    reason = next((line for line in reversed(lines)
                   if re.match(r'^\w+\.py: ', line)), None)
    return reason or (lines[-1][-400:] if lines else 'swap failed')


def launch(command: list[str], log_path: Path, path: Path, data: dict) -> tuple[int, str]:
    """Run one tool subprocess to completion; return its code and log text.

    Unbuffered, or the tool's block-buffered stdout flushes at exit and lands
    AFTER the stderr failure line, scrambling the log's chronology. Own process
    group so a server restart can clean up the whole render tree.
    """
    with render_lock, log_path.open('w') as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            start_new_session=True)
        data['pid'] = process.pid
        write_job(path, data)
        returncode = process.wait()
    return returncode, log_path.read_text(errors='replace')


def photo_command(path: Path, photo_file: Path, out: Path) -> list[str]:
    command = [str(PYTHON), str(SWAP), '--video', str(photo_file), '--out', str(out)]
    all_dir = path / 'face'
    if all_dir.is_dir():
        command += ['--face', str(all_dir), '--all-faces']
    else:
        for slot in sorted(path.glob('face-*')):
            command += ['--map', f'{slot.name.split("-", 1)[1]}={slot}']
    return command


def make_photo_thumb(result: Path) -> None:
    """Small copy for the picker strip — eight full results is ~20MB to a phone."""
    from PIL import Image
    try:
        img = Image.open(result)
        img.thumbnail((PHOTO_THUMB_WIDTH, PHOTO_THUMB_WIDTH * 4))
        img.convert('RGB').save(thumb_path(result), 'JPEG', quality=82)
    except OSError:
        pass        # a missing thumb only costs the strip a tile


def thumb_path(result: Path) -> Path:
    return result.with_name(f'{result.stem}-thumb.jpg')


def run_photo_job(path: Path, data: dict) -> None:
    """One photo through every swapper, so the choice is made by eye.

    Strictly sequential: concurrent CoreML work makes models fail with
    "Unable to compute the prediction using ML Program" — that is how a
    contended benchmark run silently lost hyperswap_1b and 1c, the two that
    held facial structure best. One failed model never sinks the job.
    """
    photo_file = next((f for f in path.iterdir() if f.stem == 'input'), None)
    if not photo_file:
        data.update(status='failed', error='job folder is missing input files')
        write_job(path, data)
        return
    suffix = photo_file.suffix.lower()
    models = data.get('models') or [None]
    data.update(status='running', started=time.time(), error=None, results=[])
    write_job(path, data)

    results = []
    for model in models:
        out = path / (f'result-{model}{suffix}' if model else f'result{suffix}')
        log = path / (f'swap-{model}.log' if model else 'swap.log')
        command = photo_command(path, photo_file, out)
        if data.get('likeness', 50) != 50:
            command += ['--likeness', str(data['likeness'])]
        if model:
            command += ['--swapper-model', model]
        returncode, text = launch(command, log, path, data)
        ok = returncode == 0 and out.is_file()
        if not ok and is_transient(text):
            time.sleep(RETRY_PAUSE)
            returncode, text = launch(command, log, path, data)
            ok = returncode == 0 and out.is_file()
        if ok:
            make_photo_thumb(out)
        results.append({'model': model, 'ok': ok,
                        'error': None if ok else failure_reason(text)})
        data['results'] = results
        data['done_models'] = sum(1 for r in results if r['ok'])
        write_job(path, data)

    data['pid'] = None
    if any(r['ok'] for r in results):
        data.update(status='done', finished=time.time())
    else:
        data.update(status='failed', finished=time.time(),
                    error=next((r['error'] for r in results if r['error']), 'swap failed'))
    write_job(path, data)


def run_meme_job(path: Path, data: dict) -> None:
    """Whole-head renders via meme.py, chained when several faces map.

    Each pass rescans the intermediate image, so left-to-right indices are
    re-derived the same way swap.py's --map chain does it.
    """
    photo_file = next((f for f in path.iterdir() if f.stem == 'input'), None)
    if not photo_file:
        data.update(status='failed', error='job folder is missing input files')
        write_job(path, data)
        return
    all_dir = path / 'face'
    if all_dir.is_dir():
        steps = [(index, all_dir) for index in data.get('indices') or [0]]
    else:
        steps = sorted((int(slot.name.split('-', 1)[1]), slot)
                       for slot in path.glob('face-*'))
    data.update(status='running', started=time.time(), error=None,
                heads=len(steps), done_heads=0)
    write_job(path, data)

    current = photo_file
    for number, (index, slot) in enumerate(steps):
        last = number == len(steps) - 1
        out = path / (f'result{photo_file.suffix}' if last
                      else f'step-{number}{photo_file.suffix}')
        command = [str(PYTHON), str(MEME),
                   '--photo', str(current), '--face', str(slot),
                   '--index', str(index),
                   '--likeness', str(data.get('likeness', 50)),
                   '--out', str(out)]
        returncode, text = launch(command, path / f'meme-{number}.log', path, data)
        data['pid'] = None
        if returncode != 0 or not out.is_file():
            data.update(status='failed', finished=time.time(),
                        error=failure_reason(text))
            write_job(path, data)
            return
        current = out
        data['done_heads'] = number + 1
        write_job(path, data)
    make_photo_thumb(current)
    data.update(status='done', finished=time.time())
    write_job(path, data)


def run_job(path: Path, data: dict) -> None:
    if data.get('kind') == 'photo':
        run_photo_job(path, data)
        return
    if data.get('kind') == 'meme':
        run_meme_job(path, data)
        return
    result_file = None
    if data.get('kind') == 'identity':
        clip = next((f for f in path.iterdir() if f.stem == 'input'), None)
        if not clip or not data.get('person'):
            data.update(status='failed', error='job folder is missing input files')
            write_job(path, data)
            return
        command = [str(PYTHON), str(IDENTITY), '--video', str(clip),
                   '--person', data['person']]
    else:
        video = next((f for f in path.iterdir() if f.stem == 'input'), None)
        face = next((f for f in path.iterdir() if f.stem == 'face'), None)  # file or dir
        if not video or not face:
            data.update(status='failed', error='job folder is missing input files')
            write_job(path, data)
            return
        result_file = path / 'result.mp4'
        command = [str(PYTHON), str(SWAP),
                   '--video', str(video), '--face', str(face),
                   '--out', str(result_file), '--quality', data['quality']]
        audio = next((f for f in path.iterdir() if f.stem == 'audio'), None)
        if audio:
            command += ['--audio', str(audio)]
        if data.get('captions'):
            command += ['--captions']
        if data.get('screen_recording'):
            command += ['--screen-recording']
        if data.get('all_faces'):
            command += ['--all-faces']
        for control, value in (data.get('edit') or {}).items():
            command += ['--edit', f'{control}={value}']
        if data.get('swapper_model'):
            command += ['--swapper-model', data['swapper_model']]
        if data.get('enhancer_model'):
            command += ['--enhancer-model', data['enhancer_model']]
    data.update(status='running', started=time.time(), error=None)
    returncode, text = launch(command, path / 'swap.log', path, data)
    data['pid'] = None
    if returncode == 0 and (result_file is None or result_file.is_file()):
        if data.get('kind') != 'identity':
            make_preview(path)
        data.update(status='done', finished=time.time())
        if data.get('kind') == 'identity':
            # identity.py's last line reports what was written and what was
            # archived — the only useful outcome of a build with no result file
            report = text.strip()
            data['summary'] = report.splitlines()[-1] if report else None
    else:
        data.update(status='failed', finished=time.time(),
                    error=failure_reason(text))
    write_job(path, data)


def watch_inbox() -> None:
    """Import clips and face photos dropped into the iCloud inbox.

    Files are only taken once iCloud has finished syncing them, judged by
    the size holding steady across two passes.
    """
    sizes: dict[Path, int] = {}
    while True:
        try:
            entries = [p for p in INBOX.iterdir()
                       if p.is_file() and not p.name.startswith('.')]
        except OSError:
            entries = []
        for entry in entries:
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            if sizes.get(entry) != size or size == 0:
                sizes[entry] = size          # still arriving; check again later
                continue
            ext = entry.suffix.lower()
            try:
                if ext in VIDEO_EXTS:
                    dest = CLIPS / f'{slug(entry.stem)}{ext}'
                    if dest.exists():
                        dest = CLIPS / f'{slug(entry.stem)}-{secrets.token_hex(2)}{ext}'
                    CLIPS.mkdir(exist_ok=True)
                    shutil.move(str(entry), dest)
                    make_thumb(dest)
                elif ext in IMAGE_EXTS:
                    # "vlad.jpg" or "vlad-2.jpg" both land on the person "vlad"
                    name = slug(re.sub(r'-\d+$', '', entry.stem))
                    person = FACES / name
                    person.mkdir(parents=True, exist_ok=True)
                    dest = person / f'photo-{len(person_photos(person)) + 1}{ext}'
                    shutil.move(str(entry), dest)
                    normalize_photo(dest)
                else:
                    continue
            except OSError:
                continue
            sizes.pop(entry, None)
        time.sleep(4)


def worker() -> None:
    while True:
        queued = []
        if JOBS.is_dir():
            for entry in JOBS.iterdir():
                data = read_job(entry)
                if data and data.get('status') == 'queued':
                    queued.append((data.get('created', 0), entry, data))
        if queued:
            _, path, data = min(queued)
            run_job(path, data)
        else:
            wake.wait(timeout=2)
            wake.clear()


@app.on_event('startup')
def start_worker() -> None:
    JOBS.mkdir(exist_ok=True)
    FACES.mkdir(exist_ok=True)
    CLIPS.mkdir(exist_ok=True)
    PHOTOS.mkdir(exist_ok=True)
    migrate_flat_faces()
    # a crashed worker leaves 'running' jobs behind — kill any orphaned render
    # (it has no supervisor left to record its result) and make them re-runnable
    for entry in JOBS.iterdir():
        data = read_job(entry)
        if data and data.get('status') == 'running':
            if data.get('pid'):
                try:
                    os.killpg(os.getpgid(data['pid']), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            data.update(status='queued', pid=None)
            write_job(entry, data)
    INBOX.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=watch_inbox, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--host', default='0.0.0.0',
                        help='bind address (LAN/Tailscale only — never expose publicly)')
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level='info',
                access_log=True)


if __name__ == '__main__':
    main()
