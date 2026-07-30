#!/usr/bin/env python3
"""SwapLab webapp — phone-friendly upload UI over swap.py (Phase 2).

Run:
    .venv/bin/python webapp/app.py            # binds 0.0.0.0:8877 (LAN/Tailscale)
    .venv/bin/python webapp/app.py --port 9000 --host 100.x.y.z

Jobs are folders on disk (jobs/<id>/job.json); a crashed worker leaves the
job re-runnable. One worker thread = one CoreML job at a time.
"""

import argparse
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / 'faces'
CLIPS = ROOT / 'clips'
THUMBS = CLIPS / '.thumbs'
JOBS = ROOT / 'jobs'
SWAP = ROOT / 'swap.py'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
INDEX = Path(__file__).resolve().parent / 'static' / 'index.html'

VIDEO_EXTS = {'.mp4', '.mov', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.ogg', '.opus', '.flac'}
QUALITIES = {'fast', 'good', 'best'}
EDIT_CONTROLS = {'smile', 'pout', 'grim', 'purse', 'lips', 'mouth-x', 'mouth-y',
                 'eyes', 'brows', 'gaze-x', 'gaze-y', 'pitch', 'yaw', 'roll'}

app = FastAPI(title='SwapLab')
wake = threading.Event()


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


def list_faces() -> list[dict]:
    faces = []
    if FACES.is_dir():
        for entry in sorted(FACES.iterdir()):
            if entry.suffix.lower() in IMAGE_EXTS and not entry.name.startswith('.'):
                faces.append({'name': entry.stem, 'file': entry.name})
    return faces


async def save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open('wb') as handle:
        while chunk := await upload.read(1 << 20):
            handle.write(chunk)


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
         '-frames:v', '1', '-vf', 'scale=320:-2', str(THUMBS / f'{clip.stem}.jpg')],
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
    return FileResponse(INDEX, media_type='text/html')


@app.get('/api/faces')
def api_faces() -> list[dict]:
    return list_faces()


@app.get('/api/faces/{file_name}')
def api_face_image(file_name: str) -> FileResponse:
    target = (FACES / file_name).resolve()
    if target.parent != FACES.resolve() or target.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404)
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


@app.post('/api/faces')
async def api_add_face(name: str = Form(...), photo: UploadFile = File(...)) -> dict:
    ext = Path(photo.filename or '').suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f'face photo must be one of {sorted(IMAGE_EXTS)}')
    FACES.mkdir(exist_ok=True)
    stem = slug(name)
    for stale in FACES.glob(f'{stem}.*'):  # same name replaces the old photo
        stale.unlink()
    dest = FACES / f'{stem}{ext}'
    await save_upload(photo, dest)
    return {'name': dest.stem, 'file': dest.name}


@app.get('/api/clips')
def api_clips() -> list[dict]:
    return list_clips()


@app.get('/api/clips/{file_name}/thumb')
def api_clip_thumb(file_name: str) -> FileResponse:
    thumb = (THUMBS / f'{Path(file_name).stem}.jpg').resolve()
    if thumb.parent != THUMBS.resolve() or not thumb.is_file():
        raise HTTPException(404)
    return FileResponse(thumb, media_type='image/jpeg')


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
) -> dict:
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
        face_label = 'uploaded photo'
    elif face_name:
        matches = [f for f in list_faces() if f['name'] == face_name]
        if not matches:
            shutil.rmtree(path)
            raise HTTPException(400, f'unknown face: {face_name}')
        source = FACES / matches[0]['file']
        shutil.copy(source, path / f'face{source.suffix.lower()}')
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
        'created': time.time(),
        'error': None,
    })
    wake.set()
    return {'id': job_id}


@app.get('/api/jobs')
def api_jobs() -> list[dict]:
    jobs = []
    if JOBS.is_dir():
        for entry in JOBS.iterdir():
            data = read_job(entry)
            if data:
                jobs.append(data)
    jobs.sort(key=lambda j: j.get('created', 0), reverse=True)
    return jobs[:30]


@app.get('/api/jobs/{job_id}')
def api_job(job_id: str) -> dict:
    data = read_job(job_dir(job_id))
    if not data:
        raise HTTPException(404)
    return data


@app.get('/api/jobs/{job_id}/result')
def api_job_result(job_id: str) -> FileResponse:
    result = job_dir(job_id) / 'result.mp4'
    if not result.is_file():
        raise HTTPException(404, 'no result yet')
    return FileResponse(result, media_type='video/mp4',
                        filename=f'swap-{job_id}.mp4')


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
         '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
         '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart',
         str(path / 'preview.mp4')],
        capture_output=True)


def run_job(path: Path, data: dict) -> None:
    video = next((f for f in path.iterdir() if f.stem == 'input'), None)
    face = next((f for f in path.iterdir() if f.stem == 'face'), None)
    if not video or not face:
        data.update(status='failed', error='job folder is missing input files')
        write_job(path, data)
        return
    command = [str(PYTHON), str(SWAP),
               '--video', str(video), '--face', str(face),
               '--out', str(path / 'result.mp4'), '--quality', data['quality']]
    audio = next((f for f in path.iterdir() if f.stem == 'audio'), None)
    if audio:
        command += ['--audio', str(audio)]
    if data.get('captions'):
        command += ['--captions']
    for control, value in (data.get('edit') or {}).items():
        command += ['--edit', f'{control}={value}']
    log = (path / 'swap.log').open('w')
    # own process group so a server restart can clean up the whole render tree
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
        start_new_session=True)
    data.update(status='running', started=time.time(), error=None, pid=process.pid)
    write_job(path, data)
    returncode = process.wait()
    log.close()
    data['pid'] = None
    if returncode == 0 and (path / 'result.mp4').is_file():
        make_preview(path)
        data.update(status='done', finished=time.time())
    else:
        tail = (path / 'swap.log').read_text(errors='replace')[-400:].strip()
        data.update(status='failed', finished=time.time(),
                    error=tail.splitlines()[-1] if tail else 'swap failed')
    write_job(path, data)


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
    threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--host', default='0.0.0.0',
                        help='bind address (LAN/Tailscale only — never expose publicly)')
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
