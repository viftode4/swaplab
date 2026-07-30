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
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / 'faces'
JOBS = ROOT / 'jobs'
SWAP = ROOT / 'swap.py'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
INDEX = Path(__file__).resolve().parent / 'static' / 'index.html'

VIDEO_EXTS = {'.mp4', '.mov', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
QUALITIES = {'fast', 'good', 'best'}

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
    dest = FACES / f'{slug(name)}{ext}'
    await save_upload(photo, dest)
    return {'name': dest.stem, 'file': dest.name}


@app.post('/api/jobs')
async def api_create_job(
    video: UploadFile = File(...),
    quality: str = Form('good'),
    face_name: str = Form(''),
    face_photo: UploadFile | None = File(None),
) -> dict:
    if quality not in QUALITIES:
        raise HTTPException(400, f'quality must be one of {sorted(QUALITIES)}')
    video_ext = Path(video.filename or '').suffix.lower()
    if video_ext not in VIDEO_EXTS:
        raise HTTPException(400, f'video must be one of {sorted(VIDEO_EXTS)}')

    job_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}'
    path = JOBS / job_id
    path.mkdir(parents=True)

    await save_upload(video, path / f'input{video_ext}')

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

    write_job(path, {
        'id': job_id,
        'status': 'queued',
        'quality': quality,
        'face': face_label,
        'video': video.filename,
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


def run_job(path: Path, data: dict) -> None:
    video = next((f for f in path.iterdir() if f.stem == 'input'), None)
    face = next((f for f in path.iterdir() if f.stem == 'face'), None)
    if not video or not face:
        data.update(status='failed', error='job folder is missing input files')
        write_job(path, data)
        return
    data.update(status='running', started=time.time(), error=None)
    write_job(path, data)
    log = (path / 'swap.log').open('w')
    result = subprocess.run(
        [str(PYTHON), str(SWAP),
         '--video', str(video), '--face', str(face),
         '--out', str(path / 'result.mp4'), '--quality', data['quality']],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    log.close()
    if result.returncode == 0 and (path / 'result.mp4').is_file():
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
    # a crashed worker leaves 'running' jobs behind — make them re-runnable
    for entry in JOBS.iterdir():
        data = read_job(entry)
        if data and data.get('status') == 'running':
            data['status'] = 'queued'
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
