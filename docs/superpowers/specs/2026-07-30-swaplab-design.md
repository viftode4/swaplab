# SwapLab — local face-swap lab for funny TikToks

Date: 2026-07-30
Status: approved (direction approved in conversation; spec review pending)

## Purpose

A local, hackable face-swap pipeline on Vlad's Mac (M4 Max, 48 GB) for making
funny TikTok videos with friends. Everything runs on-device; no cloud services
see the footage. Primary user is Vlad; later a small phone-friendly web UI so
friends on the same network (or Tailscale) can submit clips.

## Ground rules (non-negotiable, build them in)

- **Consent only**: swap faces of yourself and friends who said yes. No
  strangers, no public figures, no minors, nothing sexual.
- **Label output**: TikTok's AI-generated toggle goes on when posting; keep an
  "AI-generated" note in output metadata where practical.
- Local-only by default. The web UI binds to LAN/Tailscale, never the public
  internet.

## Approach

Reuse, don't rebuild: **FaceFusion is the engine** (cloned, run from source,
gitignored). We own a thin layer on top and extend via FaceFusion's processor
architecture when we outgrow defaults.

Rejected alternatives:
- From-scratch pipeline (insightface + GFPGAN + ffmpeg by hand): maximal
  control but repeats solved work; revisit per-module only if FaceFusion blocks
  us.
- Packaged desktop app for friends: heavy packaging cost, no current need.

## Architecture

```
swaplab/
  facefusion/        # cloned engine, gitignored, pinned by commit in README
  .venv/             # uv-managed Python 3.12, gitignored
  swap.py            # Phase 1: headless CLI wrapper (our engine API)
  webapp/            # Phase 2: FastAPI app + job queue + upload page
  faces/             # gitignored: consented face photos of friends
  jobs/              # gitignored: uploads + outputs
  docs/superpowers/specs/
```

### Pipeline (inside FaceFusion, for understanding/tuning)

video → ffmpeg frames → SCRFD detect+landmarks → align/crop →
ArcFace identity embed (source photo) → inswapper_128 swap →
GFPGAN/CodeFormer restore → mask + color-transfer paste-back →
ffmpeg reassemble + original audio.

Models (~1.1 GB total) download on first run; all ONNX via onnxruntime
CoreML execution provider.

### Phase 1 — `swap.py` (the contract everything builds on)

`swap.py --video clip.mp4 --face vlad.jpg --out result.mp4 [--quality fast|good|best]`

- Wraps FaceFusion headless mode; owns TikTok-tuned defaults (H.264, audio
  passthrough, vertical-friendly output).
- `fast` = swap only; `good` = swap + restore on frames where the face is
  large; `best` = restore everything.
- This CLI is the stable interface: the webapp and future automation call it
  (or the same underlying function), never FaceFusion directly.

### Phase 2 — webapp

- FastAPI + single-page upload form (phone-friendly), SQLite-or-folder job
  queue, one worker process (one GPU job at a time), progress + download link.
- Face gallery: named, consented faces ("Andrei", "Vlad") selectable instead
  of uploading a photo each time.

### Phase 3 — extensions (only when wanted, each isolated)

- Target-face selection by reference embedding (swap the right person in
  multi-person clips).
- Multi-face mode (everyone becomes a different friend).
- Meme templates (pre-set clips, one-tap).
- Lip-sync processor. Real-time is explicitly out of scope.

## Error handling

- `swap.py` exits nonzero with a readable message on: no face found in source
  photo, no faces in video, unsupported codec. Webapp surfaces the same
  message on the job.
- Jobs are files-on-disk; a crashed worker leaves the job re-runnable.

## Testing / verification

- Golden-path check: a bundled 5-second test clip + test face; `swap.py` run
  must produce a playable mp4 with audio, verified by ffprobe.
- Phase acceptance: Phase 0 = one successful swap via UI; Phase 1 = golden
  path passes from CLI; Phase 2 = phone upload → download roundtrip on LAN.

## Milestones

- Phase 0 (today): FaceFusion installed, first swap produced.
- Phase 1 (today/tomorrow): `swap.py` golden path green.
- Phase 2 (weekend): webapp usable from a phone.
