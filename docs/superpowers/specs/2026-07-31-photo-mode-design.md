# SwapLab photo mode — best-quality still-image face swap

Date: 2026-07-31
Status: approved in conversation; spec review pending

## Purpose

Swap faces in **pictures**, not just clips — a phone-friendly picture-editing
flow at the best quality the hardware can produce. Group photos support
per-face control: swap everyone to one person, or assign a different person to
each face.

Ground rules are unchanged from the main spec (consent only, label AI output,
LAN/Tailscale only).

## What exists today

- `swap.py` already lets an image target slip through (video-only passes are
  skipped) but runs it with **video-tuned** quality: enhancer blend capped at
  25 to avoid temporal flicker, 512px pixel boost, h264 encoder flags — all
  compromises a still image does not need.
- The webapp is video-only end to end: the upload gate rejects images, the
  result is hardcoded `result.mp4`, preview/thumbnail assume video.

## Design

### CLI (`swap.py`)

An image target (`.jpg/.jpeg/.png/.webp`) auto-branches to a photo path — no
separate mode flag; the file extension decides. `--quality` is **ignored** for
photos: they always run a stills-optimized max stack, because the constraints
that capped video quality (flicker, memory, encode time) do not exist for one
frame:

| Knob | Video `best` | Photo | Why different |
|---|---|---|---|
| pixel boost | 512x512 | **1024x1024** | one frame can afford the tiles |
| face-enhancer blend | 25 | **~80** | 25 exists only to damp flicker |
| expression restorer | 90 | 90 | keep the photo's original expression |
| mask | box+region, blur 0.4 | same | hairline handling already right |
| output | h264 q95 | **image quality 95, original resolution** | no encoder involved |

New flags:

- `--all-faces` — every detected face becomes the source person
  (`--face-selector-mode many`).
- `--map N=person` (repeatable) — face #N becomes that person (a photo file or
  a face directory, same as `--face`). Implemented as **chained single-face
  passes**: `--face-selector-mode reference --reference-face-position N`, each
  pass's output feeding the next as target.
- `--list-faces` — detect faces in the target and print numbered boxes as
  JSON (`[{index, x, y, w, h}]`). Drives the webapp tap UI. Uses facefusion's
  own detector at the pinned commit so the numbering agrees exactly with what
  gets swapped.

Face index ordering is **left-to-right** (`--face-selector-order left-right`)
in all photo passes, so indices are stable across chained passes and match the
numbered boxes shown in the UI.

Selection semantics: `--face` alone keeps today's behaviour (most prominent
face); `--face` + `--all-faces` swaps every face to that person; `--map` is
exclusive with both and names its sources inline.

Photo output gets its own conversion/validation path:

- Extension mismatch between facefusion's output and `--out` is converted with
  Pillow (already a dependency) at quality 95 — the current remux fallback
  re-encodes with h264 and would mangle an image.
- Validation opens the image with Pillow and checks resolution matches the
  target; the video QA (`check.py`) stays video-only.
- Dedupe, stabilize, chunking, and content-crop remain skipped for photos.

### Webapp

A **Clip | Photo** toggle on the main page.

Photo flow:

1. Upload a picture. `.heic` (iPhone camera default) is accepted and converted
   to JPEG on arrival via macOS `sips` — still fully on-device.
2. Server runs `--list-faces`; the photo renders with numbered tap-targets on
   each detected face.
3. Tap a face → pick a person from the existing consented-faces gallery.
   Repeat for any number of faces. Unassigned faces stay untouched. An
   **Everyone** shortcut maps all faces to one person.
4. Submit. The worker invokes `swap.py` with `--map`/`--all-faces` flags.
5. Result view shows the full-res image with a save/download button.

Infra: jobs gain `kind: photo | video` in `job.json`; the result file is
`result.jpg`/`result.png` and the result endpoint branches content-type on it;
the thumbnail is the image itself. Queue, worker loop, and face gallery are
reused untouched.

### Error handling

- No face detected → clear 400 at upload/detect time, before a job exists.
- `--map` index outside the detected range → fail immediately, reporting how
  many faces were found.
- A failed pass in a multi-pass chain cleans up intermediates and reports
  which pass died.

## Testing (manual QA)

- Single-face photo, default flow.
- Group photo: per-face mapping to two different people; verify each mapped
  face changed identity and unassigned faces are untouched.
- `--all-faces` meme mode.
- HEIC upload from the phone.
- PNG in / PNG out round trip.
- Output resolution equals input resolution in every case.

## Out of scope (rejected for now)

- Before/after wipe, per-face expression sliders in the webapp (the CLI
  `--edit`/`--puppet` sliders already exist for expression tweaks).
- Tap-precision face picking by pixel coordinates; numbered boxes are enough.
