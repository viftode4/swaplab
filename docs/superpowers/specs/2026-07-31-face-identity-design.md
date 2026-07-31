# Best-quality personal face identity — builder + swapper benchmark

Date: 2026-07-31
Status: approved in conversation; spec review pending

## Purpose

Make swaps that use Vlad's face as the source look as close to him as the
local, zero-shot stack allows. Two deliverables:

1. **Identity builder** — turn capture video(s) of a person into the best
   possible multi-angle face directory (`faces/<person>/`).
2. **Swapper benchmark** — measure every swapper model against that identity
   on fixed targets, rank them, and lock the winner in as SwapLab's default.

A cloud-trained personal DFM (DeepFaceLab on a rented NVIDIA GPU, consumed by
facefusion's `deep_swapper` from `.assets/models/custom/`) is documented as an
optional follow-up. It is NOT built here; local training is not viable
(no Apple-Silicon DFL, and the compute gap makes it 1-3 weeks vs 1-3 days).

Ground rules unchanged: consented faces only, on-device, label AI output.

## Background / constraints

- Zero-shot swappers need no training; source quality bounds identity
  likeness. Today `faces/vlad/` is one poor photo (extreme close-up,
  squinting) — the known bottleneck.
- The webapp's `sample_face_video` grabs 8 evenly spaced frames, blur and
  all. The builder replaces that approach for deliberate identity building;
  the webapp path stays as the quick-add convenience.
- facefusion internals available at the pinned commit: detector
  (`face_creator.get_many_faces`), landmarks + pose, and ArcFace embeddings
  (`face_recognizer`) — reachable the same way `scan_faces.py` already
  imports them (cwd = the checkout, venv python).
- Capture protocol (user-side): 4K, window daylight, slow head turn
  left-right + up-down (clip A), expression sweep (clip B), optional
  second-lighting repeat (clip C). Clips arrive via the iCloud inbox or any
  local path.

## Design

### 1. `identity.py` (repo root, CLI like swap.py)

```
.venv/bin/python identity.py --video vlad-angles.mov [--video vlad-expressions.mov ...] \
    --person vlad [--keep 18] [--dry-run]
```

Pipeline per video:
1. Decode frames at ~4 fps (ffmpeg pipe, like `content_box` does).
2. For each frame: detect the single most prominent face (facefusion
   detector via a helper alongside `scan_faces.py`); record detector score,
   face size, **sharpness** (variance of Laplacian over the face crop), and
   **pose** (yaw/pitch from the 5-point landmarks — nose position relative
   to the eye line gives a good-enough yaw/pitch proxy; no new models).
3. Filter: face height ≥ 256 px, detector score ≥ 0.7, sharpness above the
   40th percentile of that video (drops motion blur).
4. Bucket the survivors into a fixed pose grid (yaw: far-left / left /
   frontal / right / far-right x pitch: down / level / up — 15 buckets)
   and keep the sharpest frame per bucket, then fill remaining slots (up to
   `--keep`, default 18) with the sharpest unused frames from distinct
   videos (so expression variety from clip B survives).
5. Write crops (full frame is unnecessary — save the face crop padded 60%)
   as JPEG q95 into `faces/<person>/`, named `photo-<n>.jpg`, after moving
   any existing photos to `faces/<person>/.old-<timestamp>/` (nothing is
   deleted; the webapp sees only the new set).
6. Print a per-bucket report (angle coverage, sharpness range) and fail
   loudly if fewer than 8 usable frames survive ("reshoot: <reason>",
   e.g. all frames blurry / face too small).

`--dry-run` runs steps 1-4 and prints the report without touching faces/.

### 2. `bench.py` (repo root)

```
.venv/bin/python bench.py --face faces/vlad [--targets bench-targets/] [--models all]
```

- Targets: a small fixed set under `bench-targets/` (gitignored), seeded by
  the user or sampled once from existing clips — 3-4 photos/frames with one
  clear face each, varied pose/lighting. Reused across runs so scores are
  comparable over time.
- Models benchmarked (each with its max pixel boost from facefusion's
  choices table): `hyperswap_1a_256`, `hyperswap_1b_256`, `hyperswap_1c_256`,
  `ghost_1_256`, `ghost_2_256`, `ghost_3_256`, `simswap_256`,
  `inswapper_128_fp16`. First run downloads missing models (one-time).
- For each model x target: invoke facefusion headless directly (same
  config-path and venv as swap.py, `--processors face_swapper` ONLY, the
  model's max pixel boost, left-to-right selector) — swap.py's photo stack
  is not used because it hardwires the enhancer/expression restorer, and the
  benchmark must measure the swapper, not the polish.
- Score: ArcFace cosine similarity between the swapped face's embedding and
  the identity's averaged reference embedding (facefusion
  `face_recognizer`, same embedding family the swapper conditions on),
  plus sharpness of the swapped face region. Report per-model mean
  similarity and a ranking table.
- Contact sheet: one grid image per target (original + each model's crop,
  labelled) written to `bench-results/<timestamp>/` (gitignored) so the
  final call is made by eye, not just by metric.

### 3. Locking the winner

The chosen model goes into `facefusion-swaplab.ini` under
`[face_swapper] face_swapper_model = <winner>` (config-path already flows
into every swap). `swap.py --swapper-model` still overrides per run. The
choice and scores get a line in the README (Use section).

### 4. Docs

README: capture checklist (condensed), identity.py + bench.py usage, and a
short "personal DFM (optional, cloud)" paragraph: DFL on a rented NVIDIA
box, export .dfm, drop into `facefusion/.assets/models/custom/`, use with
`--processors deep_swapper`. No tooling built for it.

## Error handling

- identity.py: no faces / all filtered → named reason + reshoot hint; never
  leaves `faces/<person>/` empty (old set restored on abort).
- bench.py: a model that fails (download, inference) is reported and skipped,
  not fatal; needs ≥ 2 models to produce a ranking.
- Both tools refuse video files with no face in the first 100 frames early,
  before long processing.

## Testing (manual QA)

- identity.py on the two real capture clips: report shows ≥ 12 kept frames
  across ≥ 8 pose buckets; visual check of the crops (sharp, varied).
- identity.py --dry-run leaves faces/ untouched.
- Abort mid-run (Ctrl-C) → old faces/vlad restored.
- bench.py end-to-end on 3 targets: ranking table + contact sheets exist;
  eyeball agreement between metric ranking and visual ranking.
- A webapp photo swap after locking the winner uses the new default (visible
  in swap.log's command line).

## Out of scope

- DFM training or any cloud tooling (runbook paragraph only).
- Webapp UI for identity building or benchmarking (CLI only; the webapp's
  quick-add face path stays as-is).
- Changing the video/photo swap pipelines themselves.
