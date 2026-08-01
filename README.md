# SwapLab

Local face-swap lab for making funny TikTok videos with friends. Everything
runs on-device (Apple Silicon, CoreML) — no cloud, no footage leaves the Mac.

**House rules:** consented faces only (you + friends who said yes), no
strangers/public figures/minors, nothing sexual, and the AI-generated label
goes on when posting. See `docs/superpowers/specs/` for the design.

## Layout

- `facefusion/` — the engine, cloned from
  [facefusion/facefusion](https://github.com/facefusion/facefusion),
  gitignored. Pinned commit: `3f81a8a78454089d720b8f318a12ae1702c4633b`.
- `swap.py` — headless CLI wrapper, the stable interface for everything else.
- `webapp/` — phone-friendly upload UI (Phase 2).
- `faces/`, `jobs/` — gitignored private media.

## Setup

```bash
git clone --depth 1 https://github.com/facefusion/facefusion facefusion
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r facefusion/requirements.txt
# ffmpeg via homebrew required
```

## Use

```bash
.venv/bin/python swap.py --video clip.mp4 --face vlad.jpg --out result.mp4
# quality: --quality fast|good|best   (default good)
# --audio voice.m4a   lip-sync the face to a track (works without --face too)
# --captions          burn local-whisper auto-subtitles into the video
```

Photos work too — an image target always runs the max-quality stills stack
(`--quality` is ignored):

```bash
.venv/bin/python swap.py --video group.jpg --face vlad.jpg --out swapped.jpg
# --all-faces            everyone in the photo becomes --face
# --map 0=ana --map 2=vlad   face #N (left to right) becomes that person
# --list-faces           print the numbered face boxes as JSON
```

Or the FaceFusion UI directly:

```bash
cd facefusion && ../.venv/bin/python facefusion.py run
```

## Webapp (phone upload)

```bash
.venv/bin/python webapp/app.py     # http://<mac-lan-or-tailscale-ip>:8877
```

Phone-friendly page: pick a consented face from the gallery (or add one),
upload a clip, watch the queue, download the result. Jobs are folders under
`jobs/<id>/` (`job.json`, `input.*`, `face.*`, `result.mp4`, `swap.log`);
a crashed worker leaves jobs re-runnable. LAN/Tailscale only — never expose
the port publicly.

Photo mode: toggle to "photo", pick or upload a picture (HEIC fine), tap a
numbered face on the image, then tap whose face goes in — or tap 👥 everyone
to swap every face at once.

Adding a **photo** to a person appends it. Adding a **capture video** queues an
`identity.py` build that *rebuilds* that person from the video (old set archived
to `faces/<person>/.old-<stamp>/`); the clip stays in the job folder, so a
rebuild never needs a re-upload. The build reports its own reason for failing —
"only N usable face candidates … dominant filter: size" means the video came in
too small (see the capture checklist below).

## Best identity + picking your swapper

Build a strong multi-angle identity from a short capture video (slow head
turn + expressions, window light, 4K — see the capture checklist in
docs/superpowers/specs/2026-07-31-face-identity-design.md):

```bash
.venv/bin/python identity.py --video vlad-angles.mov --video vlad-expressions.mov --person vlad
.venv/bin/python bench.py --face faces/vlad     # ranks every swapper on YOUR face
```

The benchmark winner goes into `facefusion-swaplab.ini` under
`[face_swapper]` so every swap uses it by default (`--swapper-model`
still overrides per run).

Want the absolute ceiling? A personal DFM: train with DeepFaceLab on a
rented NVIDIA GPU (~1-3 days, only your own footage uploaded), export the
.dfm, drop it in `facefusion/.assets/models/custom/`, and swap with
`--processors deep_swapper`. No local training — DFL needs CUDA.
