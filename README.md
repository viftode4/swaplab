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
