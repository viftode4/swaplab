# Multi-model photo picker

A photo renders through every swapper and you choose the winner by eye in
the app, instead of one globally configured model deciding for you.

## Why

The ranking metric and the eye disagree, twice measured. `bench.py` scores
ArcFace similarity, which inswapper optimizes directly; on vlad's identity
it crowned `inswapper_128_fp16` (0.90) while `ghost_1`/`ghost_2` (0.83/0.82)
visibly feminized the face — fuller redder lips, heavier brows, softened jaw
— and `hyperswap_1b`/`1c` (0.78/0.79) held bone structure best of all while
scoring near the bottom. No single default is right for every target either:
a model that wins on one photo loses on the next.

Rendering all of them costs little for a photo and settles the question with
the only judge that matters.

## Scope

Photos only. Video is out of scope and stays on the configured default.

The asymmetry is the reason, measured from this repo's own job history:

| target | one swap | all eight |
|---|---|---|
| photo | 16-20 s | ~2.5 min |
| video (6 s clip, `best`) | 377-660 s | 1-1.5 h |
| video (screen recording) | 1036-2034 s | up to 4.5 h |

A documented follow-up — not built — is a video preview flow: render a few
sampled frames of the target clip through each model as stills, pick from
that strip, then render the full clip once with the winner. The job schema
below does not preclude it.

## Design

A photo job gains `models`, `done_models` and `results`. The worker runs
`swap.py` once per model, appending `--swapper-model` to the command it
already builds, so face mapping (`--all-faces`, `--map N=person`) is
untouched. Each render writes `result-<model>.<ext>` plus a
`result-<model>-thumb.jpg`.

```
POST /api/jobs (photo) -> job.json { kind: photo, models: [...8], results: [] }
worker -> run_photo_job -> per model: swap.py --swapper-model X
                        -> result-X.jpg + thumb -> write_job (live progress)
GET /api/jobs/{id}/result?model=X  -> that render
GET /api/jobs/{id}/thumb?model=X   -> its picker tile
```

`models` is ordered by visual family (inswapper, hyperswap 1a/1b/1c, ghost
1/2/3, simswap) so neighbouring tiles differ subtly and a real difference
stands out.

### Sequential, deliberately

One model at a time. Concurrent CoreML work makes models fail with "Unable
to compute the prediction using ML Program" — that is how a benchmark run
overlapping other on-device work silently lost `hyperswap_1b` and `1c`, the
two that hold structure best. The single worker thread already enforces this.

### One failure never sinks the job

A model that fails is recorded in `results` with its reason and skipped;
transient CoreML failures retry once after a pause. The job fails only if
every model fails. Losing seven good renders to one bad one would repeat
the bug this behaviour exists to prevent.

### Thumbnails, not full renders

Eight full results is ~20 MB to a phone for a picker. Tiles load a ~400px
thumb (37 KB vs 224 KB measured); the full file is fetched only on save.

### Compatibility

Older photo jobs hold a single `result.<ext>`. `/result` with no `model`
serves `result` first, then the first successful model, so existing cards
and links keep resolving. `model` is checked against `SWAPPER_MODELS`, so
it cannot address paths outside the job folder.

## Verification

- 8/8 models rendered, live progress 0/8 -> 8/8, all endpoints 200
- a bogus model failed while the rest completed; job status `done`
- `?model=../etc/passwd` -> 404
- picker renders and switches selection in-browser; console clean
- thumb 37 KB vs full 224 KB
