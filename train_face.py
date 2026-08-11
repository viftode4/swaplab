#!/usr/bin/env python3
"""Actually learn a face: train a per-person SDXL LoRA from their photo set.

    train_face.py --person charlie [--steps 800]

Trains on every photo in faces/<person>/ (the more angles the better — a
capture video through identity.py first gives the best set). The result
lands in comfyui/models/loras/<person>.safetensors and meme.py picks it up
automatically from then on. Runs on the M4 Max's GPU; budget roughly one to
three hours per person depending on --steps.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY = ROOT / 'comfyui'
COMFY_PYTHON = COMFY / '.venv' / 'bin' / 'python'
TRAINER = COMFY / 'train_dreambooth_lora_sdxl.py'
FACES = ROOT / 'faces'
LORAS = COMFY / 'models' / 'loras'
BASE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
# rare-token prompt: the LoRA binds the person to "ohwx person" without
# dragging the base model's idea of any real name into the render
INSTANCE_PROMPT = 'photo of ohwx person'


def fail(message: str) -> 'NoReturn':
    print(f'train_face.py: {message}', file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--person', required=True, help='name under faces/')
    # ~100 steps per photo is the working baseline for a character LoRA; 800
    # total left an 18-photo person at 44 each, well under-trained
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=1024,
                        help='SDXL native; identity lives in fine detail')
    args = parser.parse_args()

    person = FACES / args.person
    if not person.is_dir():
        fail(f'no face directory {person}')
    photos = sorted(p for p in person.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('.'))
    if len(photos) < 4:
        fail(f'{args.person} has {len(photos)} photos — need at least 4 '
             '(a capture video through identity.py gives a proper set)')
    if not TRAINER.is_file() or not COMFY_PYTHON.is_file():
        fail('training stack missing — run the meme-mode setup first')

    output = person / '.lora'
    output.mkdir(exist_ok=True)
    LORAS.mkdir(parents=True, exist_ok=True)

    # the trainer treats every image in the dir as training data; stage a
    # clean copy so .old archives and json sidecars never leak in
    with tempfile.TemporaryDirectory(prefix='train-') as staged:
        for photo in photos:
            shutil.copy(photo, Path(staged) / photo.name)
        command = [
            str(COMFY_PYTHON), str(TRAINER),
            '--pretrained_model_name_or_path', BASE_MODEL,
            '--instance_data_dir', staged,
            '--instance_prompt', INSTANCE_PROMPT,
            '--output_dir', str(output),
            '--rank', str(args.rank),
            '--resolution', str(args.resolution),
            # without this the trainer RandomCrops after resize, which cut the
            # face out of a quarter of the steps (measured on faces/vlad: the
            # whole face survived only 73% of random crops, 100% of centre ones)
            '--center_crop',
            '--train_batch_size', '1',
            '--learning_rate', '1e-4',
            '--lr_scheduler', 'constant_with_warmup',
            '--lr_warmup_steps', str(max(1, args.steps // 10)),
            '--max_train_steps', str(args.steps),
            '--checkpointing_steps', '10000',   # only the final weights matter
            '--seed', '42',
            '--gradient_checkpointing',
        ]
        print(f'training {args.person}: {len(photos)} photos, '
              f'{args.steps} steps at {args.resolution}px', flush=True)
        result = subprocess.run(
            command, cwd=COMFY,
            env={**os.environ, 'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                 'TOKENIZERS_PARALLELISM': 'false'})
        if result.returncode != 0:
            fail(f'training exited with {result.returncode}')

    weights = output / 'pytorch_lora_weights.safetensors'
    if not weights.is_file():
        fail(f'training finished but {weights} is missing')
    # diffusers writes PEFT-format keys; ComfyUI's loader wants kohya and
    # treats a mismatch as "lora key not loaded" per key, then renders as if
    # no LoRA were attached — a silent no-op that looks like weak training
    destination = LORAS / f'{args.person}.safetensors'
    convert = subprocess.run(
        [str(COMFY_PYTHON), str(ROOT / 'convert_lora.py'),
         str(weights), str(destination)],
        cwd=ROOT, capture_output=True, text=True)
    if convert.returncode != 0:
        fail(f'converting to kohya layout failed: '
             f'{(convert.stdout + convert.stderr).strip()[-300:]}')
    print(convert.stdout.strip(), flush=True)
    print(f'done: {destination} — meme.py will use it automatically', flush=True)


if __name__ == '__main__':
    main()
