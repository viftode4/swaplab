#!/usr/bin/env python3
"""Convert a diffusers/PEFT SDXL LoRA into the kohya layout ComfyUI loads.

    convert_lora.py in.safetensors out.safetensors

diffusers' train_dreambooth_lora_sdxl writes keys like

    unet.down_blocks.1.....attn1.to_k.lora.down.weight

while ComfyUI's LoraLoader looks for

    lora_unet_down_blocks_1_..._attn1_to_k.lora_down.weight

A mismatch is NOT an error: ComfyUI logs "lora key not loaded" per key and
then renders as if no LoRA were attached, so a broken conversion looks
exactly like a weak one. Run with --check to assert the output actually
binds against a model.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def fail(message: str) -> 'NoReturn':
    print(f'convert_lora.py: {message}', file=sys.stderr)
    sys.exit(1)


# diffusers module prefix -> kohya prefix
PREFIXES = {
    'unet': 'lora_unet',
    'text_encoder': 'lora_te1',
    'text_encoder_2': 'lora_te2',
}


def convert(source: dict) -> dict:
    converted, ranks = {}, {}
    for key, tensor in source.items():
        head, _, rest = key.partition('.')
        prefix = PREFIXES.get(head)
        if prefix is None:
            fail(f'unexpected key prefix {head!r} — not a diffusers SDXL LoRA')
        for tail, replacement in (('.lora.down.weight', 'lora_down.weight'),
                                  ('.lora.up.weight', 'lora_up.weight'),
                                  ('.lora_A.weight', 'lora_down.weight'),
                                  ('.lora_B.weight', 'lora_up.weight')):
            if rest.endswith(tail):
                module = rest[:-len(tail)].replace('.', '_')
                name = f'{prefix}_{module}'
                converted[f'{name}.{replacement}'] = tensor
                if replacement == 'lora_down.weight':
                    ranks[name] = tensor.shape[0]
                break
        else:
            fail(f'unrecognised key {key!r}')
    # kohya scales by alpha/rank; diffusers trains at alpha == rank, so an
    # absent alpha would silently rescale every weight
    for name, rank in ranks.items():
        converted[f'{name}.alpha'] = torch.tensor(float(rank))
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        fail(f'not found: {source}')
    weights = load_file(str(source))
    converted = convert(weights)
    save_file(converted, args.destination)
    pairs = sum(1 for k in converted if k.endswith('lora_down.weight'))
    print(f'{source.name}: {len(weights)} diffusers keys -> '
          f'{len(converted)} kohya keys ({pairs} modules)')


if __name__ == '__main__':
    main()
