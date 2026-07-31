#!/usr/bin/env python3
"""Bootstrap facefusion's internals for in-process use by repo tools.

The checkout lives at <repo>/facefusion and the real package one level
deeper, so the repo root on sys.path shadows the package. boot() chdirs
into the checkout, fixes sys.path, and feeds facefusion's own arg parser
so state defaults + facefusion-swaplab.ini apply exactly as in a real swap.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACEFUSION = ROOT / 'facefusion'
CONFIG = ROOT / 'facefusion-swaplab.ini'


def boot(config_path: str | None = None, extra_args: list[str] | None = None) -> None:
    os.chdir(FACEFUSION)
    sys.path.insert(0, str(FACEFUSION))
    sys.argv = ['facefusion.py', 'headless-run',
                '--config-path', str(config_path or CONFIG),
                '--target-path', '/dev/null/unused.jpg',
                '--output-path', '/dev/null/unused.jpg',
                '--face-selector-order', 'left-right',
                '--processors', 'face_swapper',
                *(extra_args or [])]
    from facefusion.program import create_program
    from facefusion.args import apply_args
    from facefusion import state_manager
    apply_args(vars(create_program().parse_args()), state_manager.init_item)
