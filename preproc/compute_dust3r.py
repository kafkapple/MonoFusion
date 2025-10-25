from __future__ import annotations

import sys
from pathlib import Path

import tyro

DUST3R_DIR = Path(__file__).resolve().parent / "Dust3R"
if DUST3R_DIR.exists():
    dir_str = str(DUST3R_DIR)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

from Dust3R.a_general_dust import Dust3RConfig, run as run_general_dust


def main(cfg: Dust3RConfig) -> None:
    """Entry point: thin wrapper around Dust3R/a_general_dust"""
    run_general_dust(cfg)


if __name__ == "__main__":
    main(tyro.cli(Dust3RConfig))
