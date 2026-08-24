"""Shared filesystem configuration for the data pipeline."""

import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("VIDEO2CODE_DATA_ROOT", "./data_workspace")).expanduser().resolve()


def data_path(*parts: str) -> str:
    """Return a path below the configured dataset workspace."""
    return str(DATA_ROOT.joinpath(*parts))

