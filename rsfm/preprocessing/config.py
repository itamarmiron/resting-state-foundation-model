"""Dataset config loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PATH_KEYS = ("fmriprep_dir", "output_dir", "template_mask")


def load_config(path: str | Path, output_dir: str | Path | None = None) -> dict:
    """Load a dataset YAML config, expanding ``~`` in path fields.

    ``output_dir`` overrides the configured one (used for test runs into a scratch dir).
    """
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    for key in PATH_KEYS:
        cfg[key] = Path(os.path.expanduser(cfg[key]))
    return cfg
