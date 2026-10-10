"""Shared location for one reproducible experiment batch."""

import os
from pathlib import Path


def results_root() -> Path:
    """Use a separate result tree when CTCPGRN_RESULTS_ROOT is set."""
    value = os.environ.get("CTCPGRN_RESULTS_ROOT", "results")
    if not value.strip():
        raise ValueError("CTCPGRN_RESULTS_ROOT must not be empty")
    return Path(value).expanduser().resolve()
