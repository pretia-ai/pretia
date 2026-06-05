"""Download and cache SWE-bench trajectory data."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = str(Path(__file__).parent / "data")

_DOWNLOAD_URLS = [
    (
        "https://raw.githubusercontent.com/swe-bench/experiments/main/"
        "evaluation/verified/20241029_OpenHands-CodeAct-2.1-sonnet-20241022/"
        "results/results.json"
    ),
]

_MANUAL_INSTRUCTIONS = """
Could not automatically download SWE-bench trajectory data.

Manual steps:
1. Visit https://github.com/swe-bench/experiments
2. Browse to evaluation/verified/ and pick an experiment directory
3. Download the results.json or output.jsonl file
4. Place the file in tests/synthetic/swebench/data/ as trajectories.json
5. Re-run this script
""".strip()


def download_swebench_data(
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> str:
    """Download SWE-bench trajectory data. Returns path to cached data file."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for candidate in ("trajectories.json", "trajectories.jsonl", "results.json"):
        path = cache_path / candidate
        if path.exists() and path.stat().st_size > 100:
            logger.info("Using cached SWE-bench data: %s", path)
            return str(path)

    for url in _DOWNLOAD_URLS:
        try:
            dest = cache_path / "trajectories.json"
            logger.info("Downloading SWE-bench data from %s", url)
            urllib.request.urlretrieve(url, str(dest))  # noqa: S310
            if dest.exists() and dest.stat().st_size > 100:
                logger.info("Downloaded SWE-bench data to %s", dest)
                return str(dest)
            dest.unlink(missing_ok=True)
        except Exception:
            logger.debug("Download failed from %s", url, exc_info=True)

    raise FileNotFoundError(_MANUAL_INSTRUCTIONS)
