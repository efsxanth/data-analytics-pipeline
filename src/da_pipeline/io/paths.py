"""Where a run's results go, and where its log goes.

The output directory is created beside the config file that described the run,
not beside the code and not in the working directory. That keeps results next
to the data they came from -- which for this pipeline lives outside the
repository -- and means two configs never write into the same place.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import PipelineConfig
from ..constants import outputs as out


def create_output_dir(config: PipelineConfig) -> Path:
    """Create and return this run's output directory.

    The directory is ``<config file's directory>/Data_Analytics[_<suffix>]``.
    Existing directories are reused, so re-running a config overwrites its own
    results rather than accumulating copies.
    """

    output_dir = config.config_path.resolve().parent / config.output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def setup_logging(output_dir: Path | str) -> None:
    """Send log records to both the console and a file in the output directory.

    Existing handlers are cleared first: a second run in the same process --
    a test, or a notebook -- would otherwise keep writing to the previous run's
    log file as well as its own.
    """

    log_file_path = Path(output_dir) / out.LOG_FNAME

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            # 'w': one file per run, overwriting the previous run's log.
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
