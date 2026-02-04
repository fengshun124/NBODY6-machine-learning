import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


OUTPUT_BASE_ENV = os.getenv("OUTPUT_BASE")
if OUTPUT_BASE_ENV is None:
    raise EnvironmentError("OUTPUT_BASE environment variable is not set.")
OUTPUT_BASE = Path(OUTPUT_BASE_ENV)


def setup_logger(log_file: Path | str) -> None:
    log_file = Path(log_file)

    # avoid adding multiple handlers if already set up
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return

    handlers = [logging.StreamHandler()]
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                filename=str(log_file),
                mode="a",
                maxBytes=5_000_000,
                backupCount=5,
            )
        )
    except Exception as e:
        print(
            f"Failed to create log file handler for {log_file}: {e!r}, using stream handler only."
        )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
