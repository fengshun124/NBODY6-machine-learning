import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


OUTPUT_BASE = Path(os.getenv("OUTPUT_BASE")).resolve()


def setup_logger(log_file: Path | str) -> None:
    log_file = Path(log_file)
    handlers = [logging.StreamHandler()]
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                filename=str(log_file),
                mode=5_000_000,
                maxBytes=3,
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
