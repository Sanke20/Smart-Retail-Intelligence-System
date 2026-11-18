
import os
import time
import logging

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

LOG_DIR = BASE_DIR / "logs"

MODEL_DIR = BASE_DIR / "artifacts" / "models"

VISUAL_DIR = BASE_DIR / "artifacts" / "visuals"

DATA_RAW_DIR = BASE_DIR / "artifacts" / "data" / "raw"

DATA_PROCESSED_DIR = BASE_DIR / "artifacts" / "data" / "processed"


# Create required directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


# Logging configuration
logging.Formatter.converter = time.localtime

logger = logging.getLogger("SmartRetailPipeline")

logger.setLevel(logging.INFO)

logger.propagate = False

if not logger.handlers:

    file_handler = logging.FileHandler(
        LOG_DIR / "smart_retail_intelligence.log",
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)
  