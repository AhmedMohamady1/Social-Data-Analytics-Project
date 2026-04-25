import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_3_ROOT = PROJECT_ROOT.parent / "Task_3"

CLEANING_SCRIPT_PATH = TASK_3_ROOT / "src" / "cleaning_pipeline.py"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

# Constants
GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"
