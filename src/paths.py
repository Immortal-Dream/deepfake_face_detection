from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "output"
OVERALL_FOLDER = OUTPUT_FOLDER / "overall"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OVERALL_FOLDER.mkdir(parents=True, exist_ok=True)
