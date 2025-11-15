import sys
from src.utils.csv_utils import *
from src.config.path_config import OUTPUT_FOLDER, TRAIN_CSV, VALID_CSV


def test_simplify_csv():
    csv_path1 = OUTPUT_FOLDER / "mock_output.csv"
    csv_path2 = VALID_CSV
    # simplify_csv(csv_path1)
    simplify_csv(csv_path2)
