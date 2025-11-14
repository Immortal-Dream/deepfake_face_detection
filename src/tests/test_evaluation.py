from src.evaluation.evaluate_correctness import *

def test_file_evaluation():
    predict_result = "mock_output.csv"
    evaluate_predictions(predict_result)
