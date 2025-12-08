from src.utils.data_analysis_utils import *


def test_data_analysis_1():
    for dataset in DATASET_LIST:
        model_sensitivity_analysis(Xception_BSL_baseline, dataset)


def test_compare():
    for dataset in DATASET_LIST:
        print(f"\n\nExperimenting with dataset: {dataset}")
        compare_models_difference(ViT_baseline, Xception_BSL_baseline, dataset)
