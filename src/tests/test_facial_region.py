import torch
import numpy as np
from pathlib import Path

# Adjust imports based on your project structure
from src.services.CamBaseService import CamBaseService, CAM_TYPE
from src.services.FacialRegionService import FacialRegionService
from src.config.path_config import MODEL_FOLDER, DATASET_LIST, OUTPUT_FOLDER, rvf10k
from config.LOAD_MODE import LOAD_MODE

# Import your experiment classes
from src.experiments.XceptionPyTorchExperiment import XceptionPyTorchExperiment
from src.experiments.baselines.BlockShuffleLearning.models.xception import XceptionBSL
from src.experiments.baselines.ShuffleNetV2.ShuffleNetV2Experiment import ShuffleNetV2Experiment


def run_analysis_pipeline(experiment, model_filename, dataset_name, threshold=0.3, batch_limit=50):
    """
    Helper function to run the full analysis pipeline for a single experiment/dataset.
    """
    print(f"\n========================================================")
    print(f"Running analysis for: {dataset_name} | Model: {model_filename}")
    print(f"========================================================")

    # 1. Initialize Services
    # CamBaseService handles model loading, image loading, and CAM generation
    cam_service = CamBaseService(
        experiment=experiment,
        model_name=model_filename,
        cam_method=CAM_TYPE.LAYER.value,
        image_mode=LOAD_MODE.ONLY_FAKE.value  # or ALL, depending on your needs
    )

    if batch_limit:
        cam_service.batch_limit = batch_limit

    # FacialRegionService handles landmark detection and region attention stats
    facial_service = FacialRegionService(
        threshold=threshold,
        experiment_name=experiment.experiment_name,
        dataset_name=dataset_name
    )

    # 2. Load Resources
    # This loads the model weights and the list of images to process
    cam_service.load_model()
    images, labels, filenames = cam_service.load_images()

    num_images = min(len(images), cam_service.batch_limit)
    print(f"Processing {num_images} images...")

    # 3. Main Processing Loop
    for i in range(num_images):
        try:
            image_rgb = images[i]
            filename = filenames[i]
            true_label = int(labels[i])

            # A. Generate Heatmap & Get Prediction
            # process_single_image does normalization, inference, and cam generation
            heatmap, pred_info = cam_service.process_single_image(image_rgb)

            if heatmap is None:
                print(f"Skipping {filename}: heatmap generation failed")
                continue

            # B. Analyze Facial Regions
            # This function detects landmarks on the original image, overlays the heatmap,
            # and determines which regions (eyes, nose, etc.) are 'attended' to.
            # It internally adds the result row to facial_service.accumulated_result.
            facial_service.add_result(
                filename=filename,
                heatmap=heatmap,
                image=image_rgb,  # Needed for landmark detection
                prediction=pred_info['predicted_label'],
                true_label=true_label,
                confidence=pred_info['confidence']
            )

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_images}...")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Save Final CSV Report
    csv_name = f"analysis_{dataset_name}_{experiment.experiment_name}.csv"
    saved_path = facial_service.save_results_to_csv(filename=csv_name)
    print(f"Analysis complete. Results saved to: {saved_path}")


def test_facial_region():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configuration ---
    # You can loop through datasets or pick specific ones
    target_datasets = DATASET_LIST  # e.g. ['dalle2', 'midjourney', ...]

    # 1. Run Analysis for Xception (BSL)
    print("\n>>> Starting Xception Analysis <<<")
    for dataset_name in target_datasets:
        # Check if model file exists first to avoid crashes
        model_filename = f'xception_BSL_{dataset_name}.pth'
        if not (MODEL_FOLDER / model_filename).exists():
            print(f"Skipping {dataset_name}: Model file {model_filename} not found.")
            continue

        # Setup Experiment Object
        config = {
            "dataset_name": dataset_name,
            'epochs': 1, 'batch_size': 10, 'image_size': 224
        }
        # Initialize model architecture
        model = XceptionBSL(num_class=1, is_train=False, is_bs_adv=True, is_rs_adv=True).eval().to(device)
        experiment = XceptionPyTorchExperiment(config=config, model=model, device=device)
        experiment.dataset_name = dataset_name

        # Run Pipeline
        run_analysis_pipeline(
            experiment=experiment,
            model_filename=model_filename,
            dataset_name=dataset_name,
            threshold=0.3,  # Adjust attention threshold as needed
            batch_limit=50
        )

    # 2. Run Analysis for ShuffleNetV2 (Example for specific dataset 'rvf10k')
    print("\n>>> Starting ShuffleNetV2 Analysis <<<")
    shuffle_dataset = "rvf10k"
    shuffle_model_name = "rvf10k_ShuffleNetV2_baseline_model.pth"

    if (MODEL_FOLDER / shuffle_model_name).exists():
        config = {'epochs': 1, 'batch_size': 10, 'image_size': 224}
        experiment = ShuffleNetV2Experiment(config)
        experiment.dataset_name = shuffle_dataset

        run_analysis_pipeline(
            experiment=experiment,
            model_filename=shuffle_model_name,
            dataset_name=shuffle_dataset,
            threshold=0.3,
            batch_limit=50
        )
    else:
        print(f"Skipping ShuffleNet: {shuffle_model_name} not found.")
