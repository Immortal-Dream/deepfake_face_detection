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
    cam_service = CamBaseService(
        experiment=experiment,
        model_name=model_filename,
        cam_method=CAM_TYPE.LAYER.value,
        image_mode=LOAD_MODE.ONLY_FAKE.value
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

    # Safety check if no images loaded
    if len(images) == 0:
        print(f"No images found for {dataset_name}. Skipping.")
        return

    num_images = min(len(images), cam_service.batch_limit)
    print(f"Processing {num_images} images...")

    # 3. Main Processing Loop
    for i in range(num_images):
        try:
            image_rgb = images[i]
            filename = filenames[i]
            true_label = int(labels[i])

            # A. Generate Heatmap & Get Prediction
            heatmap, pred_info = cam_service.process_single_image(image_rgb)

            if heatmap is None:
                print(f"Skipping {filename}: heatmap generation failed")
                continue

            # B. Analyze Facial Regions
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
    # Note: save_results_to_csv inside FacialRegionService usually needs a filename arg
    # or defaults to "layercam_analysis.csv". We make it unique here.
    csv_name = f"analysis_{dataset_name}_{experiment.experiment_name}.csv"
    saved_path = facial_service.save_results_to_csv(filename=csv_name)
    print(f"Analysis complete. Results saved to: {saved_path}")


def run_model_analysis(
        experiment_class,
        model_prefix: str,
        dataset_name: str,
        threshold: float = 0.5,
        batch_limit: int = 100
):
    """
    Generic analysis runner for different models.
    Handles specific initialization logic for Xception vs ShuffleNet.
    """
    print(f"\n>>> Starting {experiment_class.__name__} Analysis for {dataset_name} <<<")

    # Construct model filename
    model_filename = f"{model_prefix}{dataset_name}.pth"

    # Check if model file exists
    if not (MODEL_FOLDER / model_filename).exists():
        print(f"Skipping {experiment_class.__name__}: {model_filename} not found.")
        return

    # Define experiment configuration
    config = {
        "dataset_name": dataset_name,  # Critical for path configs inside experiment
        "epochs": 1,
        "batch_size": 10,
        "image_size": 224
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- FIX: Handle different class initializations ---
    if experiment_class == XceptionPyTorchExperiment:
        # Xception needs model and device in init
        # We initialize a base model structure here, load_weights will happen inside CamBaseService later
        model = XceptionBSL(num_class=1, is_train=False, is_bs_adv=True, is_rs_adv=True).eval().to(device)
        experiment = experiment_class(config=config, model=model, device=device)
    else:
        # Standard initialization (e.g. ShuffleNetV2Experiment)
        experiment = experiment_class(config)

    experiment.dataset_name = dataset_name

    # Run analysis pipeline
    run_analysis_pipeline(
        experiment=experiment,
        model_filename=model_filename,
        dataset_name=dataset_name,
        threshold=threshold,
        batch_limit=batch_limit
    )


def test_facial_region():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Configuration ---
    target_datasets = DATASET_LIST

    run_xception = True
    run_shuffle = True

    # 1. Run Analysis for Xception (BSL)
    if run_xception:
        print("\n>>> Batch Running Xception Analysis <<<")
        for dataset_name in target_datasets:
            run_model_analysis(
                experiment_class=XceptionPyTorchExperiment,
                model_prefix="xception_BSL_",
                dataset_name=dataset_name,
                threshold=0.75,
                batch_limit=100
            )

    # 2. Run Analysis for ShuffleNetV2
    if run_shuffle:
        print("\n>>> Batch Running ShuffleNetV2 Analysis <<<")
        for dataset_name in target_datasets:
            run_model_analysis(
                experiment_class=ShuffleNetV2Experiment,
                model_prefix="ShuffleNetV2_baseline_",
                # Note: Check your actual file naming convention.
                # It might be f"{dataset_name}_ShuffleNetV2_baseline_model.pth" based on previous logs.
                # Adjust model_prefix logic if filenames vary by dataset position.
                dataset_name=dataset_name,
                threshold=0.7,
                batch_limit=100
            )
