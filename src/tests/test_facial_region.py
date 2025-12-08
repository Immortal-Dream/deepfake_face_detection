import torch
import numpy as np

# Adjust imports based on your project structure
from src.services.CamBaseService import CamBaseService, CAM_TYPE
from src.services.FacialRegionService import FacialRegionService
from src.config.path_config import *
from config.LOAD_MODE import LOAD_MODE

# Import your experiment classes
from src.experiments.XceptionPyTorchExperiment import XceptionPyTorchExperiment
from src.experiments.baselines.BlockShuffleLearning.models.xception import XceptionBSL
from src.experiments.baselines.ShuffleNetV2.ShuffleNetV2Experiment import ShuffleNetV2Experiment


def run_model_analysis(
        experiment_class,
        model_prefix: str,
        dataset_name: str,
        min_threshold: float = 0.5,
        threshold_quantile: float = 0.7,
        batch_limit: int = 100,
        save_images: bool = True
):
    """Run full analysis pipeline with optional image saving."""
    print(f"\n{'=' * 60}")
    print(f">>> ANALYSIS: {experiment_class.__name__} on {dataset_name}")
    print(f"{'=' * 60}")

    model_filename = f"{model_prefix}{dataset_name}.pth"
    model_path = MODEL_FOLDER / model_filename

    if not model_path.exists():
        print(f"❌ MODEL NOT FOUND: {model_path}")
        return

    # Setup experiment
    config = {"dataset_name": dataset_name, "epochs": 1, "batch_size": 10, "image_size": 224}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if experiment_class == XceptionPyTorchExperiment:
        model = XceptionBSL(num_class=1, is_train=False, is_bs_adv=True, is_rs_adv=True).eval().to(device)
        experiment = experiment_class(config=config, model=model, device=device)
    else:
        experiment = experiment_class(config)

    experiment.dataset_name = dataset_name

    # Initialize services
    cam_service = CamBaseService(
        experiment=experiment,
        model_name=model_filename,
        cam_method=CAM_TYPE.LAYER.value,
        image_mode=LOAD_MODE.ONLY_FAKE.value,
        save_raw_heatmap=True,
        debug_mode=True
    )
    cam_service.batch_limit = batch_limit

    facial_service = FacialRegionService(
        min_threshold=min_threshold,
        threshold_quantile=threshold_quantile,
        experiment_name=experiment.experiment_name,
        dataset_name=dataset_name
    )

    # Load data
    print("\n" + "-" * 40)
    print("LOADING DATA")
    print("-" * 40)
    cam_service.load_model()
    cam_service.update_paths()
    images, labels, filenames = cam_service.load_images()

    if len(images) == 0:
        print("❌ No images loaded. Aborting.")
        return

    num_images = min(len(images), batch_limit)
    print(f"Processing {num_images} images...\n")

    # Process images
    success_count = 0
    for i in range(num_images):
        try:
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_images}")

            image_rgb = images[i]
            filename = filenames[i]
            true_label = int(labels[i])

            # Generate heatmap
            heatmap, pred_info = cam_service.process_single_image(image_rgb)

            if heatmap is None:
                print(f"⚠️  Skipping {filename}: heatmap generation returned None")
                continue

            # SAVE IMAGES
            if save_images:
                base_name = f"{cam_service.cam_method}_{Path(filename).stem}_gt{true_label}_pred{pred_info['predicted_label']}"

                # Save overlaid image
                overlay_filename = f"{base_name}.jpg"
                cam_service.save_cam_image(image_rgb, heatmap, overlay_filename)

            # Analyze facial regions
            facial_service.add_result(
                filename=filename,
                heatmap=heatmap,
                image=image_rgb,
                prediction=pred_info['predicted_label'],
                true_label=true_label,
                confidence=pred_info['confidence']
            )

            success_count += 1

        except Exception as e:
            print(f"❌ Error processing image {i} ({filename}): {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print("\n" + "-" * 40)
    print("SAVING RESULTS")
    print("-" * 40)
    csv_path = facial_service.save_results_to_csv()
    print(f"✅ CSV saved to: {csv_path}")
    print(f"✅ Total images processed successfully: {success_count}/{num_images}")
    print(f"✅ Images saved in: {cam_service.cam_output_folder}")
    if cam_service.save_raw_heatmap:
        print(f"✅ Heatmaps saved in: {cam_service.heatmap_output_folder}")



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
                min_threshold=0.5,
                threshold_quantile=0.8,
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
                min_threshold=0.5,
                threshold_quantile=0.8,
                batch_limit=100
            )
