import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from enum import Enum
from torchvision.transforms import functional as F

from src.experiments.BaseExperiment import BaseExperiment
from src.config.path_config import *
from src.utils.data_loader_utils import load_images_from_csv
from pytorch_grad_cam import LayerCAM
from config.LOAD_MODE import LOAD_MODE


class CAM_TYPE(Enum):
    LAYER = 'layer_cam'
    GRAD = 'grad_cam'
    HiRes = 'HiRes_cam'
    EigenGrad = 'EigenGrad_cam'


class ModelOutputWrapper(nn.Module):
    """Wrapper to handle different model output formats."""

    def __init__(self, model):
        super(ModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            if 'out' in output:
                return output['out']
            elif 'logits' in output:
                return output['logits']
            return list(output.values())[0]
        return output


class CamBaseService:
    def __init__(self, experiment: BaseExperiment = None, model_name=None,
                 cam_method=CAM_TYPE.LAYER.value, image_mode=LOAD_MODE.ONLY_FAKE.value,
                 save_raw_heatmap=False, debug_mode=True):
        self.experiment = experiment
        self.experiment_name = experiment.experiment_name
        self.dataset_name = experiment.dataset_name
        self.model_name = model_name or f"{self.dataset_name}_{self.experiment_name}_model.pth"
        self.model_path = MODEL_FOLDER / self.model_name
        self.cam_method = cam_method
        self.batch_limit = 10
        self.image_mode = image_mode
        self.save_raw_heatmap = save_raw_heatmap
        self.debug_mode = debug_mode

        # Output directories
        self.cam_output_folder = OUTPUT_FOLDER / self.experiment_name / self.dataset_name / self.cam_method
        self.heatmap_output_folder = OUTPUT_FOLDER / self.experiment_name / self.dataset_name / f"{self.cam_method}_heatmaps"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()

        # Cache
        self.target_layers = None
        self.num_classes = 1

    def _setup_directories(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.cam_output_folder, exist_ok=True)
        if self.debug_mode:
            print(f"Created/verified directory: {self.cam_output_folder}")

        if self.save_raw_heatmap:
            os.makedirs(self.heatmap_output_folder, exist_ok=True)
            if self.debug_mode:
                print(f"Created/verified directory: {self.heatmap_output_folder}")

    def save_cam_image(self, original_img, heatmap, filename, alpha=0.5):
        """Save CAM overlaid image."""
        try:
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Convert original image to BGR for cv2
            original_img_bgr = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Superimpose
            superimposed_img = heatmap_colored * alpha + original_img_bgr * (1 - alpha)

            # Save
            save_path = self.cam_output_folder / filename
            cv2.imwrite(str(save_path), superimposed_img)

            if self.debug_mode:
                print(f"  💾 Saved overlaid image: {save_path}")

        except Exception as e:
            print(f"Error saving overlaid image {filename}: {e}")

    def save_raw_heatmap_image(self, heatmap, filename):
        """Save raw heatmap (grayscale and color versions)."""
        try:
            heatmap = np.clip(heatmap, 0, 1)
            heatmap_uint8 = np.uint8(255 * heatmap)

            # Save grayscale
            gray_filename = filename.replace('.jpg', '_gray.jpg')
            gray_path = self.heatmap_output_folder / gray_filename
            cv2.imwrite(str(gray_path), heatmap_uint8)

            # Save colored
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_path = self.heatmap_output_folder / filename
            cv2.imwrite(str(colored_path), heatmap_colored)

            if self.debug_mode:
                print(f"   💾 Saved raw heatmaps: {gray_path} | {colored_path}")

        except Exception as e:
            print(f"❌ Error saving raw heatmap {filename}: {e}")

    def make_layercam_heatmap(self, input_tensor, target_layers, targets=None):
        """Generate heatmap using LayerCAM."""
        wrapped_model = ModelOutputWrapper(self.experiment.model)

        with LayerCAM(model=wrapped_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # grayscale_cam shape: [batch, height, width]
            return grayscale_cam[0, :] if grayscale_cam is not None else None

    def load_model(self):
        """Load model architecture and weights."""
        print(f"Loading model from {self.model_path}")

        if not os.path.exists(self.model_path):
            print(f"❌ CRITICAL: Model file not found at {self.model_path}")
            return

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract state_dict from checkpoint
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        if isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()

        # Auto-detect number of classes
        self.num_classes = self._detect_num_classes(state_dict)

        # Create model architecture
        if self.experiment.model is None:
            self.experiment.model = self.experiment.create_model(num_classes=self.num_classes)

        # Load weights
        try:
            self.experiment.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"⚠️ Strict loading failed: {e}. Retrying with strict=False...")
            self.experiment.model.load_state_dict(state_dict, strict=False)

        self.experiment.model.to(self.device)
        self.experiment.model.eval()
        print("✅ Model loaded successfully")

        # Cache target layers
        self.target_layers = self._get_target_layers()
        if not self.target_layers:
            print("⚠️ WARNING: Could not find suitable target layers for CAM")

    def _detect_num_classes(self, state_dict):
        """Detect number of classes from state_dict."""
        if 'fc.weight' in state_dict:
            return state_dict['fc.weight'].shape[0]
        elif 'classifier.weight' in state_dict:
            return state_dict['classifier.weight'].shape[0]
        elif 'last_linear.weight' in state_dict:
            return state_dict['last_linear.weight'].shape[0]
        return 1

    def load_images(self):
        """Load images from CSV."""
        print(f"Loading images with mode: {self.image_mode}")
        dataset_path_dict = get_dataset_paths(self.dataset_name)
        valid_csv = dataset_path_dict['VALID_CSV']

        if self.debug_mode:
            print(f"   CSV path: {valid_csv}")

        target_label = {LOAD_MODE.ONLY_FAKE.value: 0, LOAD_MODE.ONLY_REAL.value: 1}.get(self.image_mode, None)

        images, labels, filenames = load_images_from_csv(
            csv_path=valid_csv,
            dataset_name=self.dataset_name,
            is_train=False,
            target_label=target_label
        )

        print(
            f"✅ Loaded {len(images)} images ({len(labels)} fake, {len(labels) - len([l for l in labels if l == 0])} real)")
        return images, labels, filenames

    def _get_target_layers(self):
        """Determine target layers for CAM extraction."""
        model = self.experiment.model

        # XceptionBSL
        if hasattr(model, 'xception'):
            backbone = model.xception
            if hasattr(backbone, 'block11'):
                print("   🎯 Targeting Xception block11 (14x14 resolution)")
                return [backbone.block11]
            elif hasattr(backbone, 'block12'):
                print("   🎯 Targeting Xception block12 (7x7 resolution)")
                return [backbone.block12]

        # ShuffleNetV2
        if hasattr(model, 'stage3'):
            print("   🎯 Targeting ShuffleNetV2 stage3 (14x14 resolution)")
            return [model.stage3[-1]]

        # MobileNet
        if hasattr(model, 'features'):
            return [model.features[-1]]

        print("❌ ERROR: No suitable target layer found")
        return []

    def process_single_image(self, original_img):
        """Process a single image and return heatmap and prediction info."""
        if self.experiment.model is None:
            self.load_model()

        if self.target_layers is None:
            self.target_layers = self._get_target_layers()

        # Preprocess
        input_tensor = torch.from_numpy(original_img).permute(2, 0, 1).float() / 255.0
        input_tensor = F.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.experiment.model(input_tensor)
            if isinstance(outputs, dict):
                logits = outputs.get('out') or outputs.get('logits') or list(outputs.values())[0]
            else:
                logits = outputs

            prob_fake = torch.sigmoid(logits).item() if self.num_classes == 1 else torch.softmax(logits, dim=1)[
                0, 1].item()
            predicted_label = 1 if prob_fake > 0.5 else 0
            confidence = prob_fake if predicted_label == 1 else (1 - prob_fake)

        # Generate heatmap
        heatmap = self.make_layercam_heatmap(input_tensor,
                                             self.target_layers) if self.cam_method == CAM_TYPE.LAYER.value else None

        if self.debug_mode and heatmap is not None:
            print(f"   📊 Heatmap stats: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")

        return heatmap, {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'prob_fake': prob_fake
        }

        return heatmap, prediction_info

    def run(self):
        '''
        main coordination function
        '''
        self.load_model()

        if not self.target_layers:
            print("error: could not find suitable target layers for cam.")
            return

        images, valid_labels, valid_filenames = self.load_images()

        num_images_to_process = min(self.batch_limit, len(images))
        print(f"generating {self.cam_method} for first {num_images_to_process} images...")

        for i in range(num_images_to_process):
            try:
                original_img = images[i]
                filename = Path(valid_filenames[i]).stem
                ground_truth = int(valid_labels[i])

                heatmap, pred_info = self.process_single_image(original_img)

                predicted_label = pred_info['predicted_label']
                display_confidence = pred_info['confidence']

                if heatmap is not None:
                    output_filename = f"{self.cam_method}_{filename}_gt{ground_truth}_pred{predicted_label}.jpg"
                    self.save_cam_image(original_img, heatmap, output_filename)
                    print(f"saved {output_filename}")

            except Exception as e:
                print(f"error processing image {i}: {e}")
                import traceback
                traceback.print_exc()

        print("done.")

    def update_paths(self):
        """
        Internal helper to re-calculate paths and folders based on current state.
        Should be called whenever experiment, dataset_name, or model_name changes.
        """
        # Update experiment name if experiment object exists
        if self.experiment:
            self.experiment_name = self.experiment.experiment_name
            # Also sync dataset name from experiment if not explicitly overridden elsewhere
            # (Optional: depends on your logic, usually experiment holds the truth)
            if hasattr(self.experiment, 'dataset_name'):
                self.dataset_name = self.experiment.dataset_name

        # Recalculate model path
        # If model_name wasn't manually fixed, regenerate it based on new names
        if not self.model_name or (self.dataset_name and self.experiment_name in self.model_name):
            self.model_name = f"{self.dataset_name}_{self.experiment_name}_model.pth"

        self.model_path = MODEL_FOLDER / self.model_name

        # Recalculate output folder
        self.cam_output_folder = OUTPUT_FOLDER / self.experiment_name / self.dataset_name / self.cam_method

        # Ensure directory exists
        if not os.path.exists(self.cam_output_folder):
            os.makedirs(self.cam_output_folder)
            print(f"Created updated output directory: {self.cam_output_folder}")

        # Reset caches because model/data changed
        self.target_layers = None
        # self.num_classes might need reset too, but usually happens in load_model
