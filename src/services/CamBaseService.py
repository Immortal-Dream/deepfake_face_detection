import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from enum import Enum
from torchvision.transforms import functional as F

# imports from your project structure
from src.experiments.BaseExperiment import BaseExperiment
from src.config.path_config import *
from src.utils.data_loader_utils import load_images_from_csv

# import pytorch grad cam libraries
from pytorch_grad_cam import LayerCAM, GradCAM, HiResCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config.LOAD_MODE import LOAD_MODE


class CAM_TYPE(Enum):
    LAYER = 'layer_cam'
    GRAD = 'grad_cam'
    HiRes = 'HiRes_cam'
    EigenGrad = 'EigenGrad_cam'



class ModelOutputWrapper(nn.Module):
    def __init__(self, model):
        super(ModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            # Try common keys for logits
            if 'out' in output:
                return output['out']
            elif 'logits' in output:
                return output['logits']
            else:
                # Fallback: return the first value
                return list(output.values())[0]
        return output


# --------------------------------------------------

class CamBaseService:
    def __init__(self, experiment: BaseExperiment, model_name=None, cam_method=CAM_TYPE.LAYER.value,
                 image_mode=LOAD_MODE.ONLY_FAKE.value):
        self.experiment = experiment
        self.experiment_name = experiment.experiment_name
        self.dataset_name = experiment.dataset_name

        # determine model path
        self.model_name = model_name if model_name else f"{self.dataset_name}_{self.experiment_name}_model.pth"
        self.model_path = MODEL_FOLDER / self.model_name

        # set the method here (layer or grad)
        self.cam_method = cam_method
        # set how many images to process (batch control)
        self.batch_limit = 10
        # 0 -> only fake images
        # 1 -> only real images
        # None -> load all images
        self.image_mode = image_mode

        self.cam_output_folder = OUTPUT_FOLDER / self.experiment_name / self.dataset_name / self.cam_method

        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.cam_output_folder):
            os.makedirs(self.cam_output_folder)
            print(f"created output directory: {self.cam_output_folder}")

        # cache
        self.target_layers = None
        self.num_classes = 1  # default, will be updated in load_model

    def save_cam_image(self, original_img, heatmap, filename, alpha=0.5):
        '''
        save cam images to self.cam_output_folder
        '''
        try:
            # resize heatmap to match original image size
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

            # rescale heatmap to 0-255
            heatmap_uint8 = np.uint8(255 * heatmap)

            # apply jet colormap
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # convert original image to rgb if it's not already (cv2 uses bgr by default)
            # assuming original_img passed in is rgb from data loader
            original_img_bgr = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

            # superimpose
            superimposed_img = heatmap_colored * alpha + original_img_bgr * (1 - alpha)

            # save image
            save_path = self.cam_output_folder / filename
            cv2.imwrite(str(save_path), superimposed_img)

        except Exception as e:
            print(f"error saving image {filename}: {e}")

    def make_layercam_heatmap(self, input_tensor, target_layers, targets=None):
        '''
        use pytorch layer cam to generate the heat map
        '''
        wrapped_model = ModelOutputWrapper(self.experiment.model)

        with LayerCAM(model=wrapped_model, target_layers=target_layers) as cam:
            # generate cam
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0, :]

    def load_model(self):
        '''
        load model architecture and weights intelligently.
        '''
        print(f"loading model from {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model file not found at {self.model_path}")

        # 1. load the checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Handle case where checkpoint is not a direct state_dict but a dict containing it
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        elif isinstance(checkpoint, nn.Module):
            print("Warning: Loaded full model object instead of state_dict. Using its state_dict.")
            state_dict = checkpoint.state_dict()
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

        # 2. auto-detect number of classes based on final layer weights
        detected_num_classes = 1  # fallback default

        if 'fc.weight' in state_dict:
            detected_num_classes = state_dict['fc.weight'].shape[0]
            print(f"detected num_classes from 'fc.weight': {detected_num_classes}")
        elif 'classifier.weight' in state_dict:
            detected_num_classes = state_dict['classifier.weight'].shape[0]
            print(f"detected num_classes from 'classifier.weight': {detected_num_classes}")
        elif 'last_linear.weight' in state_dict:
            detected_num_classes = state_dict['last_linear.weight'].shape[0]
            print(f"detected num_classes from 'last_linear.weight': {detected_num_classes}")
        else:
            keys = list(state_dict.keys())
            weight_keys = [k for k in keys if 'weight' in k and state_dict[k].ndim > 1]
            if weight_keys:
                last_weight_key = weight_keys[-1]
                detected_num_classes = state_dict[last_weight_key].shape[0]
                print(f"inferred num_classes from '{last_weight_key}': {detected_num_classes}")
            else:
                print("Warning: Could not infer num_classes. Using default: 1")

        self.num_classes = detected_num_classes

        # 3. create model architecture
        model_instance = self.experiment.create_model(num_classes=self.num_classes)

        if self.experiment.model is None:
            if model_instance is not None:
                self.experiment.model = model_instance
            else:
                raise ValueError(
                    f"Experiment {type(self.experiment).__name__}.create_model() returned None and did not set self.model.")

        # 4. load weights
        try:
            self.experiment.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Retrying with strict=False...")
            self.experiment.model.load_state_dict(state_dict, strict=False)

        self.experiment.model.to(self.device)
        self.experiment.model.eval()
        print("model loaded successfully.")

        # cache target layers
        self.target_layers = self._get_target_layers()
        if not self.target_layers:
            print("warning: could not find suitable target layers for cam.")

    def load_images(self):
        '''
        load images using the utility function
        '''
        print(f"loading images from csv with mode: {self.image_mode}...")
        dataset_path_dict = get_dataset_paths(self.dataset_name)
        valid_csv = dataset_path_dict['VALID_CSV']

        target_label = self.image_mode
        if self.image_mode == LOAD_MODE.ONLY_FAKE.value:
            target_label = 0
        elif self.image_mode == LOAD_MODE.ONLY_REAL.value:
            target_label = 1
        else:
            target_label = None

        images, valid_labels, valid_filenames = load_images_from_csv(
            csv_path=valid_csv,
            dataset_name=self.dataset_name,
            is_train=False,
            target_label=target_label
        )

        return images, valid_labels, valid_filenames

    def _get_target_layers(self):
        '''
        helper to determine target layers
        '''
        model = self.experiment.model

        # 1. Check for XceptionBSL
        if hasattr(model, 'xception'):
            backbone = model.xception
            if hasattr(backbone, 'conv4'):
                return [backbone.conv4]
            elif hasattr(backbone, 'block12'):
                return [backbone.block12]
            else:
                last_conv = None
                for module in backbone.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        last_conv = module
                if last_conv:
                    return [last_conv]

        # 2. Check for ShuffleNetV2
        if hasattr(model, 'conv5'):
            return [model.conv5]

        # 3. Check for MobileNet
        if hasattr(model, 'features'):
            if isinstance(model.features, nn.Module):
                return [model.features[-1]]

        # 4. Generic Fallback
        target_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layers = [module]

        if target_layers:
            return target_layers

        print("Error: Could not determine target layer automatically.")
        return []

    def process_single_image(self, original_img):
        '''
        process a single image array
        '''
        if self.experiment.model is None:
            self.load_model()

        if self.target_layers is None:
            self.target_layers = self._get_target_layers()

        # --- preprocessing ---
        input_tensor = torch.from_numpy(original_img).permute(2, 0, 1).float() / 255.0
        input_tensor = F.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # 4. get model prediction
        with torch.no_grad():
            outputs = self.experiment.model(input_tensor)

            if isinstance(outputs, dict):
                if 'out' in outputs:
                    logits = outputs['out']
                elif 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = list(outputs.values())[0]
            else:
                logits = outputs
            # ----------------------------------------------------

            prob_fake = 0.0
            predicted_label = 0

            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
                prob_fake = probs.item()
                predicted_label = 1 if prob_fake > 0.5 else 0
            else:
                probs = torch.softmax(logits, dim=1)
                prob_fake = probs[0][1].item()
                predicted_label = torch.argmax(probs, dim=1).item()

            confidence = prob_fake if predicted_label == 1 else (1 - prob_fake)

        # 5. generate heatmap
        heatmap = None
        if self.cam_method == CAM_TYPE.LAYER.value:
            heatmap = self.make_layercam_heatmap(input_tensor, self.target_layers)

        prediction_info = {
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
                    output_filename = f"{self.cam_method}_{filename}_gt{ground_truth}_pred{predicted_label}_conf{display_confidence:.2f}.jpg"
                    self.save_cam_image(original_img, heatmap, output_filename)
                    print(f"saved {output_filename}")

            except Exception as e:
                print(f"error processing image {i}: {e}")
                import traceback
                traceback.print_exc()

        print("done.")