import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from enum import Enum

# imports from your project structure
from src.experiments.BaseExperiment import BaseExperiment
from src.config.path_config import *
from src.utils.data_loader_utils import load_images_from_csv

# import pytorch grad cam libraries
from pytorch_grad_cam import LayerCAM, GradCAM, HiResCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config.LOAD_MODE import LOAD_MODE
from torchvision.transforms import functional as F

class CAM_TYPE(Enum):
    LAYER = 'layer_cam'
    GRAD = 'grad_cam'
    HiRes = 'HiRes_cam'
    EigenGrad = 'EigenGrad_cam'


class CamBaseService:
    def __init__(self, experiment: BaseExperiment, model_name=None, cam_method=CAM_TYPE.LAYER.value, image_mode=LOAD_MODE.ONLY_FAKE.value):
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

    def save_cam_image(self, original_img, heatmap, filename, alpha=0.5):
        '''
        save cam images to self.cam_output_folder
        args:
            original_img: numpy array (h, w, 3), range [0, 255]
            heatmap: numpy array (h, w), range [0, 1]
            filename: output filename string
            alpha: transparency factor
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
        args:
            input_tensor: torch tensor (1, c, h, w)
            target_layers: list of layer objects
            targets: list of ClassifierOutputTarget objects (optional)
        returns:
            grayscale_cam: numpy array (h, w) range [0, 1]
        '''
        # initialize layer cam
        # construct the cam object once or per call depending on requirements
        # here we instantiate per call for simplicity
        with LayerCAM(model=self.experiment.model, target_layers=target_layers) as cam:
            # generate cam
            # if targets is none, it uses the category with highest score
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # returns shape (batch, h, w), take the first one
            return grayscale_cam[0, :]

    def load_model(self):
        '''
        load model architecture and weights
        '''
        print(f"loading model from {self.model_path}")

        # 1. create model architecture using the experiment class
        # assuming binary classification for rvf10k (real vs fake)
        self.experiment.create_model(num_classes=1)

        # 2. load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.experiment.model.load_state_dict(state_dict)
            self.experiment.model.to(self.device)
            self.experiment.model.eval()
            print("model loaded successfully.")
        else:
            raise FileNotFoundError(f"model file not found at {self.model_path}")

    def load_images(self):
        '''
        load images using the utility function with filtering support
        args:
            load_mode: LOAD_MODE enum (ALL, ONLY_FAKE, ONLY_REAL)
        '''
        print(f"loading images from csv with mode: {self.image_mode}...")
        dataset_path_dict = get_dataset_paths(self.dataset_name)
        valid_csv = dataset_path_dict['VALID_CSV']

        # determine target label based on load mode
        # assuming rvf10k convention: 0 is fake, 1 is real
        target_label = self.image_mode
        if self.image_mode == LOAD_MODE.ONLY_FAKE.value:
            target_label = 0
        elif self.image_mode == LOAD_MODE.ONLY_REAL.value:
            target_label = 1
        else:
            target_label = None  # or 'all'

        # load images using existing utility
        # returns numpy array (n, h, w, 3)
        images, valid_labels, valid_filenames = load_images_from_csv(
            csv_path=valid_csv,
            dataset_name=self.dataset_name,
            is_train=False,
            target_label=target_label
        )

        return images, valid_labels, valid_filenames

    def _get_target_layers(self):
        '''
        helper to determine target layers based on model architecture.
        you might need to adjust this based on the specific model (e.g. shufflenet vs mobilenet).
        '''
        model = self.experiment.model

        # example for shufflenet_v2 (usually conv5 is the last conv layer)
        if hasattr(model, 'conv5'):
            return [model.conv5]

        # example for mobilenetv3 (usually features[-1] or classifier predecessor)
        if hasattr(model, 'features'):
            return [model.features[-1]]

        # fallback: try to find the last conv2d layer
        target_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layers = [module]
        return target_layers

    def run(self):
        '''
        main coordination function to run deepfake face heat map generation
        '''
        # 1. load model
        self.load_model()

        # 2. get target layers for cam
        target_layers = self._get_target_layers()
        if not target_layers:
            print("error: could not find suitable target layers for cam.")
            return

        # 3. load images
        images, valid_labels, valid_filenames = self.load_images()

        # determine limit
        num_images_to_process = min(self.batch_limit, len(images))
        print(f"generating {self.cam_method} for first {num_images_to_process} images...")

        for i in range(num_images_to_process):
            try:
                original_img = images[i]  # shape (h, w, 3), 0-255, RGB
                filename = Path(valid_filenames[i]).stem

                ground_truth = int(valid_labels[i])

                # --- PREPROCESSING START ---
                # 1. Convert to Tensor [0, 1] (C, H, W)
                input_tensor = torch.from_numpy(original_img).permute(2, 0, 1).float() / 255.0

                # 2. Apply Normalization (CRITICAL STEP if model was trained with it)
                # Standard ImageNet normalization values
                # If your training code didn't use this, remove this step.
                # But usually, shufflenet_v2_x1_0(weights="DEFAULT") expects this.
                input_tensor = F.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                # 3. Add batch dimension
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                # --- PREPROCESSING END ---

                # 4. Get model prediction
                with torch.no_grad():
                    outputs = self.experiment.model(input_tensor)
                    probs = torch.sigmoid(outputs)

                    # Probability of class 1 (Fake)
                    prob_fake = probs.item()

                    # Determine label (Threshold 0.5)
                    predicted_label = 1 if prob_fake > 0.5 else 0

                    # Confidence is the probability of the *predicted* class
                    display_confidence = prob_fake if predicted_label == 1 else (1 - prob_fake)

                # 5. Generate heatmap
                heatmap = None
                if self.cam_method == CAM_TYPE.LAYER.value:
                    heatmap = self.make_layercam_heatmap(input_tensor, target_layers)

                # 6. Save result
                if heatmap is not None:
                    # format: {method}_{filename}_gt{gt}_pred{pred}_conf{conf}.jpg
                    output_filename = f"{self.cam_method}_{filename}_gt{ground_truth}_pred{predicted_label}_conf{display_confidence:.2f}.jpg"

                    # Note: We pass original_img (0-255 numpy) for visualization,
                    # but we used normalized input_tensor for model inference.
                    self.save_cam_image(original_img, heatmap, output_filename)
                    print(f"saved {output_filename}")

            except Exception as e:
                print(f"error processing image {i}: {e}")
                import traceback
                traceback.print_exc()

        print("done.")