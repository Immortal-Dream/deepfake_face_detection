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


class CAM_TYPE(Enum):
    LAYER = 'layer_cam'
    GRAD = 'grad_cam'
    HiRes = 'HiRes_cam'
    EigenGrad = 'EigenGrad_cam'


class CamBaseService:
    def __init__(self, experiment: BaseExperiment, model_name=None, cam_method=CAM_TYPE.LAYER.value):
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
        load images using the utility function
        '''
        print("loading images from csv...")
        dataset_path_dict = get_dataset_paths(self.dataset_name)
        valid_csv = dataset_path_dict['VALID_CSV']

        # load images using existing utility
        # returns numpy array (n, h, w, 3)
        images, valid_labels, valid_filenames = load_images_from_csv(valid_csv, self.dataset_name, False)
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
                original_img = images[i]  # shape (h, w, 3), 0-255
                filename = Path(valid_filenames[i]).stem

                # preprocess for pytorch:
                # 1. normalize to [0, 1]
                # 2. transpose to (c, h, w)
                # 3. add batch dimension
                input_tensor = torch.from_numpy(original_img).float() / 255.0
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

                # generate heatmap
                heatmap = None
                if self.cam_method == CAM_TYPE.LAYER.value:
                    heatmap = self.make_layercam_heatmap(input_tensor, target_layers)

                # save result
                if heatmap is not None:
                    output_filename = f"{self.cam_method}_{i}_{filename}.jpg"
                    self.save_cam_image(original_img, heatmap, output_filename)
                    print(f"saved {output_filename}")

            except Exception as e:
                print(f"error processing image {i}: {e}")
                import traceback
                traceback.print_exc()

        print("done.")