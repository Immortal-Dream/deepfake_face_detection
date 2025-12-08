import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import dlib
import cv2
import pandas as pd
import os
from pathlib import Path

from pytorch_grad_cam import GradCAM, EigenGradCAM, LayerCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config.path_config import OUTPUT_FOLDER, MODEL_FOLDER, get_dataset_paths, DATA_ROOT
from src.utils.csv_utils import create_dataset_csv
from src.utils.data_loader_utils import load_images_from_csv
from src.experiments.baselines.ViT.ViTBaselineExperiment import ViTBaselineExperiment

from typing import List, Callable

FACIAL_REGIONS = {
    "jaw": list(range(0, 17)),
    "eyebrows": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "eyes": list(range(36, 48)),
    "mouth": list(range(48, 60)),
    "forehead": list(range(68, 81)),
}


CAM_METHODS = {
    "GradCAM": GradCAM,
    "EigenGradCAM": EigenGradCAM,
    "LayerCAM": LayerCAM,
    "HiResCAM": HiResCAM,
}

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits

def analyze_attention(heatmap, landmarks, threshold=0.3):
    """Which regions have attention > threshold"""
    if heatmap.max() - heatmap.min() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    if landmarks is None:
        return {}
    
    results = {}
    for region_name, indices in FACIAL_REGIONS.items():
        region_points = landmarks[indices]
        if len(region_points) < 3:
            results[region_name] = 0
            continue
            
        # Simple bounding box approach
        x_min, y_min = region_points.min(axis=0)
        x_max, y_max = region_points.max(axis=0)
        
        # Get attention in this region
        x_min, x_max = int(x_min), int(x_max)
        y_min, y_max = int(y_min), int(y_max)
        
        if 0 <= x_min < x_max < heatmap.shape[1] and 0 <= y_min < y_max < heatmap.shape[0]:
            region_attention = heatmap[y_min:y_max, x_min:x_max]
            max_attention = np.max(region_attention)
            results[region_name] = 1 if max_attention > threshold else 0
        else:
            results[region_name] = 0
    
    return results

def reshape_transform_vit_huggingface(x):
    # Remove CLS token
    activations = x[:, 1:, :]
    # Reshape: for ViT-base-patch16-224, we have 196 patches = 14x14
    activations = activations.view(activations.shape[0], 14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

def load_images_from_valid_csv(dataset_name, image_filter=0):
    paths = get_dataset_paths(dataset_name)
    valid_csv = paths['VALID_CSV']
    
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Validation CSV not found at: {valid_csv}")
    
    print(f"Loading images from validation CSV: {valid_csv}")
    
    images, labels, filenames = load_images_from_csv(
        csv_path=valid_csv,
        dataset_name=dataset_name,
        is_train=False,
        target_label=image_filter
    )
    
    return images, labels, filenames
    
""" Helper function to run a given CAM on an image and create a visualization."""
def get_cam_visualizations(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_list_for_gradcam: List[List[Callable]],
                          reshape_transform: Callable,
                          input_tensors: List[torch.nn.Module],
                          input_images: List[Image.Image],
                          method: Callable=GradCAM):
    all_results = []

    with method(model=HuggingfaceToTensorModelWrapper(model), target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        for img_idx, (input_tensor, input_image, targets_for_gradcam) in enumerate(zip(input_tensors, input_images, targets_list_for_gradcam)):
            batch_results = cam(input_tensor=input_tensor, targets=targets_for_gradcam)
            results = []
            for grayscale_cam in batch_results:
                visualization = show_cam_on_image(np.float32(input_image)/255, grayscale_cam, use_rgb=True)
                results.append(visualization)

            all_results.append(np.hstack(results))

    return all_results

def get_cam_analysis(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_list_for_gradcam: List[List[Callable]],
                          reshape_transform: Callable,
                          input_tensors: List[torch.nn.Module],
                          input_images: List[Image.Image],
                          method: Callable=LayerCAM):
    all_heatmaps = []
    all_landmarks = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(DATA_ROOT/"shape_predictor_81_face_landmarks.dat"))

    with method(model=HuggingfaceToTensorModelWrapper(model), target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        for img_idx, (input_tensor, input_image, targets_for_gradcam) in enumerate(zip(input_tensors, input_images, targets_list_for_gradcam)):
            gray = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2GRAY)
            faces = detector(gray)

            landmarks = None
            if len(faces) > 0:
                # Get landmarks
                shape = predictor(gray, faces[0])
                landmarks = np.array([[shape.part(j).x, shape.part(j).y] for j in range(81)])

            all_landmarks.append(landmarks)

            raw_heatmap = cam(input_tensor=input_tensor, targets=targets_for_gradcam)[0]
            all_heatmaps.append(raw_heatmap)

    return all_heatmaps, all_landmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get GradCAM visualization or analysis on an existing ViT model'
    )
    parser.add_argument('--dataset', type=str, default="rvf10k",
                        choices=["rvf10k", "dalle2", "latent_diffusion", "midjourney", "StableDiffusion", "STARGAN", "taming_transformer_VQGAN"],
                        help='The dataset that the model trained on. ' +
                        'Options: rvf10k, dalle2, latent_diffusion, midjourney, StableDiffusion, STARGAN, taming_transformer_VQGAN')
    parser.add_argument('--mode', type=str, default="visualization", 
                       choices=["visualization", "analysis"],
                       help='Mode: visualization (generate 10 visualizations) or analysis (complete attention analysis)')

    args = parser.parse_args()

    # Load model
    dataset_name = args.dataset
    model_path = MODEL_FOLDER / f"ViT_baseline_{dataset_name}_model.pth"

    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    processor = checkpoint['processor']

    experiment = ViTBaselineExperiment(config)
    experiment.processor = processor
    experiment.create_model(2)
    experiment.vit_model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        experiment.vit_model = experiment.vit_model.cuda()
    else:
        device = torch.device('cpu')
        experiment.vit_model = experiment.vit_model.cpu()

    experiment.vit_model.eval()

    target_layer = experiment.vit_model.vit.encoder.layer[-2].output
    

    # Load data
    fake_images, fake_labels, filenames = load_images_from_valid_csv(dataset_name)

    if args.mode == "visualization":
        print(f"\n=== Visualization Mode ===")
        print(f"Generating visualizations for {dataset_name}")
        batch_size = 10
        
        batch_images = fake_images[:batch_size]
        batch_filenames = filenames[:batch_size]
        batch_true_labels = fake_labels[:batch_size]

        batch_predictions = experiment.model.predict(batch_images, verbose=0)
        batch_pred_labels = [np.argmax(pred) for pred in batch_predictions]
        batch_confidences = [batch_predictions[i][batch_pred_labels[i]] for i in range(len(batch_images))]

        pil_images = []
        rgb_images = []
        tensors = []
        targets_list = []
        for i, (img, filename, true_label, pred_label) in enumerate(zip(batch_images, batch_filenames, batch_true_labels, batch_pred_labels)):
            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((224, 224))
            pil_images.append(pil_img)
            rgb_images.append(np.array(pil_img) / 255.0)
            inputs = processor(images=[pil_img], return_tensors="pt")
            input_tensor = inputs['pixel_values'].to(device)
            tensors.append(input_tensor)
            targets_list.append([ClassifierOutputTarget(pred_label)])
        
        reverse_label_dict = {1: "Real", 0: "Fake"}

        for method_name, method in CAM_METHODS.items():
            results = get_cam_visualizations(
                model=experiment.vit_model,
                target_layer=target_layer,
                targets_list_for_gradcam=targets_list,
                input_tensors=tensors,
                input_images=pil_images,
                reshape_transform=reshape_transform_vit_huggingface,
                method=method
            )

            for i, result in enumerate(results):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                axes[0].imshow(rgb_images[i])
                axes[0].set_title(f'Original\nTrue: {reverse_label_dict[batch_true_labels[i]]}', fontsize=12)
                axes[0].axis('off')

                axes[1].imshow(result)
                axes[1].set_title(f'{method_name} Result\nPred: {reverse_label_dict[batch_pred_labels[i]]} ({batch_confidences[i]:.1%})', fontsize=12)
                axes[1].axis('off')

                plt.suptitle(f'{experiment.experiment_name} {method_name} Analysis for Deepfake Detection', fontsize=14, fontweight='bold')
                plt.tight_layout()

                output_dir = OUTPUT_FOLDER / experiment.experiment_name / dataset_name / method_name
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f'{Path(batch_filenames[i]).stem}_result.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')

                plt.show()
    elif args.mode == "analysis":
        print(f"\n=== Analysis Mode ===")
        print(f"Running attention analysis for {dataset_name}")

        csv_data = []
        batch_size = 16

        for batch_start in range(0, len(fake_images), batch_size):
            batch_end = min(batch_start + batch_size, len(fake_images))
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(fake_images)+batch_size-1)//batch_size}")

            batch_images = fake_images[batch_start:batch_end]
            batch_filenames = filenames[batch_start:batch_end]
            batch_true_labels = fake_labels[batch_start:batch_end]

            batch_predictions = experiment.model.predict(batch_images, verbose=0)
            batch_pred_labels = [np.argmax(pred) for pred in batch_predictions]
            batch_confidences = [batch_predictions[i][batch_pred_labels[i]] for i in range(len(batch_images))]

            pil_images = []
            rgb_images = []
            tensors = []
            targets_list = []
            for i, (img, filename, true_label, pred_label) in enumerate(zip(batch_images, batch_filenames, batch_true_labels, batch_pred_labels)):
                pil_img = Image.fromarray(img.astype(np.uint8))
                pil_img = pil_img.resize((224, 224))
                pil_images.append(pil_img)
                rgb_images.append(np.array(pil_img) / 255.0)
                inputs = processor(images=[pil_img], return_tensors="pt")
                input_tensor = inputs['pixel_values'].to(device)
                tensors.append(input_tensor)
                targets_list.append([ClassifierOutputTarget(pred_label)])
                    
            results, landmarks_list = get_cam_analysis(
                model=experiment.vit_model,
                target_layer=target_layer,
                targets_list_for_gradcam=targets_list,
                input_tensors=tensors,
                input_images=pil_images,
                reshape_transform=reshape_transform_vit_huggingface,
                method=LayerCAM
            )

            for i, (result, filename, true_label, pred_label, confidence) in enumerate(
                zip(results, batch_filenames, batch_true_labels, batch_pred_labels, batch_confidences)):
                
                region_attention = analyze_attention(result, landmarks_list[i])
                
                csv_row = {
                    "filename": Path(filename).name,
                    "prediction": pred_label,
                    "true_label": true_label,
                    "confidence": confidence,
                }
                
                for region in FACIAL_REGIONS.keys():
                    csv_row[f"attention_{region}"] = region_attention.get(region, 0)
                
                csv_data.append(csv_row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_output_path = OUTPUT_FOLDER / experiment.experiment_name / dataset_name / "layercam_analysis.csv"
            csv_output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_output_path, index=False)
            print(f"\nCSV saved to: {csv_output_path}")
            print(f"Total rows: {len(df)}")