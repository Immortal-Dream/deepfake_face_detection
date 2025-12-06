import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config.path_config import OUTPUT_FOLDER, MODEL_FOLDER
from src.utils.data_loader_utils import load_and_preprocess_data
from src.experiments.baselines.ViT.ViTBaselineExperiment import ViTBaselineExperiment

from typing import List, Callable

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits
    
""" Helper function to run GradCAM on an image and create a visualization."""
def run_grad_cam_on_images(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_list_for_gradcam: List[List[Callable]],
                          reshape_transform: Callable,
                          input_tensors: List[torch.nn.Module],
                          input_images: List[Image.Image],
                          method: Callable=GradCAM):
    all_results = []

    with method(model=HuggingfaceToTensorModelWrapper(model), target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        for img_idx, (input_tensor, input_image, targets_for_gradcam) in enumerate(zip(input_tensors, input_images, targets_list_for_gradcam)):
        
            # Replicate the tensor for each of the categories we want to create Grad-CAM for:
            repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

            batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)
            results = []
            for grayscale_cam in batch_results:
                visualization = show_cam_on_image(np.float32(input_image)/255, grayscale_cam, use_rgb=True)
                results.append(visualization)

            all_results.append(np.hstack(results))

    return all_results

def reshape_transform_vit_huggingface(x):
    # Remove CLS token
    activations = x[:, 1:, :]
    # Reshape: for ViT-base-patch16-224, we have 196 patches = 14x14
    activations = activations.view(activations.shape[0], 14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

if __name__ == "__main__":
    # Load model
    model_path = MODEL_FOLDER / "ViT_baseline_model.pth"
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

    # Load data
    csv_path = OUTPUT_FOLDER / "ViT_dataset.csv"
    X, y, label_dict = load_and_preprocess_data(csv_path)

    reverse_label_dict = {v: k for k, v in label_dict.items()}

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    batch_size = 10
    test_images = X_test[:batch_size]
    true_labels_one_hot = y_test[:batch_size]
    true_labels = [np.argmax(label) for label in true_labels_one_hot]

    predictions = experiment.model.predict(test_images, verbose=0)
    pred_labels = [np.argmax(pred) for pred in predictions]
    confidences = [predictions[i][pred_labels[i]] for i in range(batch_size)]

    pil_images = []
    rgb_images = []
    tensors = []
    targets_list = []

    for i, (img, pred_label) in enumerate(zip(test_images, pred_labels)):
        pil_img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        pil_img = pil_img.resize((224, 224))
        
        pil_images.append(pil_img)
        rgb_images.append(np.array(pil_img) / 255.0)
        tensors.append(transforms.ToTensor()(pil_img).to(device))
        targets_list.append([ClassifierOutputTarget(pred_label)])

    target_layer_gradcam = experiment.vit_model.vit.encoder.layer[-2].output

    results = run_grad_cam_on_images(
        model=experiment.vit_model,
        target_layer=target_layer_gradcam,
        targets_list_for_gradcam=targets_list,
        input_tensors=tensors,
        input_images=pil_images,
        reshape_transform=reshape_transform_vit_huggingface
    )

    for i, result in enumerate(results):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(rgb_images[i])
        axes[0].set_title(f'Original\nTrue: {reverse_label_dict[true_labels[i]]}', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(result)
        axes[1].set_title(f'Grad-CAM Result\nPred: {reverse_label_dict[pred_labels[i]]} ({confidences[i]:.1%})', fontsize=12)
        axes[1].axis('off')

        plt.suptitle(f'{experiment.experiment_name} Grad-CAM Analysis for Deepfake Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_dir = OUTPUT_FOLDER / experiment.experiment_name / "gradcam"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'grad_cam_result_{i}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
