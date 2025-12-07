"""
Image Data Loader for Deepfake Detection
Provides utilities to load images and labels from CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import warnings

from config.path_config import TRAIN_ROOT, VALID_ROOT
from src.config.path_config import RVF10K_ROOT, get_dataset_paths

warnings.filterwarnings('ignore')

# Global configuration
DATA_ROOT = Path("data/rvf10k")
TRAIN_CSV = DATA_ROOT / "train.csv"
VALID_CSV = DATA_ROOT / "valid.csv"


def load_and_preprocess_data(csv_path, height=224, width=224):
    """
    Load images from CSV and preprocess for training.
    Returns: X (images), y (one-hot labels), label_dict (name->id mapping)
    """
    print(f"Loading images from {csv_path}...")
    data = pd.read_csv(csv_path)

    label_names = data['label'].unique()
    label_dict = {name: idx for idx, name in enumerate(label_names)}
    print(f"Classes found: {label_dict}")

    y = data['label'].map(label_dict).values
    y = to_categorical(y, num_classes=len(label_dict))

    X = np.empty((data.shape[0], height, width, 3), dtype=np.float32)
    for i, img_path in enumerate(data['image'].values):
        if i % 100 == 0:
            print(f"Loading image {i + 1}/{data.shape[0]}")
        img = load_img(img_path, target_size=(height, width))
        img_array = img_to_array(img)
        X[i] = img_array

    print(f"Data loaded: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, label_dict


def load_image_list_from_csv(csv_path, return_labels=False):
    """
    Load image filenames from CSV file.

    Args:
        csv_path: Path to CSV file
        return_labels: If True, also return labels and label strings

    Returns:
        If return_labels=False: List of filenames
        If return_labels=True: Tuple of (filenames, labels, label_strings)
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    if 'filename' not in df.columns:
        raise ValueError("CSV must contain 'filename' column")

    filenames = df['filename'].tolist()

    if return_labels:
        if 'label' not in df.columns or 'label_str' not in df.columns:
            raise ValueError("CSV must contain 'label' and 'label_str' columns for labels")

        labels = df['label'].tolist()
        label_strings = df['label_str'].tolist()

        print(f"Loaded {len(filenames)} images from {csv_path}")
        print(f"  - Real images: {sum(1 for l in labels if l == 1)}")
        print(f"  - Fake images: {sum(1 for l in labels if l == 0)}")

        return filenames, labels, label_strings
    else:
        print(f"Loaded {len(filenames)} image filenames from {csv_path}")
        return filenames


def load_image_with_filename(filename, is_real, is_train):
    image_path = RVF10K_ROOT
    if is_train:
        image_path = image_path / "train"
    else:
        image_path = image_path / "valid"

    if is_real:
        image_path = image_path / "real"
    else:
        image_path = image_path / "fake"

    image_path = image_path / filename
    return load_image(image_path)


def load_image(image_path, resize=None, to_rgb=True):
    """
    Load a single image file.

    Args:
        image_path: Path to image file
        resize: Tuple (width, height) to resize image, None to keep original
        to_rgb: Convert to RGB format

    Returns:
        numpy.ndarray: Image array
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load with PIL
    img = Image.open(image_path)

    if to_rgb and img.mode != 'RGB':
        img = img.convert('RGB')

    if resize is not None:
        img = img.resize(resize, Image.BILINEAR)

    return np.array(img)


def load_images_from_csv(csv_path, dataset_name, is_train = True, resize=None, max_images=None, verbose=True):
    """
    Load all images listed in CSV file.

    Args:
        csv_path: Path to CSV file
        dataset_name: Name of dataset "like rvf10k"
        is_train: whether to load train dataset
        resize: Tuple (width, height) to resize images
        max_images: Maximum number of images to load (None for all)
        verbose: Print loading progress

    Returns:
        Tuple of (images, labels, filenames)
            images: numpy array of shape (N, H, W, C)
            labels: list of labels
            filenames: list of filenames
    """
    filenames, labels, label_strings = load_image_list_from_csv(csv_path, return_labels=True)

    if max_images is not None:
        filenames = filenames[:max_images]
        labels = labels[:max_images]
    path_dict = get_dataset_paths(dataset_name)
    specific_path = path_dict[TRAIN_ROOT] if is_train else path_dict[VALID_ROOT]

    images = []
    valid_labels = []
    valid_filenames = []

    for i, (filename, label) in enumerate(zip(filenames, labels)):
        try:
            # Determine subdirectory based on label
            if label == 1:
                img_path = specific_path / "real" / filename
            else:
                img_path = specific_path / "fake" / filename

            img = load_image(img_path, resize=resize)
            images.append(img)
            valid_labels.append(label)
            valid_filenames.append(filename)

            if verbose and (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(filenames)} images...")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {filename}: {e}")
            continue

    images = np.array(images)

    if verbose:
        print(f"\nSuccessfully loaded {len(images)} images")
        print(f"Image shape: {images.shape}")

    return images, valid_labels, valid_filenames
