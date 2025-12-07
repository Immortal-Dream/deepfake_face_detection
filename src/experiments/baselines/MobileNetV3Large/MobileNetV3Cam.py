from navigator_updater.static.css import DATA_PATH
from pytorch_grad_cam import GradCAM, EigenGradCAM, LayerCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.utils.data_loader_utils import load_images_from_csv
from src.experiments.baselines.MobileNetV3Large import MobileNetV3LargeBaselineExperiment

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path

from src.config.path_config import *
from enum import Enum

class CAM_TYPE(Enum):
    LAYER = 'layer_cam'
    GRAD = 'grad_cam'
    HiRes = 'HiRes_cam'
    EigenGrad = 'EigenGrad_cam'


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # create a model that maps the input image to the activations
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # this is the gradient of the output neuron
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # global average pooling for grad-cam
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiplication
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def make_layercam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # layer-cam implementation for tensorflow
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # remove batch dimension for processing
    grads = grads[0]
    activations = last_conv_layer_output[0]

    # layer-cam logic: we use element-wise gradients (rectified) as weights
    # instead of global average pooling. this preserves spatial details.
    weighted_grads = tf.maximum(grads, 0)

    # element-wise multiplication
    cam = tf.multiply(activations, weighted_grads)

    # sum along the channel axis
    heatmap = tf.reduce_sum(cam, axis=-1)

    # relu on the result
    heatmap = tf.maximum(heatmap, 0)

    # normalize (avoid division by zero)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def save_cam_image(img, heatmap, filename, alpha=0.4):
    # rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # use jet colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # create an image with rgb colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)

    # resize the heatmap to match the original image dimensions
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # ensure img is 0-255 for saving
    if np.max(img) <= 1.0:
        img = img * 255.0

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # save the file
    superimposed_img.save(filename)


if __name__ == '__main__':
    # define configuration
    experiment_name = 'MobileNetV3Large_baseline'
    model_path = MODEL_FOLDER / 'MobileNetV3Large_baseline_model.h5'
    dataset_name = "rvf10k"

    # set the method here (LAYER or GRAD)
    cam_method = CAM_TYPE.LAYER.value

    # set how many images to process (batch control)
    batch_limit = 10

    # get dataset paths
    dataset_path_dict = get_dataset_paths(dataset_name)
    valid_csv = dataset_path_dict['VALID_CSV']

    # define output folder
    cam_output_folder = OUTPUT_FOLDER / experiment_name / dataset_name / cam_method

    # create output directory if it doesn't exist
    if not os.path.exists(cam_output_folder):
        os.makedirs(cam_output_folder)
        print(f"created output directory: {cam_output_folder}")

    # load images
    print("loading images...")
    images, valid_labels, valid_filenames = load_images_from_csv(valid_csv, dataset_name, False)

    # load model
    print(f"loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # layer name for mobilenetv3
    last_conv_layer_name = "Conv_1"

    # determine how many images to loop through
    num_images_to_process = min(batch_limit, len(images))
    print(f"generating {cam_method} for first {num_images_to_process} images...")

    for i in range(num_images_to_process):
        try:
            # get original image and filename
            original_img = images[i]
            original_fname = Path(valid_filenames[i]).stem

            # resize image to 224x224 for the model
            img_resized = tf.image.resize(original_img, (224, 224))
            img_tensor = np.expand_dims(img_resized, axis=0)

            # suppress tf logging
            tf.get_logger().setLevel('ERROR')

            # generate heatmap based on selected method
            heatmap = None
            if cam_method == CAM_TYPE.LAYER.value:
                heatmap = make_layercam_heatmap(
                    img_tensor,
                    model,
                    last_conv_layer_name
                )
            elif cam_method == CAM_TYPE.GRAD.value:
                heatmap = make_gradcam_heatmap(
                    img_tensor,
                    model,
                    last_conv_layer_name
                )

            # save the result
            if heatmap is not None:
                output_filename = f"{cam_method}_{i}_{original_fname}.jpg"
                save_path = cam_output_folder / output_filename

                # pass original_img to keep high resolution in saved file
                save_cam_image(original_img, heatmap, save_path)

                print(f"saved {output_filename}")

        except Exception as e:
            print(f"error processing image {i}: {e}")

    print("done.")


