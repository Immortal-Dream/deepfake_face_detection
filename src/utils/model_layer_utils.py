import torch
import os

from config.path_config import MODEL_FOLDER
from src.config import *


def inspect_pth_file(file_path):
    """
    Loads a .pth file and prints its structure, layer names, and shapes.
    """
    print(f"\n{'=' * 60}")
    print(f"Inspecting file: {file_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        # Load with map_location='cpu' to avoid CUDA errors on non-GPU machines
        content = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # 1. Determine the type of content
    state_dict = None
    if isinstance(content, torch.nn.Module):
        print("Detected Type: [Full Model Object]")
        state_dict = content.state_dict()
    elif isinstance(content, dict):
        # Check if it's a checkpoint dict (contains 'epoch', 'optimizer', etc.)
        # Common keys are 'state_dict', 'model_state_dict', 'model', 'params'
        possible_keys = ['state_dict', 'model_state_dict', 'model', 'params']
        found_key = None
        for key in possible_keys:
            if key in content and isinstance(content[key], dict):
                found_key = key
                break

        if found_key:
            print(f"Detected Type: [Checkpoint Dictionary]")
            print(f"Top-level keys found: {list(content.keys())}")
            print(f"Extracting weights from key: '{found_key}'")
            state_dict = content[found_key]
        else:
            # Assume it is a pure state_dict
            print("Detected Type: [Pure State Dict]")
            state_dict = content
    else:
        print(f"Unknown data structure type: {type(content)}")
        return

    # 2. Print Layer Information
    if state_dict:
        print(f"\n{'-' * 90}")
        print(f"{'Layer Name (Key)':<50} | {'Shape':<25} | {'Dtype'}")
        print(f"{'-' * 90}")

        total_params = 0
        try:
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    shape_str = str(list(value.shape))
                    dtype_str = str(value.dtype).replace('torch.', '')
                    print(f"{key:<50} | {shape_str:<25} | {dtype_str}")
                    total_params += value.numel()
                else:
                    # Handle non-tensor data (e.g., batch_norm num_batches_tracked or metadata)
                    print(f"{key:<50} | {'[Non-Tensor]':<25} | {type(value)}")

            print(f"{'-' * 90}")
            print(f"Total Parameters: {total_params:,}")
            print(f"Total Layers/Keys: {len(state_dict)}")
            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"Error parsing state_dict: {e}")
    else:
        print("Failed to extract a valid state_dict.")


if __name__ == "__main__":
    inspect_pth_file(MODEL_FOLDER / 'ShuffleNetV2_baseline_midjourney.pth')
    inspect_pth_file(MODEL_FOLDER / 'xception_BSL_latent_diffusion.pth')