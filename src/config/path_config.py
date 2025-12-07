from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Constants
rvf10k = "rvf10k"
STARGAN = "STARGAN"
STABLE_DIFFUSION = "StableDiffusion"
DALLE2 = "dalle2"
LATENT_DIFFUSION = "latent_diffusion"
MIDJOURNEY = "midjourney"
TAMING_TRANSFORMER_VQGAN = "taming_transformer_VQGAN"


# -----------------------------
# DATASET ROOTS
# -----------------------------
DATASET_ROOTS = {
    "rvf10k": DATA_ROOT / "rvf10k",
    "STARGAN": DATA_ROOT / "STARGAN",
    "StableDiffusion": DATA_ROOT / "StableDiffusion",
    "dalle2": DATA_ROOT / "dalle2",
    "latent_diffusion": DATA_ROOT / "latent_diffusion",
    "midjourney": DATA_ROOT / "midjourney",
    "taming_transformer_VQGAN": DATA_ROOT / "taming_transformer_VQGAN",
}


# -----------------------------
#  FUNCTION — RETURNS PATHS FOR ANY DATASET
# -----------------------------
def get_dataset_paths(dataset_name: str):
    root = DATASET_ROOTS.get(dataset_name)

    if root is None:
        raise ValueError(f"[ERROR] Unknown dataset: {dataset_name}")

    return {
        "ROOT": root,
        "TRAIN_ROOT": root / "train",
        "VALID_ROOT": root / "valid",
        "TRAIN_CSV": root / "train.csv",
        "VALID_CSV": root / "valid.csv",
    }


# -----------------------------
# OUTPUT FOLDERS
# -----------------------------
OUTPUT_FOLDER = DATA_ROOT / "output"
OVERALL_FOLDER = OUTPUT_FOLDER / "overall"
RVF10K_ROOT = DATA_ROOT / "rvf10k"
# Models folder (required by BaseExperiment)
MODEL_FOLDER = PROJECT_ROOT / "models"


VALID_CSV = 'VALID_CSV'
VALID_ROOT = "VALID_ROOT"
TRAIN_ROOT = 'TRAIN_ROOT'
TRAIN_CSV = 'TRAIN_CSV'
