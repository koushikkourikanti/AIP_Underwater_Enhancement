
# ============================================================
# AIP PROJECT - UNDERWATER IMAGE ENHANCEMENT
# Target Model: U-Shape Transformer (2023)
# Student: Koushik Kourikanti
# University: Georgia State University
# Dataset: LSUI
#
# Features:
# 1. Loads the original pretrained U-Shape Transformer
# 2. Uses NVIDIA GPU when available
# 3. Processes individual images or all pilot images
# 4. Saves original clipped model outputs
# 5. Saves author-style normalized outputs
# 6. Records model inference time
# ============================================================

from pathlib import Path
import sys
import time
import argparse

import cv2
import numpy as np
import torch
from torchvision.utils import save_image


# ============================================================
# STEP 1: PROJECT DIRECTORIES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = ROOT / "external" / "U-shape_Transformer"

sys.path.insert(0, str(REPO_DIR))

from net.Ushape_Trans import Generator

INPUT_DIR = ROOT / "data" / "pilot" / "input"

OUTPUT_DIR = ROOT / "outputs" / "u_shape"

NORMALIZED_DIR = ROOT / "outputs" / "u_shape_normalized"

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "u_shape"
    / "saved_models"
    / "G"
    / "generator_795.pth"
)

IMAGE_SIZE = 256


# ============================================================
# STEP 2: LOAD PRETRAINED MODEL
# ============================================================

def load_model(device):

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    print("\nBuilding U-Shape Transformer...")

    model = Generator()

    print("Loading pretrained model weights...")

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(device)

    model.eval()

    print("Pretrained checkpoint loaded successfully.")

    return model


# ============================================================
# STEP 3: IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path, device):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read image: {image_path}"
        )

    original_height, original_width = image.shape[:2]

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb)
    )

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0).to(device)

    return tensor, (original_width, original_height)


# ============================================================
# STEP 4: ENHANCE UNDERWATER IMAGE
# ============================================================

def enhance_image(model, image_path, device):

    print("\n" + "=" * 60)
    print(f"Processing LSUI image: {image_path.name}")
    print("=" * 60)

    image_tensor, original_size = preprocess_image(
        image_path,
        device
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Run pretrained model
    with torch.inference_mode():

        results = model(image_tensor)

        if not isinstance(results, (tuple, list)):
            raise RuntimeError(
                "Unexpected generator output type."
            )

        if len(results) != 4:
            raise RuntimeError(
                f"Expected 4 outputs, got {len(results)}."
            )

        # Final full-resolution generator prediction
        enhanced = results[3]

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time

    # ========================================================
    # STEP 5: CHECK MODEL OUTPUT
    # ========================================================

    raw_min = enhanced.min().item()
    raw_max = enhanced.max().item()

    print(f"Raw output minimum: {raw_min:.4f}")
    print(f"Raw output maximum: {raw_max:.4f}")

    if not torch.isfinite(enhanced).all().item():
        raise RuntimeError(
            "Model output contains NaN or infinity."
        )

    print("Model inference completed.")

    # ========================================================
    # STEP 6: SAVE CLIPPED MODEL OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / f"{image_path.stem}.png"
    )

    clipped_output = enhanced.detach().clamp(0.0, 1.0)

    output_rgb = clipped_output.squeeze(0).cpu()

    output_rgb = output_rgb.permute(1, 2, 0).numpy()

    output_uint8 = np.round(
        output_rgb * 255.0
    ).astype(np.uint8)

    output_bgr = cv2.cvtColor(
        output_uint8,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        str(output_path),
        output_bgr
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {output_path}"
        )

    print(f"Clipped model output saved: {output_path}")

    # ========================================================
    # STEP 7: SAVE AUTHOR-STYLE NORMALIZED OUTPUT
    # ========================================================

    NORMALIZED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    normalized_path = (
        NORMALIZED_DIR / f"{image_path.stem}.png"
    )

    # Uses min-max normalization for visualization.
    # Do not use this version for quantitative evaluation
    # unless this processing is part of the declared method.

    save_image(
        enhanced.detach().cpu(),
        str(normalized_path),
        normalize=True
    )

    print(f"Normalized output saved: {normalized_path}")

    # ========================================================
    # STEP 8: DISPLAY IMAGE INFORMATION
    # ========================================================

    print("\nIMAGE INFORMATION")
    print("-" * 60)

    print(f"Input image: {image_path.name}")

    print(
        f"Original resolution: "
        f"{original_size[0]}x{original_size[1]}"
    )

    print(
        f"Model output resolution: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Inference time: "
        f"{inference_time:.3f} seconds"
    )

    print("Status: SUCCESS")
    print("=" * 60)

    return inference_time


# ============================================================
# STEP 9: MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="U-Shape Transformer LSUI Inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        default="1003.jpg",
        help="Single image from LSUI pilot dataset"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all LSUI pilot images"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 60)
    print("AIP PROJECT - U-SHAPE TRANSFORMER")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # Load model once
    model = load_model(device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    NORMALIZED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Select input images
    if args.all:

        images = sorted(
            path for path in INPUT_DIR.iterdir()
            if path.suffix.lower() in (
                ".jpg", ".jpeg", ".png"
            )
        )

    else:

        if Path(args.image).name != args.image:
            raise ValueError(
                "Provide an image filename only."
            )

        images = [
            INPUT_DIR / args.image
        ]

    if not images:
        raise RuntimeError(
            "No LSUI input images were found."
        )

    inference_times = []

    # Process selected images
    for index, image_path in enumerate(images, start=1):

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        print(
            f"\nImage {index} of {len(images)}"
        )

        elapsed = enhance_image(
            model,
            image_path,
            device
        )

        inference_times.append(elapsed)

    # ========================================================
    # STEP 10: FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("U-SHAPE TRANSFORMER - FINAL SUMMARY")
    print("=" * 60)

    print(
        f"Total images processed: {len(inference_times)}"
    )

    print(
        f"Average model inference time: "
        f"{np.mean(inference_times):.3f} seconds"
    )

    print(
        f"Clipped outputs: {OUTPUT_DIR}"
    )

    print(
        f"Normalized outputs: {NORMALIZED_DIR}"
    )

    print("STATUS: ALL IMAGES PROCESSED SUCCESSFULLY")

    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
