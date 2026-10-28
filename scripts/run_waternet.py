
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# AIP PROJECT - UNDERWATER IMAGE ENHANCEMENT
# Baseline Model: Water-Net
# Student: Koushik Kourikanti
# Dataset: LSUI
# Framework: TensorFlow 2.15.1 (TF1 compatibility)
# ============================================================

tf.compat.v1.disable_eager_execution()

# ============================================================
# STEP 1: PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    ROOT
    / "external"
    / "Water-Net_Code"
    / "testing_code_by_Branimir Ambrekovic"
)

sys.path.insert(0, str(MODEL_DIR))

import model as waternet

# Select image from command-line argument.
# Default image is 1003.jpg.
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "1003.jpg"

INPUT_DIR = ROOT / "data" / "pilot" / "input"
INPUT = INPUT_DIR / IMAGE_NAME

OUTPUT_DIR = ROOT / "outputs" / "water_net"
OUTPUT = OUTPUT_DIR / f"{INPUT.stem}.png"

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "water_net"
    / "checkpoint"
    / "coarse_112"
)

CHECKPOINT = CHECKPOINT_DIR / "coarse.model-1500"

# Pilot resolution
WIDTH = 320
HEIGHT = 256


# ============================================================
# STEP 2: PREPARE IMAGE BATCH
# ============================================================

def prepare_batch(image_bgr):
    """
    Convert BGR image into normalized RGB float32 batch.

    Output shape: (1, HEIGHT, WIDTH, 3)
    Pixel values: 0.0 to 1.0
    """

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_float = image_rgb.astype(np.float32) / 255.0

    return np.expand_dims(image_float, axis=0)


# ============================================================
# STEP 3: WATER-NET PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Generate the four inputs for Water-Net.

    1. Original underwater image
    2. White-balanced image
    3. Contrast-enhanced image
    4. Gamma-corrected image

    Uses the contributed Python preprocessing.
    """

    # Original image
    original = image.copy()

    # White balance
    wb = waternet.white_balance(
        image.copy(),
        0.5
    )

    # Contrast enhancement using CLAHE
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(gray)

    ce = cv2.cvtColor(
        contrast,
        cv2.COLOR_GRAY2BGR
    )

    # Gamma correction
    gc = waternet.adjust_gamma(
        image.copy(),
        0.7
    )

    return (
        prepare_batch(original),
        prepare_batch(wb),
        prepare_batch(ce),
        prepare_batch(gc)
    )


# ============================================================
# STEP 4: LOAD AND RUN WATER-NET
# ============================================================

def run_waternet():

    print("\n" + "=" * 55)
    print("AIP PROJECT - WATER-NET BASELINE INFERENCE")
    print("=" * 55)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check input image
    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Input image not found: {INPUT}"
        )

    # Check pretrained model
    if not Path(
        str(CHECKPOINT) + ".index"
    ).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # Read original image
    image = cv2.imread(
        str(INPUT),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {INPUT}"
        )

    print(f"\nInput image: {IMAGE_NAME}")
    print(f"Original resolution: {image.shape[1]}x{image.shape[0]}")

    # Resize for initial inference
    image = cv2.resize(
        image,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    print(f"Inference resolution: {WIDTH}x{HEIGHT}")

    # Generate Water-Net inputs
    print("\nPreparing input images...")

    original, wb, ce, gc = preprocess_image(image)

    print("Original image prepared")
    print("White balance completed")
    print("Contrast enhancement completed")
    print("Gamma correction completed")

    # Build TensorFlow graph
    with tf.Graph().as_default():

        with tf.compat.v1.Session() as sess:

            print("\nBuilding Water-Net model...")

            network = waternet.T_CNN(
                sess,
                image_height=HEIGHT,
                image_width=WIDTH,
                label_height=112,
                label_width=112,
                batch_size=1,
                c_dim=3,
                c_depth_dim=1,
                checkpoint_dir=str(
                    CHECKPOINT_DIR.parent
                ),
                sample_dir=str(OUTPUT_DIR),
                test_image_name=str(INPUT),
                id=0
            )

            # Initialize model variables
            sess.run(
                tf.compat.v1.global_variables_initializer()
            )

            # Restore pretrained model
            print("\nRestoring pretrained Water-Net weights...")

            network.saver.restore(
                sess,
                str(CHECKPOINT)
            )

            print("Checkpoint restored successfully.")

            # Run actual inference
            print("\nRunning Water-Net inference...")

            start_time = time.perf_counter()

            result = sess.run(
                network.pred_h,
                feed_dict={
                    network.images: original,
                    network.images_wb: wb,
                    network.images_ce: ce,
                    network.images_gc: gc
                }
            )

            inference_time = (
                time.perf_counter() - start_time
            )

    # ========================================================
    # STEP 5: SAVE ENHANCED IMAGE
    # ========================================================

    output_rgb = np.clip(
        result[0],
        0.0,
        1.0
    )

    output_uint8 = np.round(
        output_rgb * 255.0
    ).astype(np.uint8)

    output_bgr = cv2.cvtColor(
        output_uint8,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        str(OUTPUT),
        output_bgr
    )

    if not success:
        raise RuntimeError(
            "Failed to save enhanced image."
        )

    print("\n" + "=" * 55)
    print("WATER-NET INFERENCE COMPLETED")
    print("=" * 55)

    print(f"Input: {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Output resolution: {WIDTH}x{HEIGHT}")
    print(f"Inference time: {inference_time:.3f} seconds")
    print("Status: SUCCESS")
    print("=" * 55 + "\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_waternet()
