
# ============================================================
# CSC 8260 - ADVANCED IMAGE PROCESSING
# FULL LSUI TEST SET INFERENCE
#
# Models:
#   1. Water-Net
#   2. U-Shape Transformer
#
# Student: Koushik Kourikanti
#
# Uses existing tested inference scripts.
# Does NOT modify the original dataset or pilot outputs.
# ============================================================

from pathlib import Path
import argparse
import csv
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    ROOT / "data" / "lsui"
    / "original_split" / "LSUI" / "test"
)

OUTPUT_ROOT = ROOT / "outputs"
LOG_DIR = OUTPUT_ROOT / "metadata"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

IMAGE_SIZE = 256


def collect_images():
    """Load all 400 official LSUI test-image filenames."""

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DIR}"
        )

    images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"LSUI test images found: {len(images)}")

    if len(images) != 400:
        raise RuntimeError(
            f"Expected 400 images, found {len(images)}"
        )

    if len({p.stem for p in images}) != len(images):
        raise RuntimeError(
            "Duplicate image stems detected."
        )

    return images


def verify_output(path):
    """Check that an output exists and is readable."""

    if not path.is_file():
        return False

    image = cv2.imread(str(path))

    if image is None:
        return False

    return image.shape[:2] == (
        IMAGE_SIZE, IMAGE_SIZE
    )


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["waternet", "ushape"],
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process first N images; 0 means all 400"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs"
    )

    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")

    images = collect_images()

    if args.limit > 0:
        images = images[:args.limit]

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SELECT MODEL
    # ========================================================

    if args.model == "waternet":

        import run_waternet as model_script

        # Both full-test models use 256 x 256.
        model_script.WIDTH = IMAGE_SIZE
        model_script.HEIGHT = IMAGE_SIZE

        output_dir = OUTPUT_ROOT / "water_net_400"

        model_script.OUTPUT_DIR = output_dir

        device_name = "TensorFlow environment"

    else:

        import run_ushape as model_script

        import torch

        output_dir = OUTPUT_ROOT / "u_shape_400"

        model_script.OUTPUT_DIR = output_dir

        model_script.NORMALIZED_DIR = (
            OUTPUT_ROOT / "u_shape_normalized_400"
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        device_name = str(device)

        # Load pretrained generator once for all images.
        model = model_script.load_model(device)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path = (
        LOG_DIR / f"{args.model}_400_runtime.csv"
    )

    write_header = (
        not log_path.exists()
        or log_path.stat().st_size == 0
    )

    completed = 0
    skipped = 0

    print("\n" + "=" * 60)
    print("AIP - FULL LSUI DATASET INFERENCE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device_name}")
    print(f"Images selected: {len(images)}")
    print(f"Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Output folder: {output_dir}")
    print("=" * 60)

    with log_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "image_name",
                "processing_wall_seconds",
                "output_width",
                "output_height",
                "output_file"
            ])

        # ====================================================
        # INFERENCE LOOP
        # ====================================================

        for index, image_path in enumerate(images, 1):

            output_path = (
                output_dir / f"{image_path.stem}.png"
            )

            print(
                f"\n[{index}/{len(images)}] "
                f"Processing {image_path.name}"
            )

            if not args.force and verify_output(output_path):
                print("Already processed - skipping")
                skipped += 1
                continue

            start = time.perf_counter()

            if args.model == "waternet":

                # Override the paths used by the working
                # single-image Water-Net script.
                model_script.INPUT = image_path
                model_script.IMAGE_NAME = image_path.name
                model_script.OUTPUT = output_path

                model_script.run_waternet()

            else:

                # Use the same pretrained U-Shape generator.
                model_script.enhance_image(
                    model,
                    image_path,
                    device
                )

            elapsed = time.perf_counter() - start

            if not verify_output(output_path):
                raise RuntimeError(
                    f"Invalid output: {output_path}"
                )

            writer.writerow([
                image_path.name,
                f"{elapsed:.4f}",
                IMAGE_SIZE,
                IMAGE_SIZE,
                str(output_path)
            ])

            file.flush()

            completed += 1

            print(
                f"Completed {completed}; "
                f"skipped {skipped}; "
                f"wall time {elapsed:.3f} s"
            )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    valid_outputs = sum(
        verify_output(
            output_dir / f"{p.stem}.png"
        )
        for p in images
    )

    print("\n" + "=" * 60)
    print("FULL INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"New outputs: {completed}")
    print(f"Skipped existing: {skipped}")
    print(f"Valid outputs: {valid_outputs}/{len(images)}")
    print(f"Log: {log_path}")
    print(f"Output folder: {output_dir}")
    print("=" * 60)

    if valid_outputs != len(images):
        raise RuntimeError(
            "Some test images are missing valid outputs."
        )


if __name__ == "__main__":
    run()
