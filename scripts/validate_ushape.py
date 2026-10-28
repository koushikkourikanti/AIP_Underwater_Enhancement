
from pathlib import Path
import cv2
import numpy as np
import torch

from run_ushape import (
    ROOT,
    load_model,
    preprocess_image,
)

# Official example images from the authors' repository
TEST_DIR = ROOT / "external" / "U-shape_Transformer" / "test"
INPUT_DIR = TEST_DIR / "input"
GT_DIR = TEST_DIR / "GT"

OUTPUT_DIR = ROOT / "outputs" / "validation_u_shape"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 256

def main():
    if not INPUT_DIR.is_dir() or not GT_DIR.is_dir():
        raise FileNotFoundError(
            "Official input or GT folder is missing."
        )

    inputs = {
        p.stem: p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    references = {
        p.stem: p for p in GT_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    common = sorted(set(inputs) & set(references))

    print("Matching official image pairs:", len(common))

    if not common:
        raise RuntimeError("No matching example pairs found.")

    # Select one corresponding input/reference pair.
    name = common[0]
    image_path = inputs[name]
    reference_path = references[name]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    tensor, _ = preprocess_image(image_path, device)

    with torch.inference_mode():
        prediction = model(tensor)[3]

    print("Output range:",
          prediction.min().item(),
          prediction.max().item())

    prediction = prediction.clamp(0, 1)
    prediction = prediction[0].permute(1, 2, 0)
    prediction = prediction.cpu().numpy()

    enhanced = (
        prediction * 255.0
    ).round().astype(np.uint8)

    enhanced = cv2.cvtColor(
        enhanced, cv2.COLOR_RGB2BGR
    )

    original = cv2.imread(str(image_path))
    reference = cv2.imread(str(reference_path))

    if original is None or reference is None:
        raise RuntimeError("Could not read the example pair.")

    original = cv2.resize(original, (SIZE, SIZE))
    reference = cv2.resize(reference, (SIZE, SIZE))

    input_psnr = cv2.PSNR(original, reference)
    output_psnr = cv2.PSNR(enhanced, reference)

    print(f"Example: {name}")
    print(f"Input PSNR: {input_psnr:.2f} dB")
    print(f"Enhanced PSNR: {output_psnr:.2f} dB")

    images = [original, enhanced, reference]
    labels = ["Original", "U-Shape", "Reference"]

    canvas = np.full(
        (SIZE + 55, SIZE * 3, 3), 255, np.uint8
    )

    for i, (image, label) in enumerate(zip(images, labels)):
        x = i * SIZE
        canvas[55:, x:x + SIZE] = image
        cv2.putText(
            canvas,
            label,
            (x + 12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (25, 25, 25),
            2
        )

    output_path = OUTPUT_DIR / f"{name}_validation.png"

    cv2.imwrite(str(output_path), canvas)

    print("Validation image saved:", output_path)


if __name__ == "__main__":
    main()
