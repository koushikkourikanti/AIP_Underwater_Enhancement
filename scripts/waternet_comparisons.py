
from pathlib import Path
import math

import cv2
from PIL import Image, ImageDraw, ImageFont

# AIP Project - Water-Net Baseline Visual Comparisons

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "pilot" / "input"
OUTPUT_DIR = ROOT / "outputs" / "water_net"
COMPARE_DIR = ROOT / "outputs" / "comparisons"

COMPARE_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 320
HEIGHT = 256
PADDING = 20
GAP = 20
HEADER = 80

PANEL_W = WIDTH * 2 + GAP + PADDING * 2
PANEL_H = HEIGHT + HEADER + PADDING

def get_font(size):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/arial.ttf", size
        )
    except OSError:
        return ImageFont.load_default()

title_font = get_font(21)
label_font = get_font(17)

def make_comparison(image_path):
    name = image_path.stem
    enhanced_path = OUTPUT_DIR / f"{name}.png"

    if not enhanced_path.exists():
        print(f"Skipping {name}: output not found")
        return None

    original = cv2.imread(str(image_path))
    enhanced = cv2.imread(str(enhanced_path))

    if original is None or enhanced is None:
        print(f"Skipping {name}: unreadable image")
        return None

    original = cv2.resize(
        original, (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    enhanced = cv2.resize(
        enhanced, (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    original = Image.fromarray(
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    )

    enhanced = Image.fromarray(
        cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    )

    canvas = Image.new(
        "RGB", (PANEL_W, PANEL_H), "white"
    )
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (PADDING, 12),
        f"LSUI Image: {name}",
        font=title_font,
        fill=(15, 41, 74)
    )

    x1 = PADDING
    x2 = PADDING + WIDTH + GAP
    y = HEADER

    draw.text(
        (x1, 53),
        "Original LSUI (resized)",
        font=label_font,
        fill=(50, 50, 50)
    )

    draw.text(
        (x2, 53),
        "Water-Net enhanced",
        font=label_font,
        fill=(50, 50, 50)
    )

    canvas.paste(original, (x1, y))
    canvas.paste(enhanced, (x2, y))

    output_path = COMPARE_DIR / f"{name}_comparison.png"
    canvas.save(output_path)

    print(f"Saved: {output_path.name}")
    return canvas

def main():
    images = sorted(INPUT_DIR.glob("*.jpg"))

    panels = []

    for image_path in images:
        panel = make_comparison(image_path)

        if panel is not None:
            panels.append(panel)

    if not panels:
        raise RuntimeError(
            "No comparison images were created."
        )

    columns = 2
    rows = math.ceil(len(panels) / columns)
    margin = 20

    sheet_width = columns * PANEL_W + (columns + 1) * margin
    sheet_height = rows * PANEL_H + (rows + 1) * margin

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        (225, 233, 241)
    )

    for i, panel in enumerate(panels):
        row = i // columns
        col = i % columns

        x = margin + col * (PANEL_W + margin)
        y = margin + row * (PANEL_H + margin)

        sheet.paste(panel, (x, y))

    sheet_path = COMPARE_DIR / "waternet_all_comparisons.png"
    sheet.save(sheet_path)

    print("\nComparison generation completed!")
    print(f"Total comparisons: {len(panels)}")
    print(f"Combined sheet: {sheet_path}")

if __name__ == "__main__":
    main()
