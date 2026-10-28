
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "pilot" / "input"
WATER = ROOT / "outputs" / "water_net"
USHAPE = ROOT / "outputs" / "u_shape"
NORMAL = ROOT / "outputs" / "u_shape_normalized"
DEST = ROOT / "outputs" / "comparisons"
DEST.mkdir(parents=True, exist_ok=True)

PANEL_W, PANEL_H = 260, 260
GAP = 16
TOP = 54
ROW_H = TOP + PANEL_H + 20

LABELS = [
    "Original LSUI",
    "Water-Net",
    "U-Shape clipped",
    "U-Shape normalized"
]

try:
    font = ImageFont.truetype("arial.ttf", 17)
    small_font = ImageFont.truetype("arial.ttf", 15)
except OSError:
    font = ImageFont.load_default()
    small_font = font


def make_row(stem):
    paths = [
        INPUT / f"{stem}.jpg",
        WATER / f"{stem}.png",
        USHAPE / f"{stem}.png",
        NORMAL / f"{stem}.png",
    ]

    if not all(p.exists() for p in paths):
        print(f"Skipping {stem}: missing output")
        return None

    width = 4 * PANEL_W + 5 * GAP
    canvas = Image.new(
        "RGB", (width, ROW_H), "white"
    )
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (GAP, 6), f"LSUI Image {stem}",
        font=font, fill=(18, 42, 76)
    )

    for i, (path, label) in enumerate(zip(paths, LABELS)):
        x = GAP + i * (PANEL_W + GAP)
        draw.text(
            (x, 32), label,
            font=small_font, fill=(40, 40, 40)
        )

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            resized = ImageOps.contain(
                rgb, (PANEL_W, PANEL_H),
                Image.Resampling.LANCZOS
            )

        px = x + (PANEL_W - resized.width) // 2
        py = TOP + (PANEL_H - resized.height) // 2
        canvas.paste(resized, (px, py))

    canvas.save(DEST / f"{stem}_four_way.png")
    return canvas


def main():
    rows = []

    for path in sorted(INPUT.glob("*.jpg")):
        panel = make_row(path.stem)
        if panel is not None:
            rows.append(panel)

    if not rows:
        raise RuntimeError("No complete image comparisons found.")

    sheet = Image.new(
        "RGB",
        (rows[0].width, len(rows) * ROW_H),
        (230, 235, 240)
    )

    for i, panel in enumerate(rows):
        sheet.paste(panel, (0, i * ROW_H))

    output = DEST / "all_models_comparison.png"
    sheet.save(output)

    print(f"Compared {len(rows)} images.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
