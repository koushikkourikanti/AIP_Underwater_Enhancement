
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
import math

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data/lsui/original_split/LSUI/test"
WATER_DIR = ROOT / "outputs/water_net_400"
USHAPE_DIR = ROOT / "outputs/u_shape_400"

BASE = ROOT / "outputs/comparisons/full_400"
INDIVIDUAL = BASE / "individual"
SHEETS = BASE / "contact_sheets"
SLIDES = BASE / "ppt_slides"

for folder in [INDIVIDUAL, SHEETS, SLIDES]:
    folder.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

LABELS = [
    "Original LSUI",
    "Water-Net Baseline",
    "U-Shape Transformer"
]

NAVY = "#17243B"
MUTED = "#58677C"
BG = "#F3F6FB"
WHITE = "#FFFFFF"


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def input_images():
    return {
        p.stem: p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }


def paths_for(stem, inputs):
    return [
        inputs[stem],
        WATER_DIR / f"{stem}.png",
        USHAPE_DIR / f"{stem}.png"
    ]


def paste_contained(canvas, path, area):
    x, y, w, h = area

    with Image.open(path) as img:
        image = ImageOps.contain(
            img.convert("RGB"),
            (w, h),
            Image.Resampling.LANCZOS
        )

    px = x + (w - image.width) // 2
    py = y + (h - image.height) // 2
    canvas.paste(image, (px, py))


def create_comparison(stem, paths):
    W, H = 1500, 580
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (35, 24),
        f"LSUI IMAGE {stem}  |  MODEL COMPARISON",
        font=font(30, True),
        fill=NAVY
    )

    card_w = 466
    gap = 16
    start_x = 35

    for i, (label, path) in enumerate(zip(LABELS, paths)):
        x = start_x + i * (card_w + gap)

        draw.rounded_rectangle(
            (x, 80, x + card_w, 525),
            radius=15,
            fill=WHITE
        )

        draw.text(
            (x + 18, 100),
            label,
            font=font(23, True),
            fill=NAVY
        )

        paste_contained(
            canvas,
            path,
            (x + 12, 145, card_w - 24, 355)
        )

    draw.text(
        (35, 545),
        "CSC 8260 | Advanced Image Processing | LSUI Test Set",
        font=font(18),
        fill=MUTED
    )

    outfile = INDIVIDUAL / f"{stem}_comparison.png"
    canvas.save(outfile, optimize=True)
    return outfile


def create_contact_sheets(files):
    per_sheet = 10

    for start in range(0, len(files), per_sheet):
        batch = files[start:start + per_sheet]

        thumb_w, thumb_h = 850, 329
        gap = 16
        cols = 2
        rows = math.ceil(len(batch) / cols)

        sheet = Image.new(
            "RGB",
            (
                cols * thumb_w + (cols + 1) * gap,
                rows * thumb_h + (rows + 1) * gap
            ),
            BG
        )

        for j, path in enumerate(batch):
            with Image.open(path) as img:
                thumb = img.convert("RGB").resize(
                    (thumb_w, thumb_h),
                    Image.Resampling.LANCZOS
                )

            x = gap + (j % cols) * (thumb_w + gap)
            y = gap + (j // cols) * (thumb_h + gap)
            sheet.paste(thumb, (x, y))

        number = start // per_sheet + 1

        sheet.save(
            SHEETS / f"contact_sheet_{number:02d}.jpg",
            quality=92
        )

    print(f"Contact sheets created: {math.ceil(len(files) / per_sheet)}")


def create_ppt_slide(stem, paths):
    W, H = 1920, 1080

    slide = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(slide)

    draw.rectangle((0, 0, W, 18), fill="#2879B8")

    draw.text(
        (70, 64),
        "UNDERWATER IMAGE ENHANCEMENT",
        font=font(53, True),
        fill=NAVY
    )

    draw.text(
        (72, 139),
        f"LSUI Test Image {stem} | Pretrained Model Comparison",
        font=font(29),
        fill=MUTED
    )

    card_w = 570
    card_y = 235
    card_h = 675
    gap = 34
    start_x = 71

    for i, (label, path) in enumerate(zip(LABELS, paths)):
        x = start_x + i * (card_w + gap)

        draw.rounded_rectangle(
            (x, card_y, x + card_w, card_y + card_h),
            radius=20,
            fill=WHITE
        )

        draw.text(
            (x + 22, card_y + 25),
            label,
            font=font(30, True),
            fill=NAVY
        )

        paste_contained(
            slide,
            path,
            (x + 15, card_y + 95, card_w - 30, 530)
        )

    draw.text(
        (72, 965),
        "Georgia State University | CSC 8260 | Advanced Image Processing",
        font=font(24),
        fill=MUTED
    )

    draw.text(
        (72, 1005),
        "Visual comparison only - quantitative evaluation performed separately",
        font=font(20),
        fill=MUTED
    )

    outfile = SLIDES / f"LSUI_{stem}_presentation.png"
    slide.save(outfile, optimize=True)

    print(f"PPT-ready image saved: {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide-id",
        default="1003",
        help="Image ID for a PowerPoint slide"
    )
    parser.add_argument(
        "--slide-only",
        action="store_true",
        help="Generate only one PPT-ready image"
    )
    args = parser.parse_args()

    inputs = input_images()
    stems = sorted(inputs)

    if len(stems) != 400:
        raise RuntimeError(
            f"Expected 400 LSUI test images, found {len(stems)}"
        )

    for stem in stems:
        missing = [
            p for p in paths_for(stem, inputs)
            if not p.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing files for {stem}: {missing}"
            )

    print("All 400 input/Water-Net/U-Shape sets verified.")

    if not args.slide_only:
        files = []

        for index, stem in enumerate(stems, 1):
            file = create_comparison(
                stem, paths_for(stem, inputs)
            )
            files.append(file)
            print(f"[{index}/400] {stem} comparison saved")

        create_contact_sheets(files)

    if args.slide_id not in inputs:
        raise ValueError(
            f"Image ID {args.slide_id} not found."
        )

    create_ppt_slide(
        args.slide_id,
        paths_for(args.slide_id, inputs)
    )

    print("\nCOMPARISON GENERATION COMPLETED")


if __name__ == "__main__":
    main()
