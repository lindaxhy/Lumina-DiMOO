#!/usr/bin/env python3
"""Render the fixed prompt #10 / seed 0 extended-model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SUBJECTS = ("dog", "cat2", "backpack", "colorful_sneaker")


def square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    edge = min(image.size)
    left = (image.width - edge) // 2
    top = (image.height - edge) // 2
    return image.crop((left, top, left + edge, top + edge)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, text_font) -> None:
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2
    draw.text((x, y), text, fill="#111111", font=text_font)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--bagel-root", type=Path, required=True)
    parser.add_argument("--sdxl-root", type=Path, required=True)
    parser.add_argument("--z-image-root", type=Path, required=True)
    parser.add_argument(
        "--base-img2img",
        action="store_true",
        help="Label SDXL and Z-Image columns as base img2img instead of LoRA.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=320)
    args = parser.parse_args()

    roots = {
        "qwen": args.qwen_root,
        "bagel": args.bagel_root,
        "sdxl": args.sdxl_root,
        "z_image": args.z_image_root,
    }
    columns = (
        ("Reference", None),
        ("Qwen zero-shot", "qwen"),
        ("BAGEL zero-shot", "bagel"),
        ("SDXL base img2img" if args.base_img2img else "SDXL LoRA", "sdxl"),
        ("Z-Image base img2img" if args.base_img2img else "Z-Image LoRA", "z_image"),
    )
    cell = args.cell_size
    label_width = 190
    header_height = 76
    canvas = Image.new(
        "RGB",
        (label_width + len(columns) * cell, header_height + len(SUBJECTS) * cell),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = font(25)
    row_font = font(23)
    long_row_font = font(17)

    for column_index, (label, _) in enumerate(columns):
        x0 = label_width + column_index * cell
        centered(draw, (x0, 0, x0 + cell, header_height), label, header_font)

    for row_index, subject in enumerate(SUBJECTS):
        y0 = header_height + row_index * cell
        centered(
            draw,
            (0, y0, label_width, y0 + cell),
            subject,
            long_row_font if subject == "colorful_sneaker" else row_font,
        )
        for column_index, (_, model_key) in enumerate(columns):
            if model_key is None:
                image_path = args.reference_root / subject / "00.jpg"
            else:
                image_path = roots[model_key] / subject / "10_0.png"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            tile = square(Image.open(image_path), cell)
            canvas.paste(tile, (label_width + column_index * cell, y0))

    line = "#d8d8d8"
    draw.line((label_width, 0, label_width, canvas.height), fill=line, width=2)
    draw.line((0, header_height, canvas.width, header_height), fill=line, width=2)
    for index in range(1, len(columns)):
        x = label_width + index * cell
        draw.line((x, 0, x, canvas.height), fill=line, width=1)
    for index in range(1, len(SUBJECTS)):
        y = header_height + index * cell
        draw.line((0, y, canvas.width, y), fill=line, width=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
