import argparse
import os

from PIL import Image
import re


def images_to_pdf(img_dir, output_pdf):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    files = sorted(
        [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(exts)
        ],
        key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[0])
        if re.findall(r"\d+", os.path.basename(f))
        else float("inf"),
    )

    if not files:
        print(f"No images found in {img_dir}.")
        return

    imgs = []
    for fp in files:
        img = Image.open(fp)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        imgs.append(img)

    first, rest = imgs[0], imgs[1:]
    first.save(output_pdf, "PDF", save_all=True, append_images=rest)
    print(f"PDF created: {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all images in a folder into a single PDF file."
    )
    parser.add_argument("img_dir", help="Directory containing image files")
    parser.add_argument(
        "-o", "--output", default="output.pdf", help="Output PDF file path"
    )
    args = parser.parse_args()

    images_to_pdf(args.img_dir, args.output)
