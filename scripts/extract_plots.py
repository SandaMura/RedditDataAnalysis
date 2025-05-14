import argparse
import base64
import os

import nbformat


def extract_plots(notebook_path, out_dir):
    notebook_path = os.path.abspath(notebook_path)
    out_dir = os.path.abspath(out_dir)

    # Load the notebook
    nb = nbformat.read(notebook_path, as_version=4)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    # Iterate through all cells and their outputs
    for cell_idx, cell in enumerate(nb.cells):
        for out_idx, output in enumerate(cell.get("outputs", [])):
            # Look for image/png data
            data = output.get("data", {})
            png = data.get("image/png")
            if png:
                # Handle if the PNG data is a list of lines or a single string
                if isinstance(png, list):
                    b64_data = "".join(png)
                else:
                    b64_data = png

                img_data = base64.b64decode(b64_data)
                filename = f"fig_{cell_idx}_{out_idx}.png"
                filepath = os.path.join(out_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(img_data)

                print(f"Saved: {filepath}")
                count += 1

    print(f"Total plots extracted: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract all embedded plots from a Jupyter notebook and save them as PNG files."
    )
    parser.add_argument("notebook", help="Path to the .ipynb file")
    parser.add_argument(
        "-o", "--outdir", default="figures", help="Directory to save extracted figures"
    )
    args = parser.parse_args()

    extract_plots(args.notebook, args.outdir)
