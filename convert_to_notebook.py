"""Convert MARec_Full_Experiments.py to proper .ipynb notebook."""
import json, re, os

base = r"c:\Users\alih1\Downloads\files (1)"
input_path = os.path.join(base, "MARec_Full_Experiments.py")
output_path = os.path.join(base, "MARec_Full_Experiments.ipynb")

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split on # %% markers
raw_cells = re.split(r'^# %%.*$', content, flags=re.MULTILINE)
markers = re.findall(r'^# %%(.*)$', content, flags=re.MULTILINE)

cells = []
for i, cell_content in enumerate(raw_cells):
    cell_content = cell_content.strip()
    if not cell_content:
        continue

    if i > 0 and i - 1 < len(markers) and "[markdown]" in markers[i - 1]:
        lines = cell_content.split("\n")
        md_lines = []
        for line in lines:
            if line.startswith("# "):
                md_lines.append(line[2:])
            elif line == "#":
                md_lines.append("")
            else:
                md_lines.append(line)
        source = [l + "\n" for l in md_lines]
        if source:
            source[-1] = source[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        })
    else:
        lines = cell_content.split("\n")
        source = [l + "\n" for l in lines]
        if source:
            source[-1] = source[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source
        })

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "A100",
            "machine_shape": "hm"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created {output_path} with {len(cells)} cells")
