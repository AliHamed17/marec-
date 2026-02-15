"""Extract code cells from the notebook and run as a single script."""
import json, os, sys

base = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(base, "MARec_Fast_Experiments.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

script_path = os.path.join(base, "run_fast_experiments.py")
with open(script_path, "w", encoding="utf-8") as f:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            f.write(code + "\n\n")

print(f"Extracted {sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells to {script_path}")
