import json

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "import re" in src:
            print(f"Found at {i}")
            with open("models/train_unet.py", "r", encoding="utf-8") as f2:
                lines = f2.readlines()
            new_source = ["%%writefile models/train_unet.py\n"] + lines
            nb["cells"][i]["source"] = new_source

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Colab rewrite complete.")
