import json

with open("models/train_unet.py", "r", encoding="utf-8") as f:
    train_content = f.read()

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# The writefile cell is cell index 13 or 14. We can just find it by looking for %%writefile
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "%%writefile models/train_unet.py\\n" in cell.get("source", []):
        lines = train_content.splitlines(True) # Keep newlines
        nb["cells"][i]["source"] = ["%%writefile models/train_unet.py\\n"] + lines
        break

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook writefile cell fixed.")
