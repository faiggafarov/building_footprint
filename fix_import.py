import json

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "%%writefile models/train_unet.py" in "".join(cell.get("source", [])):
        source = cell["source"]
        for j, line in enumerate(source):
            if "sys.path.append" in line:
                source[j] = line.replace("sys.path.append", "sys.path.insert(0, ").rstrip() + ")\n"
        nb["cells"][i]["source"] = source
        break

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("ModuleNotFoundError fix applied.")
