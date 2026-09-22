import json

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Change the git clone cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "git clone" in "".join(cell.get("source", [])):
        cell["source"] = [
            "# TODO: Aşağıdakı URL-i öz GitHub repository URL-nizlə əvəz edin\n",
            "!git clone https://github.com/SizinHesabiniz/footprint.git\n",
            "%cd footprint\n"
        ]
        break

# Remove the %%writefile cell
new_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "%%writefile models/train_unet.py" in src:
        # Instead of writing the file, just remind user the file is already perfect
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5) Eğitim Scripti Hazırdır\n",
                "\n",
                "Biz artıq `models/train_unet.py` faylını öz repository-mizdə Massachusetts dataset-ə və Early Stopping-ə uyğun düzəltmişik. Əlavə heç nəyə ehtiyac yoxdur!"
            ]
        })
    elif "5) Eğitim Scriptindeki Hardcoded Parametreleri Düzelt" in src:
        pass # Skip the markdown that belonged to the writefile cell
    else:
        new_cells.append(cell)

nb["cells"] = new_cells

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook cleaned up.")
