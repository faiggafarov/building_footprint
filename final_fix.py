import json

with open("models/train_unet.py", "r", encoding="utf-8") as f:
    train_content = f.read()

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# The first 14 cells (up to !python mass_tile_generator.py) are fine. Let's find the index of "## 5)"
idx_5 = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "5) Eğitim Script" in "".join(cell.get("source", [])):
        idx_5 = i
        break

if idx_5 != -1:
    new_cells = nb["cells"][:idx_5]
else:
    new_cells = nb["cells"]

# Append Cell 5 Markdown
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5) Eğitim Scriptindeki Hardcoded Parametreleri Düzelt (Colab Uyumu)\n",
        "\n",
        "Bu hücre, GitHub-dan klonlanmış olan `train_unet.py` dosyasını Colab mühitinə uyğun, Early Stopping və Massachusetts verilənləri (3 kanal) üçün uyğunlaşdırılmış versiya ilə əvəz edir."
    ]
})

# Append Writefile Cell
lines = [line + "\\n" for line in train_content.split("\\n")]
if lines and lines[-1] == "\\n":
    lines = lines[:-1]
    
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["%%writefile models/train_unet.py\n"] + lines
})

# Append Cell 6 Markdown
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6) Eğitimi Başlat\n",
        "\n",
        "Artık modelimizi Massachusetts verisi ile eğitebiliriz."
    ]
})

# Append Cell 6 Code
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!python models/train_unet.py"]
})

# Append Cell 7 Markdown
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7) Sonuçları Analiz Et (Loss ve Accuracy/Dice)\n",
        "\n",
        "Eğitim sırasında kaydedilen log dosyasını oxuyaraq Train Loss, Validation Dice ve Validation IoU metriklerini analiz edirik."
    ]
})

# Append Cell 7 Code
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "\n",
        "log_path = 'logs/unet_metrics.csv'\n",
        "if os.path.exists(log_path):\n",
        "    df = pd.read_csv(log_path)\n",
        "    print(\"--- Eğitim Tablosu ---\")\n",
        "    display(df)\n",
        "    \n",
        "    plt.figure(figsize=(12, 5))\n",
        "    \n",
        "    # Loss Plot\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.plot(df['Epoch'], df['Train Loss'], marker='o', label='Train Loss', color='red')\n",
        "    plt.title('Training Loss')\n",
        "    plt.xlabel('Epoch')\n",
        "    plt.ylabel('Loss')\n",
        "    plt.legend()\n",
        "    plt.grid(True)\n",
        "    \n",
        "    # Accuracy / Metrics Plot\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.plot(df['Epoch'], df['Val Dice'], marker='s', label='Val Dice', color='blue')\n",
        "    plt.plot(df['Epoch'], df['Val IoU'], marker='^', label='Val IoU', color='green')\n",
        "    plt.title('Validation Metrics')\n",
        "    plt.xlabel('Epoch')\n",
        "    plt.ylabel('Score')\n",
        "    plt.legend()\n",
        "    plt.grid(True)\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"Henüz log dosyası oluşmamış. Lütfen önce 6. adımdaki eğitimi başlatın.\")"
    ]
})

nb["cells"] = new_cells

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Final fix completed successfully!")
