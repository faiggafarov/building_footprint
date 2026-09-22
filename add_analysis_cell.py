import json

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Add Markdown for analysis
analysis_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7) Sonuçları Analiz Et (Loss ve Accuracy/Dice)\n",
        "\n",
        "Eğitim sırasında kaydedilen log dosyasını oxuyaraq Train Loss, Validation Dice ve Validation IoU metriklerini analiz edirik."
    ]
}
nb["cells"].append(analysis_md)

analysis_code = {
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
}
nb["cells"].append(analysis_code)

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Analysis cells added.")
