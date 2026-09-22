import json

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Restore cell 9
cell_9_source = [
    "# Yedek plan: index.html'i parse edip dosyaları tek tek indir (yukarıdaki wget -r başarısız olursa bu hücreyi çalıştırın)\n",
    "import requests\n",
    "from bs4 import BeautifulSoup\n",
    "from urllib.parse import urljoin\n",
    "import os\n",
    "\n",
    "def download_split(split, kind):\n",
    "    index_url = f\"{BASE}/{split}/{kind}/index.html\"\n",
    "    out_dir = f\"{DATA_ROOT}/{split}/{kind}\"\n",
    "    os.makedirs(out_dir, exist_ok=True)\n",
    "    r = requests.get(index_url, timeout=30)\n",
    "    soup = BeautifulSoup(r.text, \"html.parser\")\n",
    "    links = [a[\"href\"] for a in soup.find_all(\"a\", href=True) if a[\"href\"].lower().endswith((\".tif\", \".tiff\"))]\n",
    "    print(f\"{split}/{kind}: {len(links)} dosya bulundu\")\n",
    "    for href in links:\n",
    "        file_url = urljoin(index_url, href)\n",
    "        fname = os.path.join(out_dir, os.path.basename(href))\n",
    "        if os.path.exists(fname):\n",
    "            continue\n",
    "        data = requests.get(file_url, timeout=60).content\n",
    "        with open(fname, \"wb\") as f:\n",
    "            f.write(data)\n",
    "    return len(links)\n",
    "\n",
    "# Sadece dosya sayısı 0 çıkan split/kind kombinasyonları için çalıştırın, örnek:\n",
    "# download_split(\"train\", \"sat\")\n",
    "# download_split(\"train\", \"map\")"
]
nb["cells"][9]["source"] = cell_9_source

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Restored cell 9")
