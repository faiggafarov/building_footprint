import json

with open("building_footprint_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep cells up to index 11
new_cells = nb["cells"][:12]

tile_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4) Massachusetts Verisini Modele Uygun Hale Getir (Tiling)\n",
        "\n",
        "Repodaki `tile_generator.py` Vegas (GeoJSON) verisi için yazılmış. Massachusetts verisi (TIF) için özel bir tile oluşturucu script yazıp çalıştırıyoruz. Bu script görüntüleri 256x256 boyutunda kırpıp `train_unet.py`'nin beklediği `.npz` (C,H,W) formatında kaydeder."
    ]
}
new_cells.append(tile_md)

tile_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%%writefile mass_tile_generator.py\n",
        "import os, glob\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "from tqdm import tqdm\n",
        "\n",
        "def process_massachusetts():\n",
        "    sat_dir = \"raw_mass_buildings/train/sat\"\n",
        "    map_dir = \"raw_mass_buildings/train/map\"\n",
        "    out_img = \"data/mass_tiles_npz/images\"\n",
        "    out_lbl = \"data/mass_tiles_npz/labels\"\n",
        "    os.makedirs(out_img, exist_ok=True)\n",
        "    os.makedirs(out_lbl, exist_ok=True)\n",
        "    \n",
        "    sat_files = sorted(glob.glob(f\"{sat_dir}/*.tiff\"))[:10]  # Sadece ilk 10 resmi örnek olarak al (hızlı test için)\n",
        "    for sat_path in tqdm(sat_files, desc=\"Tiling Images\"):\n",
        "        basename = os.path.basename(sat_path).replace(\".tiff\", \".tif\")\n",
        "        map_path = os.path.join(map_dir, basename)\n",
        "        if not os.path.exists(map_path): continue\n",
        "        \n",
        "        img = np.array(Image.open(sat_path)) # H, W, 3\n",
        "        mask = np.array(Image.open(map_path)) # H, W\n",
        "        mask = (mask > 127).astype(np.uint8)\n",
        "        \n",
        "        H, W = img.shape[:2]\n",
        "        size = 256\n",
        "        for y in range(0, H - size + 1, size):\n",
        "            for x in range(0, W - size + 1, size):\n",
        "                img_tile = img[y:y+size, x:x+size]\n",
        "                mask_tile = mask[y:y+size, x:x+size]\n",
        "                if mask_tile.sum() == 0 and np.random.rand() > 0.1: continue # Boş tile'ların %90'ını at\n",
        "                \n",
        "                img_tile = img_tile.transpose(2, 0, 1) # C, H, W\n",
        "                tile_name = f\"{basename.replace('.tif','')}_{y}_{x}.npz\"\n",
        "                np.savez_compressed(os.path.join(out_img, tile_name), arr_0=img_tile)\n",
        "                np.savez_compressed(os.path.join(out_lbl, tile_name), arr_0=mask_tile)\n",
        "\n",
        "process_massachusetts()\n",
        "print('Tiling işlemi bitti!')\n"
    ]
}
new_cells.append(tile_code)

run_tile = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!python mass_tile_generator.py"]
}
new_cells.append(run_tile)

patch_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5) Eğitim Scriptindeki Hardcoded Parametreleri Düzelt\n",
        "\n",
        "`models/train_unet.py` içinde kanal sayısı (8'den 3'e RGB için) ve dosya yolları hardcoded (sabit) olarak yazılmış. Bunları yeni oluşturduğumuz Massachusetts tile yollarına uyacak şekilde değiştiriyoruz."
    ]
}
new_cells.append(patch_md)

patch_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import re\n",
        "with open('models/train_unet.py', 'r') as f:\n",
        "    content = f.read()\n",
        "\n",
        "# 8 kanal (Vegas) yerine 3 kanal (Massachusetts RGB) yap\n",
        "content = re.sub(r'in_channels=8', 'in_channels=3', content)\n",
        "# Hardcoded dizinleri güncelle\n",
        "content = re.sub(r'IMAGE_DIR\\s*=\\s*\"[^\"]+\"', 'IMAGE_DIR = \"data/mass_tiles_npz/images\"', content)\n",
        "content = re.sub(r'LABEL_DIR\\s*=\\s*\"[^\"]+\"', 'LABEL_DIR = \"data/mass_tiles_npz/labels\"', content)\n",
        "# Hızlı test için epoch sayısını düşürelim (isteğe bağlı)\n",
        "content = re.sub(r'EPOCHS\\s*=\\s*\\d+', 'EPOCHS = 5', content)\n",
        "content = re.sub(r'BATCH_SIZE\\s*=\\s*\\d+', 'BATCH_SIZE = 16', content)\n",
        "\n",
        "with open('models/train_unet.py', 'w') as f:\n",
        "    f.write(content)\n",
        "print('train_unet.py güncellendi!')"
    ]
}
new_cells.append(patch_code)

train_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6) Eğitimi Başlat\n",
        "\n",
        "Artık modelimizi Massachusetts verisi ile eğitebiliriz."
    ]
}
new_cells.append(train_md)

train_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!python models/train_unet.py"
    ]
}
new_cells.append(train_code)

nb["cells"] = new_cells

with open("building_footprint_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated.")
