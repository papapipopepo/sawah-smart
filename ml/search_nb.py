import json, sys
sys.stdout.reconfigure(encoding="utf-8")

nb_path = r"E:\THESIS\code\Thesis_3Class (4).ipynb"
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

new_src = [
    "import os, pandas as pd\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# RESUME MODE — set FORCE_REEXTRACT = True jika dataset/fitur berubah.\n",
    "# Set False jika hanya ingin lanjut dari CSV yang sudah ada di Drive.\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "FORCE_REEXTRACT = True   # ← ganti False kalau mau resume dari CSV lama\n",
    "\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "CSV_DRIVE = \"/content/drive/MyDrive/features_3class.csv\"\n",
    "\n",
    "RESUME = os.path.exists(CSV_DRIVE) and not FORCE_REEXTRACT\n",
    "\n",
    "if RESUME:\n",
    "    df_all = pd.read_csv(CSV_DRIVE)\n",
    "    print(f\"✅ RESUME MODE: fitur dimuat dari Drive ({len(df_all)} baris)\")\n",
    "    print(\"   Langsung jalankan cell dari bagian Definisi Kelas (CLASS_ORDER) ke bawah.\")\n",
    "    print(f\"   Lewati cell: upload zip, ekstraksi gambar, dan ekstraksi fitur.\")\n",
    "    display(df_all.groupby([\"split\",\"label\"]).size().unstack(fill_value=0))\n",
    "else:\n",
    "    if FORCE_REEXTRACT:\n",
    "        print(\"🔄 FORCE REEXTRACT: mengabaikan CSV lama, jalankan ekstraksi fitur dari awal.\")\n",
    "    else:\n",
    "        print(\"⏭️  FRESH RUN: features_3class.csv belum ada di Drive.\")\n",
    "    print(\"   Jalankan cell berikutnya secara normal (upload zip → ekstraksi fitur).\")\n",
]

nb["cells"][5]["source"] = new_src
print("Cell 5 updated OK")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved.")
