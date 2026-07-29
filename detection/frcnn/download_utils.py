import os
import requests

# Lokasi target download (Folder detection/frcnn)
TARGET_DIR = "detection/frcnn"

# Daftar file yang wajib ada untuk training Faster R-CNN
FILES_TO_DOWNLOAD = [
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py"
]

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)
    
    print(f"⬇️ Downloading {filename} ...")
    try:
        r = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved to {file_path}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

if __name__ == "__main__":
    print("--- START DOWNLOADING HELPER SCRIPTS ---")
    for url in FILES_TO_DOWNLOAD:
        download_file(url, TARGET_DIR)
    print("--- DONE ---")
    print("Sekarang coba jalankan training lagi!")