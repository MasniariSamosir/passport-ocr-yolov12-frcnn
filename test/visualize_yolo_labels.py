import cv2
import argparse

def load_classes(classes_path):
    """Baca daftar nama kelas dari file classes.txt"""
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def visualize_yolo_labels(image_path, label_path, classes_path):
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Gambar tidak ditemukan: {image_path}")
    orig_h, orig_w = img.shape[:2]

    # Load nama kelas
    class_names = load_classes(classes_path)

    # Load file label
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"⚠️ Format salah di baris: {line}")
            continue

        cls_id, x_center, y_center, width, height = map(float, parts)
        cls_id = int(cls_id)

        # Konversi koordinat YOLO → pixel
        x_center *= orig_w
        y_center *= orig_h
        width *= orig_w
        height *= orig_h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Warna & label
        color = (0, 255, 0) if cls_id % 2 == 0 else (255, 100, 0)
        label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        # Gambar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---- 🔧 Resize gambar untuk tampilan ----
    max_width, max_height = 900, 700  # ukuran tampilan maksimum
    scale_w = max_width / orig_w
    scale_h = max_height / orig_h
    scale = min(1.0, scale_w, scale_h)  # pertahankan aspek rasio

    if scale < 1.0:
        img_display = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
    else:
        img_display = img

    # ---- 🖼️ Tampilkan ----
    cv2.imshow("📦 YOLO Label Visualization (press any key to close)", img_display)
    print(f"\n✅ Ukuran asli: {orig_w}x{orig_h} | Ditampilkan: {img_display.shape[1]}x{img_display.shape[0]}")
    print("✅ Tekan tombol apa pun untuk menutup jendela...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisasi label YOLO pada gambar paspor (auto-resize window)")
    parser.add_argument("--image", required=True, help="Path ke gambar (.png/.jpg)")
    parser.add_argument("--label", required=True, help="Path ke file label (.txt)")
    parser.add_argument("--classes", required=True, help="Path ke file classes.txt")

    args = parser.parse_args()
    visualize_yolo_labels(args.image, args.label, args.classes)
