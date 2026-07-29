from ultralytics import YOLO
import os

def main():

    data_yaml = "dataset_paspor/data.yaml"

    model_path = "detection/yolo/yolov12n.pt"

    if not os.path.exists(model_path):
        print("[ERROR] File yolov12n.pt tidak ditemukan:", model_path)
        return

    print("[INFO] Memuat model:", model_path)
    model = YOLO(model_path)

    print("[INFO] Mulai Training… (GPU MODE)")
    model.train(
    data=data_yaml,
    imgsz=640,
    epochs=150,
    batch=4,
    workers=4,
    device=0,
    amp=False,          # <--- WAJIB
    pretrained=False,   # <--- cegah YOLO11 fallback
)


    print("\n===================================")
    print("TRAINING COMPLETED (GPU MODE)")
    print("Model disimpan di:")
    print("runs_yolov12_manual/yolov12_paspor_manual")
    print("===================================\n")


if __name__ == "__main__":
    main()
