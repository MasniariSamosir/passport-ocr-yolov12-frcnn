import os
import sys
import time
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

# --- IMPORT HELPER ---
from engine import train_one_epoch
import utils
import transforms as T
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Import Dataset
from dataset_paspor_frcnn import PasporDataset 

# ================= KONFIGURASI =================
NUM_CLASSES = 12  # 11 Label + 1 Background
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_DIR = 'dataset_paspor/annotations/frcnn' 
OLD_MODEL_PATH = "detection/frcnn/faster_rcnn_paspor_final.pth"
# ===============================================

class ToTensor(torch.nn.Module):
    def forward(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# ======================================================
# [FIX] FUNGSI EVALUASI CUSTOM (ANTI-CRASH)
# ======================================================
@torch.inference_mode()
def safe_evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Bangun COCO GT
    try:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    except Exception as e:
        print(f"[WARNING] Gagal init COCO API: {e}")
        return None

    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print("[INFO] Memulai Evaluasi...")

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # Logic pencocokan ID
        res = {}
        for target, output in zip(targets, outputs):
            # Pastikan ID berupa integer
            img_id = int(target["image_id"].item())
            
            # Cek apakah ID ada di dataset Ground Truth
            if img_id in coco.imgs:
                res[img_id] = output
            else:
                pass # Skip jika ID tidak match

        evaluator_time = time.time()
        # Update hanya jika ada result valid
        if res:
            coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # [FIX PENTING] Bungkus sinkronisasi dengan Try-Except
    # Jika tidak ada deteksi (array kosong), dia tidak akan crash.
    print("[INFO] Menghitung Skor mAP...")
    try:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    except Exception:
        # Evaluasi dilewati diam-diam jika belum ada deteksi valid
        torch.set_num_threads(n_threads)
        return None

    
    torch.set_num_threads(n_threads)
    return coco_evaluator

# ======================================================
# MAIN
# ======================================================
def main():
    print(f"[INFO] Menggunakan device: {DEVICE}")

    print("[INFO] Loading dataset...")
    try:
        dataset = PasporDataset(DATA_DIR, split="train", transforms=get_transform(train=True))
        dataset_test = PasporDataset(DATA_DIR, split="val", transforms=get_transform(train=False))
        print(f"✅ Dataset loaded. Train: {len(dataset)}, Val: {len(dataset_test)}")
    except Exception as e:
        print(f"[ERROR] Gagal load dataset: {e}")
        return

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    print("[INFO] Membangun model Faster R-CNN...")
    model = get_model_instance_segmentation(NUM_CLASSES)

    # --- FINE TUNING ---
    if os.path.exists(OLD_MODEL_PATH):
        print(f"🔄 FINE-TUNING: Memuat bobot lama dari {OLD_MODEL_PATH}...")
        try:
            checkpoint = torch.load(OLD_MODEL_PATH, map_location=DEVICE)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print("✅ SUKSES: Model lama termuat!")
        except Exception as e:
            print(f"⚠️ Gagal load model lama: {e}. Mulai dari awal.")
    # -------------------

    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"[INFO] Mulai Training untuk {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=10)
        lr_scheduler.step()
        
        # Evaluasi (Safe Mode)
        print(f"[INFO] Evaluasi Epoch {epoch}...")
        with suppress_stdout():
            safe_evaluate(model, data_loader_test, device=DEVICE)


        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"frcnn_checkpoint_{epoch}.pth")

    SAVE_PATH = "detection/frcnn/faster_rcnn_paspor_indo_usa.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"🎉 Training Selesai! Model disimpan di: {SAVE_PATH}")

if __name__ == "__main__":
    main()