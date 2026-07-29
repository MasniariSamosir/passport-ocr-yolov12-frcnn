import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image

class PasporDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.transforms = transforms

        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "Annotations")

        list_path = os.path.join(root, f"{split}.txt")
        with open(list_path) as f:
            self.ids = [x.strip() for x in f.readlines() if x.strip()]

        self.classes = [
            "photo", "given_names", "surname", "passport_number",
            "date_of_birth", "date_of_issue", "date_of_expiry",
            "sex", "nationality", "place_of_birth", "mrz"
        ]
        self.class_to_idx = {cls.lower(): i + 1 for i, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        file_id = self.ids[idx].strip()

        if not file_id:
            raise ValueError(
                f"[DATASET ERROR] file_id kosong pada index {idx}"
            )

        # === LOAD IMAGE (FIXED) ===
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(self.img_dir, file_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(
                f"[DATASET ERROR] Gambar tidak ditemukan: {file_id}"
            )

        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size

        # === LOAD XML ===
        ann_path = os.path.join(self.ann_dir, file_id + ".xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes, labels = [], []

        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in self.class_to_idx:
                continue

            label_id = self.class_to_idx[name]
            bnd = obj.find("bndbox")

            xmin = max(0, float(bnd.find("xmin").text))
            ymin = max(0, float(bnd.find("ymin").text))
            xmax = min(w_img, float(bnd.find("xmax").text))
            ymax = min(h_img, float(bnd.find("ymax").text))

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        if len(boxes) == 0:
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
