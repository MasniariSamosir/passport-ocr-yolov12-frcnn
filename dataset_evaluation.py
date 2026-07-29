from typing import List, Dict

def compute_dataset_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """
    Menghitung rata-rata evaluasi metrik dari seluruh dataset.

    metrics_list: list hasil compute_metrics_for_passport
    """

    if not metrics_list:
        return {}

    total = {
        "car": 0.0,
        "war": 0.0,
        "wer": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }

    n = len(metrics_list)

    for m in metrics_list:
        total["car"]       += m.get("car", 0)
        total["war"]       += m.get("war", 0)
        total["wer"]       += m.get("wer", 0)
        total["accuracy"]  += m.get("accuracy", 0)
        total["precision"] += m.get("precision", 0)
        total["recall"]    += m.get("recall", 0)
        total["f1"]        += m.get("f1", 0)

    # Rata-rata
    return {
        "CAR": round(total["car"] / n, 2),
        "WAR": round(total["war"] / n, 2),
        "WER": round(total["wer"] / n, 4),
        "CER": round(100 - (total["car"] / n), 2),  # CER = 100 - CAR
        "Accuracy": round(total["accuracy"] / n, 2),
        "Precision": round(total["precision"] / n, 2),
        "Recall": round(total["recall"] / n, 2),
        "F1-score": round(total["f1"] / n, 2),
        "Total_Data": n
    }

def print_dataset_report(model_name: str, metrics: Dict[str, float]):
    print(f"\nHASIL EVALUASI DATASET — {model_name}")
    print("-" * 45)
    print(f"Jumlah Dataset : {metrics['Total_Data']}")
    print(f"CAR       : {metrics['CAR']}%")
    print(f"CER       : {metrics['CER']}%")
    print(f"WAR       : {metrics['WAR']}%")
    print(f"WER       : {metrics['WER']}")
    print(f"Precision : {metrics['Precision']}%")
    print(f"Recall    : {metrics['Recall']}%")
    print(f"Accuracy  : {metrics['Accuracy']}%")
    print(f"F1-score  : {metrics['F1-score']}%")

def compute_dataset_metrics(metrics_list):
    # ⛑️ GUARD CLAUSE — WAJIB
    if not metrics_list:
        return {
            "CAR": 0.0,
            "CER": 0.0,
            "WAR": 0.0,
            "WER": 0.0,
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-score": 0.0,
            "Total_Data": 0
        }

    total = {
        "car": 0.0,
        "war": 0.0,
        "wer": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }

    n = len(metrics_list)

    for m in metrics_list:
        total["car"]       += m.get("car", 0)
        total["war"]       += m.get("war", 0)
        total["wer"]       += m.get("wer", 0)
        total["accuracy"]  += m.get("accuracy", 0)
        total["precision"] += m.get("precision", 0)
        total["recall"]    += m.get("recall", 0)
        total["f1"]        += m.get("f1", 0)

    return {
        "CAR": round(total["car"] / n, 2),
        "CER": round(100 - (total["car"] / n), 2),
        "WAR": round(total["war"] / n, 2),
        "WER": round(total["wer"] / n, 4),
        "Accuracy": round(total["accuracy"] / n, 2),
        "Precision": round(total["precision"] / n, 2),
        "Recall": round(total["recall"] / n, 2),
        "F1-score": round(total["f1"] / n, 2),
        "Total_Data": n
    }

# ============================================================
# HITUNG RATA-RATA KECEPATAN INFERENSI (31 DATASET)
# ============================================================
def compute_speed_metrics(rows):
    """
    Menghitung rata-rata waktu inferensi & FPS dari database
    """
    if not rows:
        return {
            "Avg_Time": 0.0,
            "Avg_FPS": 0.0
        }

    speeds = [r["speed"] for r in rows if r.get("speed")]
    fps_vals = [r["fps"] for r in rows if r.get("fps")]

    avg_time = sum(speeds) / len(speeds) if speeds else 0.0
    avg_fps  = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0

    return {
        "Avg_Time": round(avg_time, 4),
        "Avg_FPS": round(avg_fps, 2)
    }


# ============================================================
# MENENTUKAN MODEL PALING UNGGUL
# ============================================================
def conclude_best_model(yolo, frcnn):
    """
    Menentukan model paling unggul berdasarkan agregasi metrik evaluasi
    """

    def score(m):
        return (
            m.get("CAR", 0.0) +
            m.get("WAR", 0.0) +
            m.get("Precision", 0.0) +
            m.get("Recall", 0.0) +
            m.get("Accuracy", 0.0) +
            m.get("F1-score", 0.0)
            - (m.get("WER", 1.0) * 100)
        )

    score_yolo = score(yolo)
    score_frcnn = score(frcnn)

    print("\nSKOR AKHIR MODEL:")
    print(f"YOLOv12      : {round(score_yolo, 2)}")
    print(f"Faster R-CNN : {round(score_frcnn, 2)}")

    if score_yolo > score_frcnn:
        return "YOLOv12"
    else:
        return "Faster R-CNN"



    score_frcnn = (
        frcnn["CAR"] +
        frcnn["WAR"] +
        frcnn["Precision"] +
        frcnn["Recall"] +
        frcnn["Accuracy"] +
        frcnn["F1-score"]
        - frcnn["WER"] * 100
    )

    print("\nSKOR AKHIR MODEL:")
    print(f"YOLOv12      : {round(score_yolo, 2)}")
    print(f"Faster R-CNN : {round(score_frcnn, 2)}")

    return "YOLOv12" if score_yolo > score_frcnn else "Faster R-CNN"

    