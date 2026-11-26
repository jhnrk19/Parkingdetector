"""Szerveroldali Deep Learning tanító, kiértékelő és riportkészítő modul (YOLOv8x-OBB).

Az 1. lépés (POST /api/train) részeként fut le, és lefedi a teljes ML-életciklust:
1. Saját adatkészlet összeállítása a feltöltött videóból (képkocka-mintavétel + pszeudo-címkézés).
2. Adattisztítás (elmosódott, duplikált és szélsőséges/hibás képkockák szűrése).
3. Adatszivárgás-mentes időrendi Train / Validation / Test felosztás (70% / 15% / 15%).
4. Transfer Learning kísérletek összevetése:
   - Baseline: Előre tanított YOLOv8x-OBB (zero-shot)
   - Variáns A: Befagyasztott gerincháló (freeze=10, AdamW) – feature extraction
   - Variáns B: Teljes finomhangolás (freeze=0, SGD) – fine-tuning
5. Részletes kiértékelés az independent test halmazon (Precision, Recall, F1, mAP@50, mAP@50-95, Confusion Matrix).
6. Hibaelemzés (FP/FN vizuális összevetés) és Interpretálhatóság (Feature Activation / Grad-CAM jellegű hőtérkép).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

VEHICLE_CLASSES = [9, 10]  # DOTAv1: 9 = large vehicle, 10 = small vehicle

DOTAV1_NAMES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "roundabout",
    12: "soccer ball field",
    13: "swimming pool",
    14: "helicopter",
}

MAX_DATASET_FRAMES = int(os.environ.get("PK_DATASET_MAX_FRAMES", "48"))
BLUR_THRESHOLD = float(os.environ.get("PK_BLUR_THRESH", "30.0"))
DEDUP_DIFF_THRESHOLD = float(os.environ.get("PK_DEDUP_DIFF", "2.2"))
PSEUDO_CONF = float(os.environ.get("PK_PSEUDO_CONF", "0.30"))
TRAIN_EPOCHS = int(os.environ.get("PK_TRAIN_EPOCHS", "5"))
TRAIN_IMGSZ = int(os.environ.get("PK_TRAIN_IMGSZ", "640"))
TRAIN_BATCH = int(os.environ.get("PK_TRAIN_BATCH", "4"))
RANDOM_SEED = int(os.environ.get("PK_SEED", "42"))


def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _extract_obb_instances(res: Any, w: int, h: int, min_conf: float) -> list[dict[str, Any]]:
    """Kinyeri a jármű OBB-detektálásokat YOLO-OBB formátumban."""
    if res is None or getattr(res, "obb", None) is None or len(res.obb) == 0:
        return []
    pts_all = res.obb.xyxyxyxy.cpu().numpy()
    confs = res.obb.conf.cpu().numpy()
    clss = res.obb.cls.cpu().numpy().astype(int)
    out: list[dict[str, Any]] = []
    for pts, conf, cls in zip(pts_all, confs, clss):
        if int(cls) not in VEHICLE_CLASSES or float(conf) < min_conf:
            continue
        norm_pts = []
        for x, y in pts:
            nx = min(max(float(x) / max(w, 1), 0.0), 1.0)
            ny = min(max(float(y) / max(h, 1), 0.0), 1.0)
            norm_pts.append((nx, ny))
        out.append({
            "cls": int(cls),
            "conf": float(conf),
            "pts_px": pts.astype(np.float32),
            "pts_norm": norm_pts,
        })
    return out


def build_clean_yolo_dataset(
    video_path: Path,
    dataset_dir: Path,
    base_model: Any,
    device: str,
    model_lock: Any,
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Képkockákat gyűjt a videóból, tisztítja az adatokat, és időrendi Train/Val/Test splitet készít."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nem sikerült megnyitni a videót az adatkészlethez: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_video_frames <= 0:
        # Fallback ha a konténer nem adja vissza a frame számot
        raw_frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            raw_frames.append(fr)
        cap.release()
        n_cand = min(len(raw_frames), MAX_DATASET_FRAMES)
        idxs = np.linspace(0, max(0, len(raw_frames) - 1), n_cand, dtype=int) if raw_frames else []
        candidates = [(int(i), raw_frames[int(i)]) for i in idxs]
    else:
        n_cand = min(total_video_frames, MAX_DATASET_FRAMES)
        idxs = np.linspace(0, max(0, total_video_frames - 1), n_cand, dtype=int)
        candidates = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok and fr is not None:
                candidates.append((int(idx), fr))
        cap.release()

    stats = {
        "video_total_frames": total_video_frames,
        "sampled_candidates": len(candidates),
        "removed_blur": 0,
        "removed_duplicates": 0,
        "removed_outliers": 0,
        "kept_frames": 0,
        "blur_threshold": BLUR_THRESHOLD,
        "dedup_diff_threshold": DEDUP_DIFF_THRESHOLD,
        "pseudo_conf_threshold": PSEUDO_CONF,
        "total_vehicle_labels": 0,
        "class_counts": {"large_vehicle": 0, "small_vehicle": 0},
        "splits": {},
    }

    kept: list[dict[str, Any]] = []
    prev_small_gray: np.ndarray | None = None

    for i, (fidx, frame) in enumerate(candidates):
        if progress_cb and i % 4 == 0:
            pct = 42 + int(10 * (i + 1) / max(len(candidates), 1))
            progress_cb(pct, f"Adattisztítás és címkézés ({i + 1}/{len(candidates)})…")

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Elmosódottság szűrése (Laplacian variancia)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < BLUR_THRESHOLD:
            stats["removed_blur"] += 1
            continue

        # 2. Szélsőséges fényerő / hibás képkocka szűrése
        mean_luma = float(gray.mean())
        if mean_luma < 12.0 or mean_luma > 245.0:
            stats["removed_outliers"] += 1
            continue

        # 3. Duplikációk eltávolítása (időbeli közelség miatti majdnem azonos képkockák)
        small_gray = cv2.resize(gray, (160, 90))
        if prev_small_gray is not None:
            mad = float(np.mean(np.abs(small_gray.astype(np.float32) - prev_small_gray.astype(np.float32))))
            if mad < DEDUP_DIFF_THRESHOLD:
                stats["removed_duplicates"] += 1
                continue

        # 4. Pszeudo-címkézés előre tanított YOLOv8x-OBB-vel
        with model_lock:
            res = base_model.predict(
                frame,
                imgsz=TRAIN_IMGSZ,
                conf=PSEUDO_CONF,
                iou=0.45,
                classes=VEHICLE_CLASSES,
                device=device,
                verbose=False,
            )[0]
        instances = _extract_obb_instances(res, w, h, PSEUDO_CONF)
        if not instances:
            stats["removed_outliers"] += 1
            continue

        prev_small_gray = small_gray
        kept.append({
            "frame_idx": fidx,
            "frame": frame,
            "instances": instances,
            "blur_score": blur_score,
            "mean_luma": mean_luma,
        })

    stats["kept_frames"] = len(kept)
    if len(kept) < 6:
        raise RuntimeError(
            f"Túl kevés tiszta, járművet tartalmazó képkocka maradt a tisztítás után ({len(kept)} db). "
            "Legalább 6 képkocka szükséges a Train/Val/Test felosztáshoz."
        )

    # Időrendi (adatszivárgás-mentes) 70% / 15% / 15% felosztás
    n = len(kept)
    n_test = max(2, int(round(n * 0.15)))
    n_val = max(2, int(round(n * 0.15)))
    n_train = n - n_val - n_test
    if n_train < 2:
        n_train = max(2, n - 4)
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val

    split_slices = {
        "train": kept[:n_train],
        "val": kept[n_train : n_train + n_val],
        "test": kept[n_train + n_val :],
    }

    for split_name, items in split_slices.items():
        inst_count = 0
        for item in items:
            stem = f"frame_{item['frame_idx']:06d}"
            img_out = dataset_dir / "images" / split_name / f"{stem}.jpg"
            lbl_out = dataset_dir / "labels" / split_name / f"{stem}.txt"
            cv2.imwrite(str(img_out), item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 92])

            lines = []
            for inst in item["instances"]:
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in inst["pts_norm"])
                lines.append(f"{inst['cls']} {coords}")
                inst_count += 1
                stats["total_vehicle_labels"] += 1
                if inst["cls"] == 9:
                    stats["class_counts"]["large_vehicle"] += 1
                else:
                    stats["class_counts"]["small_vehicle"] += 1
            lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats["splits"][split_name] = {
            "frames": len(items),
            "instances": inst_count,
        }

    data_yaml = dataset_dir / "data.yaml"
    yaml_doc = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": DOTAV1_NAMES,
    }
    data_yaml.write_text(yaml.safe_dump(yaml_doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    stats["data_yaml"] = str(data_yaml)
    return stats


def _parse_results_csv(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            cleaned: dict[str, float] = {}
            for k, v in raw.items():
                if k is None:
                    continue
                try:
                    cleaned[k.strip()] = float(str(v).strip())
                except ValueError:
                    pass
            if cleaned:
                rows.append(cleaned)
    return rows


def _extract_val_metrics(val_res: Any) -> dict[str, Any]:
    rd = getattr(val_res, "results_dict", {}) or {}

    def pick(*keys: str) -> float:
        for k in keys:
            if k in rd and rd[k] is not None:
                val = float(rd[k])
                if math.isfinite(val):
                    return val
        return 0.0

    p = pick("metrics/precision(B)", "metrics/precision(M)", "precision")
    r = pick("metrics/recall(B)", "metrics/recall(M)", "recall")
    map50 = pick("metrics/mAP50(B)", "metrics/mAP50(M)", "mAP50")
    map50_95 = pick("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "mAP50-95")
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    cm_3x3 = None
    cm_obj = getattr(val_res, "confusion_matrix", None)
    if cm_obj is not None and getattr(cm_obj, "matrix", None) is not None:
        mat = np.asarray(cm_obj.matrix, dtype=float)
        if mat.shape[0] >= 16 and mat.shape[1] >= 16:
            # 9: large vehicle, 10: small vehicle, 15: background
            idx = [9, 10, 15]
            cm_3x3 = mat[np.ix_(idx, idx)].tolist()

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "confusion_matrix_3x3": cm_3x3,
    }


def _poly_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    x_min = max(0, int(min(pts_a[:, 0].min(), pts_b[:, 0].min())))
    y_min = max(0, int(min(pts_a[:, 1].min(), pts_b[:, 1].min())))
    x_max = int(max(pts_a[:, 0].max(), pts_b[:, 0].max())) + 2
    y_max = int(max(pts_a[:, 1].max(), pts_b[:, 1].max())) + 2
    w, h = max(x_max - x_min, 1), max(y_max - y_min, 1)
    if w > 4000 or h > 4000:
        return 0.0
    ma = np.zeros((h, w), dtype=np.uint8)
    mb = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ma, [(pts_a - [x_min, y_min]).astype(np.int32)], 1)
    cv2.fillPoly(mb, [(pts_b - [x_min, y_min]).astype(np.int32)], 1)
    inter = float(np.logical_and(ma, mb).sum())
    union = float(np.logical_or(ma, mb).sum())
    return inter / union if union > 0 else 0.0


def _load_gt_polys(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.is_file():
        return []
    out: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cls = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        out.append((cls, coords))
    return out


def _plot_dataset_summary(ds_stats: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Bal: Adattisztítás eredménye
    clean_labels = ["Megtartva", "Elmosódott", "Duplikátum", "Outlier/Üres"]
    clean_vals = [
        ds_stats["kept_frames"],
        ds_stats["removed_blur"],
        ds_stats["removed_duplicates"],
        ds_stats["removed_outliers"],
    ]
    colors_left = ["#10b981", "#f59e0b", "#38bdf8", "#ef4444"]
    bars1 = axes[0].bar(clean_labels, clean_vals, color=colors_left)
    axes[0].set_title("Adattisztítás (képkockák szűrése)", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Képkockák száma", color="#cbd5e1")
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(int(b.get_height())),
                     ha="center", va="bottom", color="#f8fafc", fontsize=9)

    # Jobb: Train / Val / Test felosztás
    splits = ds_stats["splits"]
    sp_names = ["Train (70%)", "Val (15%)", "Test (15%)"]
    sp_frames = [splits["train"]["frames"], splits["val"]["frames"], splits["test"]["frames"]]
    sp_inst = [splits["train"]["instances"], splits["val"]["instances"], splits["test"]["instances"]]
    x = np.arange(len(sp_names))
    w_bar = 0.35
    b_f = axes[1].bar(x - w_bar / 2, sp_frames, w_bar, label="Képkockák", color="#38bdf8")
    b_i = axes[1].bar(x + w_bar / 2, sp_inst, w_bar, label="Jármű címkék", color="#a855f7")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sp_names, color="#e2e8f0")
    axes[1].set_title("Időrendi Train / Val / Test felosztás", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for b in list(b_f) + list(b_i):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(int(b.get_height())),
                     ha="center", va="bottom", color="#f8fafc", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_learning_curves(curves: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#e2e8f0")
        ax.grid(color="#334155", linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    palette = {
        "Frozen (AdamW)": ("#38bdf8", "#0284c7"),
        "Fine-tuned (SGD)": ("#10b981", "#f59e0b"),
    }

    for label, rows in curves.items():
        if not rows:
            continue
        epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
        tr_loss = [r.get("train/box_loss", 0.0) + r.get("train/cls_loss", 0.0) + r.get("train/dfl_loss", 0.0) for r in rows]
        va_loss = [r.get("val/box_loss", 0.0) + r.get("val/cls_loss", 0.0) + r.get("val/dfl_loss", 0.0) for r in rows]
        map50 = [r.get("metrics/mAP50(B)", r.get("metrics/mAP50(M)", 0.0)) for r in rows]

        c_tr, c_va = palette.get(label, ("#f8fafc", "#94a3b8"))
        axes[0].plot(epochs, tr_loss, marker="o", color=c_tr, label=f"{label} Train Loss")
        axes[0].plot(epochs, va_loss, marker="s", linestyle="--", color=c_va, label=f"{label} Val Loss")
        axes[1].plot(epochs, map50, marker="o", color=c_tr, label=f"{label} Val mAP@50")

    axes[0].set_title("Tanulási görbék: Összes Loss (Train vs Val)", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", color="#cbd5e1")
    axes[0].set_ylabel("Loss (Box + Cls + DFL)", color="#cbd5e1")
    axes[0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)

    axes[1].set_title("Validációs mAP@50 fejlődése", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", color="#cbd5e1")
    axes[1].set_ylabel("mAP@50", color="#cbd5e1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_model_comparison(models_test: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#e2e8f0")
    ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    names = list(models_test.keys())
    metric_keys = [("precision", "Precision", "#38bdf8"), ("recall", "Recall", "#a855f7"),
                   ("f1", "F1-score", "#10b981"), ("map50", "mAP@50", "#f59e0b")]
    x = np.arange(len(names))
    w = 0.19

    for idx, (k, lbl, col) in enumerate(metric_keys):
        vals = [models_test[n].get(k, 0.0) for n in names]
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=lbl, color=col)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                    ha="center", va="bottom", color="#f8fafc", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#f8fafc", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Modellek összehasonlítása a független Test halmazon", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", ncol=4, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_confusion_matrix(cm: list[list[float]] | None, title_model: str, out_path: Path) -> None:
    if cm is None:
        cm_arr = np.zeros((3, 3), dtype=float)
    else:
        cm_arr = np.asarray(cm, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    im = ax.imshow(cm_arr, cmap="Blues")

    labels = ["Large Vehicle", "Small Vehicle", "Background"]
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels, color="#f8fafc", fontsize=9)
    ax.set_yticklabels(labels, color="#f8fafc", fontsize=9)
    ax.set_xlabel("Valós osztály (Ground Truth)", color="#cbd5e1")
    ax.set_ylabel("Prediktált osztály", color="#cbd5e1")
    ax.set_title(f"Confusion Matrix — {title_model}", color="#f8fafc", fontsize=11, fontweight="bold")

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 0.5
    for i in range(3):
        for j in range(3):
            val = int(round(cm_arr[i, j]))
            ax.text(j, i, str(val), ha="center", va="center",
                    color="#0f172a" if cm_arr[i, j] > thresh else "#f8fafc", fontweight="bold", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#e2e8f0")
    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_error_analysis(
    dataset_dir: Path,
    best_model: Any,
    device: str,
    model_lock: Any,
    out_path: Path,
) -> dict[str, Any]:
    test_imgs = sorted((dataset_dir / "images" / "test").glob("*.jpg"))
    samples = test_imgs[:4]
    total_tp, total_fp, total_fn = 0, 0, 0

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=125)
    fig.patch.set_facecolor("#0f172a")
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        ax.set_facecolor("#1e293b")
        ax.axis("off")
        if idx >= len(samples):
            continue
        img_p = samples[idx]
        lbl_p = dataset_dir / "labels" / "test" / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        gt_list = _load_gt_polys(lbl_p, w, h)

        with model_lock:
            pred_res = best_model.predict(
                img, imgsz=TRAIN_IMGSZ, conf=0.25, iou=0.45, classes=VEHICLE_CLASSES, device=device, verbose=False
            )[0]
        preds = _extract_obb_instances(pred_res, w, h, 0.25)

        matched_gt = set()
        fp_polys = []
        tp_polys = []
        for pr in preds:
            best_iou, best_j = 0.0, -1
            for j, (_, gt_poly) in enumerate(gt_list):
                if j in matched_gt:
                    continue
                iou = _poly_iou(pr["pts_px"], gt_poly)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.45 and best_j >= 0:
                matched_gt.add(best_j)
                tp_polys.append(pr["pts_px"])
            else:
                fp_polys.append(pr["pts_px"])

        fn_polys = [gt_poly for j, (_, gt_poly) in enumerate(gt_list) if j not in matched_gt]
        total_tp += len(tp_polys)
        total_fp += len(fp_polys)
        total_fn += len(fn_polys)

        vis = img.copy()
        for p in tp_polys:
            cv2.polylines(vis, [p.astype(np.int32)], True, (0, 255, 0), 2)
        for p in fp_polys:
            cv2.polylines(vis, [p.astype(np.int32)], True, (0, 0, 255), 3)
        for p in fn_polys:
            cv2.polylines(vis, [p.astype(np.int32)], True, (0, 215, 255), 3)

        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{img_p.name} | TP={len(tp_polys)} (Zöld)  FP={len(fp_polys)} (Piros)  FN={len(fn_polys)} (Sárga)",
            color="#f8fafc", fontsize=9, fontweight="bold",
        )

    fig.suptitle("Hibaelemzés a Test képeken (TP / False Positive / False Negative)",
                 color="#f8fafc", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "inspected_test_frames": len(samples),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "typical_causes": [
            "A képkocka szélén részlegesen látszó járművek (alacsonyabb IoU vagy kihagyott FN).",
            "Erős árnyékok vagy tető-kontrasztok miatti téves riasztás (FP).",
            "Kisméretű távoli járművek esetén csökkenő konfidencia.",
        ],
    }


def _plot_interpretability_saliency(
    dataset_dir: Path,
    best_model: Any,
    device: str,
    model_lock: Any,
    out_path: Path,
) -> None:
    """Feature Activation / Grad-CAM jellegű hőtérkép a YOLO gerincháló mély rétegéből."""
    test_imgs = sorted((dataset_dir / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        return
    img = cv2.imread(str(test_imgs[0]))
    h, w = img.shape[:2]

    captured: list[Any] = []
    layers = getattr(getattr(best_model, "model", None), "model", None)
    hook_handle = None
    if layers is not None and len(layers) > 2:
        target_idx = min(9, len(layers) - 2)
        def _hook(_mod: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, (tuple, list)) else out
            try:
                captured.append(t.detach().cpu().numpy())
            except Exception:
                pass
        hook_handle = layers[target_idx].register_forward_hook(_hook)

    try:
        with model_lock:
            _ = best_model.predict(
                img, imgsz=TRAIN_IMGSZ, conf=0.25, iou=0.45, classes=VEHICLE_CLASSES, device=device, verbose=False
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.axis("off")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb)
    axes[0].set_title("Eredeti Test képkocka", color="#f8fafc", fontsize=10, fontweight="bold")

    if captured:
        feat = captured[0][0]  # (C, H', W')
        act = np.mean(np.maximum(feat, 0.0), axis=0)
        act = (act - act.min()) / (act.max() - act.min() + 1e-8)
        act_resized = cv2.resize(act.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        heat_bgr = cv2.applyColorMap(np.uint8(255 * act_resized), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb, 0.55, heat_rgb, 0.45, 0)
        axes[1].imshow(overlay)
        axes[1].set_title("Interpretálhatóság: Gerincháló Aktivációs Hőtérkép", color="#f8fafc", fontsize=10, fontweight="bold")
    else:
        axes[1].imshow(rgb)
        axes[1].set_title("Aktivációs hőtérkép nem elérhető", color="#f8fafc", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_full_training_suite(
    video_path: Path,
    job_dir: Path,
    base_model_path: str,
    base_model: Any,
    device: str,
    model_lock: Any,
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Lefuttatja a teljes adatelőkészítő, transfer learning, kiértékelő és riportkészítő láncot."""
    _set_seed(RANDOM_SEED)
    t_start = time.time()
    dataset_dir = job_dir / "dataset"
    runs_dir = job_dir / "runs"
    report_dir = job_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(40, "Saját adatkészlet összeállítása és tisztítása a videóból…")

    ds_stats = build_clean_yolo_dataset(
        video_path=video_path,
        dataset_dir=dataset_dir,
        base_model=base_model,
        device=device,
        model_lock=model_lock,
        progress_cb=progress_cb,
    )
    data_yaml = ds_stats["data_yaml"]

    runs_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO, settings
    settings.update({"runs_dir": str(runs_dir), "datasets_dir": str(dataset_dir.parent)})

    # 1. Baseline: Előre tanított YOLOv8x-OBB kiértékelése a Test halmazon
    if progress_cb:
        progress_cb(55, "Baseline modell (előre tanított YOLOv8x-OBB) kiértékelése…")
    with model_lock:
        base_val = base_model.val(
            data=data_yaml, split="val", imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH, device=device,
            project=str(runs_dir), name="val_base", exist_ok=True, plots=False, verbose=False
        )
        base_test = base_model.val(
            data=data_yaml, split="test", imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH, device=device,
            project=str(runs_dir), name="test_base", exist_ok=True, plots=False, verbose=False
        )
    base_val_m = _extract_val_metrics(base_val)
    base_test_m = _extract_val_metrics(base_test)

    # 2. Transfer Learning Variáns A: Befagyasztott gerincháló (freeze=10, AdamW)
    if progress_cb:
        progress_cb(62, "Transfer Learning A: Befagyasztott gerincháló (AdamW) tanítása…")
    model_a = YOLO(base_model_path)
    with model_lock:
        model_a.train(
            data=data_yaml,
            epochs=TRAIN_EPOCHS,
            imgsz=TRAIN_IMGSZ,
            batch=TRAIN_BATCH,
            device=device,
            freeze=10,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            patience=3,
            mosaic=0.5,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            seed=RANDOM_SEED,
            project=str(runs_dir),
            name="frozen_adamw",
            exist_ok=True,
            plots=False,
            verbose=False,
        )
    best_a_path = runs_dir / "frozen_adamw" / "weights" / "best.pt"
    eval_a = YOLO(str(best_a_path)) if best_a_path.is_file() else model_a
    with model_lock:
        a_val_m = _extract_val_metrics(
            eval_a.val(data=data_yaml, split="val", imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH, device=device,
                       project=str(runs_dir), name="val_frozen_adamw", exist_ok=True, plots=False, verbose=False)
        )
        a_test_m = _extract_val_metrics(
            eval_a.val(data=data_yaml, split="test", imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH, device=device,
                       project=str(runs_dir), name="test_frozen_adamw", exist_ok=True, plots=False, verbose=False)
        )
    curves_a = _parse_results_csv(runs_dir / "frozen_adamw" / "results.csv")

    # 3. Transfer Learning Variáns B: Teljes finomhangolás (freeze=0, SGD)
    if progress_cb:
        progress_cb(76, "Transfer Learning B: Teljes finomhangolás (SGD) tanítása…")
    model_b = YOLO(base_model_path)
    with model_lock:
        model_b.train(
            data=data_yaml,
            epochs=TRAIN_EPOCHS,
            imgsz=TRAIN_IMGSZ,
            batch=TRAIN_BATCH,
            device=device,
            freeze=0,
            optimizer="SGD",
            lr0=0.0005,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            patience=3,
            mosaic=0.5,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            seed=RANDOM_SEED,
            project=str(runs_dir),
            name="finetune_sgd",
            exist_ok=True,
            plots=False,
            verbose=False,
        )
    best_b_path = runs_dir / "finetune_sgd" / "weights" / "best.pt"
    eval_b = YOLO(str(best_b_path)) if best_b_path.is_file() else model_b
    with model_lock:
        b_val_m = _extract_val_metrics(
            eval_b.val(data=data_yaml, split="val", imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH, device=device,
                       project=str(runs_dir), name="val_finetune_sgd", exist_ok=True, plots=False, verbose=False)
        )
        b_test_m = _extract_val_metrics(
            eval_b.val(data=data_yaml, split="test", imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH, device=device,
                       project=str(runs_dir), name="test_finetune_sgd", exist_ok=True, plots=False, verbose=False)
        )
    curves_b = _parse_results_csv(runs_dir / "finetune_sgd" / "results.csv")

    # Modellválasztás kizárólag a Validation halmaz alapján (adatszivárgás elkerülése)
    candidates_val = [
        ("Baseline (Pretrained)", base_val_m["map50_95"], base_model, base_test_m),
        ("Frozen (AdamW)", a_val_m["map50_95"], eval_a, a_test_m),
        ("Fine-tuned (SGD)", b_val_m["map50_95"], eval_b, b_test_m),
    ]
    winner_name, _, winner_model, winner_test_m = max(candidates_val, key=lambda x: x[1])

    if progress_cb:
        progress_cb(88, "Ábrák, hibaelemzés és interpretálhatósági hőtérkép generálása…")

    models_test = {
        "Baseline (Pretrained)": base_test_m,
        "Frozen (AdamW)": a_test_m,
        "Fine-tuned (SGD)": b_test_m,
    }

    _plot_dataset_summary(ds_stats, report_dir / "dataset_summary.png")
    _plot_learning_curves({"Frozen (AdamW)": curves_a, "Fine-tuned (SGD)": curves_b}, report_dir / "learning_curves.png")
    _plot_model_comparison(models_test, report_dir / "model_comparison.png")
    _plot_confusion_matrix(winner_test_m.get("confusion_matrix_3x3"), winner_name, report_dir / "confusion_matrix.png")
    err_summary = _plot_error_analysis(dataset_dir, winner_model, device, model_lock, report_dir / "error_analysis.png")
    _plot_interpretability_saliency(dataset_dir, winner_model, device, model_lock, report_dir / "interpretability_saliency.png")

    # Overfitting / Underfitting diagnosztika az utolsó epoch alapján
    def _loss_gap(rows: list[dict[str, float]]) -> dict[str, float] | None:
        if not rows:
            return None
        last = rows[-1]
        tr = last.get("train/box_loss", 0.0) + last.get("train/cls_loss", 0.0) + last.get("train/dfl_loss", 0.0)
        va = last.get("val/box_loss", 0.0) + last.get("val/cls_loss", 0.0) + last.get("val/dfl_loss", 0.0)
        return {"train_loss": round(tr, 4), "val_loss": round(va, 4), "gap": round(va - tr, 4)}

    report_doc = {
        "duration_sec": round(time.time() - t_start, 2),
        "seed": RANDOM_SEED,
        "device": device,
        "base_model": base_model_path,
        "winner_model": winner_name,
        "problem": {
            "task": "Objektumdetektálás (Oriented Bounding Box) + Parkolóhely-foglaltság becslés",
            "input": "Drón / magaslati kamera videófolyam (MP4/AVI/MOV)",
            "output": "Forgatott jármű-OBB koordináták, parkolóhely-poligonok, foglaltsági állapot (szabad/foglalt) és feldolgozott MP4 videó",
            "metrics": ["Precision", "Recall", "F1-score", "mAP@50", "mAP@50-95", "Confusion Matrix"],
        },
        "dataset": ds_stats,
        "split_justification": (
            "A videó egymást követő képkockái erősen korreláltak; véletlenszerű keverés esetén azonos pillanatok "
            "kerülnének a Train és a Test halmazba (temporális adatszivárgás). Ezért szigorúan időrendi, "
            "összefüggő blokkokra bontott 70% / 15% / 15% felosztást alkalmaztunk."
        ),
        "hyperparameters": {
            "loss_function": "YOLOv8-OBB összetett loss (Box/ProbIoU + DFL + BCE Class Loss)",
            "epochs": TRAIN_EPOCHS,
            "imgsz": TRAIN_IMGSZ,
            "batch_size": TRAIN_BATCH,
            "early_stopping_patience": 3,
            "augmentation": "Mosaic (0.5), HSV színeltolás (h=0.015, s=0.4, v=0.3)",
            "variants": {
                "Frozen (AdamW)": {"freeze": 10, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005},
                "Fine-tuned (SGD)": {"freeze": 0, "optimizer": "SGD", "lr0": 0.0005, "momentum": 0.937, "weight_decay": 0.0005},
            },
        },
        "validation_metrics": {
            "Baseline (Pretrained)": base_val_m,
            "Frozen (AdamW)": a_val_m,
            "Fine-tuned (SGD)": b_val_m,
        },
        "test_metrics": models_test,
        "overfitting_check": {
            "Frozen (AdamW)": _loss_gap(curves_a),
            "Fine-tuned (SGD)": _loss_gap(curves_b),
        },
        "error_analysis": err_summary,
        "responsible_ai": {
            "interpretability": "Mélységi gerincháló aktivációs hőtérkép (Grad-CAM jellegű térbeli saliency térkép).",
            "limitations": [
                "A pszeudo-címkézés az előre tanított modell hibáit örökölheti kézi annotáció hiányában.",
                "Extrém időjárás (köd, hó, éjszakai sötétség) és erős takarás (fák lombja) rontja a detektálást.",
            ],
            "privacy_ethics": (
                "Magaslati nézetben a rendszámok és arcok jellemzően nem olvashatók, azonban közterületi "
                "kamerafelvételeknél a GDPR adatvédelmi előírásait (anonimizálás, tárolási idő korlátozása) be kell tartani."
            ),
        },
        "images": [
            "dataset_summary.png",
            "learning_curves.png",
            "model_comparison.png",
            "confusion_matrix.png",
            "error_analysis.png",
            "interpretability_saliency.png",
        ],
    }

    (report_dir / "training_report.json").write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if progress_cb:
        progress_cb(96, "Tanítási riport és ábrák elkészültek.")
    return report_doc
