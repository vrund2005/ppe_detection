# PPE Compliance Detection System

A Computer Vision project for detecting Personal Protective Equipment (PPE) compliance and safety violations using YOLO.

This project identifies whether workers are following workplace safety protocols by detecting the presence or absence of required PPE items such as:

* Hardhat
* Safety Vest
* Mask

It also detects violations such as:

* NO-Hardhat
* NO-Safety Vest
* NO-Mask

This helps improve workplace monitoring, industrial safety compliance, and automated safety audits.

---

## Project Overview

The model was trained on a PPE detection dataset containing approximately **44,000 total images**, which was later balanced and capped to **500 images per class** across **6 classes** for cleaner and more effective training.

### Classes Used

### Safe Classes

```python
SAFE_CLASSES = {
    "Hardhat",
    "Safety Vest",
    "Mask"
}
```

### Violation Classes

```python
VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask"
}
```

---

## Model Performance

### Evaluation Metric

* **mAP@50:** `0.67`

This indicates decent detection performance for real-world PPE compliance monitoring scenarios.

---

## Tech Stack

* Python
* YOLO (Ultralytics)
* OpenCV
* Roboflow
* NumPy
* Matplotlib

---

## Dataset Preparation

### Original Dataset

* Total Images: ~44,000
* Link : PPE Detection[https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model]

### Final Training Dataset

To reduce class imbalance and improve consistency:

* 6 classes selected
* Maximum 500 images per class
* Balanced dataset used for training

This prevents model bias toward dominant classes and improves generalization.

---

## Training

The model was trained using YOLO with custom annotations exported from Roboflow.

Example training code:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # nano — trains ~4x faster than small, good enough for 500/class

results = model.train(
    data="ppe_reduced/data.yaml",   # ← your reduced dataset
    epochs=60,                       # fewer epochs needed — smaller dataset converges faster
    patience=15,
    imgsz=640,
    batch=32,                        # can increase batch since dataset is smaller
    device=0,
    project="runs/ppe",
    name="reduced_v1",
    pretrained=True,
)
```

---

## Use Cases

* Construction Site Monitoring
* Factory Safety Compliance
* Industrial Worker Surveillance
* Automated PPE Violation Detection
* Smart Safety Auditing Systems

---

## Author

**Patel Vrund Kalpeshbhai**

B.Tech - Data Science

---
