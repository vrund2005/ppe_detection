import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "best.pt"
CONF_THRESH  = 0.45
SOURCE       = 0          # 0 = webcam, or "video.mp4" for a file

# Which class IDs are "violation" classes (no helmet, no vest, etc.)
# Check your data.yaml — adjust these indices to match YOUR class order
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
SAFE_CLASSES      = {"Hardhat", "Safety Vest", "Mask"}

# ── Load model ────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
class_names = model.names  # dict: {0: 'Hardhat', 1: 'Mask', ...}

# ── Supervision annotators ────────────────────────────────────────────────────
# supervision handles all box drawing — clean, production-grade
box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# ── Stats tracking ────────────────────────────────────────────────────────────
frame_count      = 0
violation_log    = []   # each entry: {"time": ..., "violation": ..., "frame": ...}
recent_violation = False
violation_flash  = 0    # frames to show red flash

cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Run inference ─────────────────────────────────────────────────────────
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    # Convert to supervision Detections object
    detections = sv.Detections.from_ultralytics(results)

    # ── Check for violations ──────────────────────────────────────────────────
    detected_names = [class_names[cls_id] for cls_id in detections.class_id]
    violations_this_frame = [n for n in detected_names if n in VIOLATION_CLASSES]
    safe_this_frame       = [n for n in detected_names if n in SAFE_CLASSES]

    if violations_this_frame:
        violation_log.append({
            "time":      datetime.now().strftime("%H:%M:%S"),
            "violation": ", ".join(violations_this_frame),
            "frame":     frame_count,
        })
        violation_flash = 15  # flash red for 15 frames

    # ── Build labels for each box ─────────────────────────────────────────────
    labels = []
    for cls_id, conf in zip(detections.class_id, detections.confidence):
        name = class_names[cls_id]
        is_violation = name in VIOLATION_CLASSES
        label = f"{'! ' if is_violation else ''}{name} {conf:.2f}"
        labels.append(label)

    # ── Colour boxes: red for violation, green for safe ───────────────────────
    colors = []
    for cls_id in detections.class_id:
        name = class_names[cls_id]
        if name in VIOLATION_CLASSES:
            colors.append(sv.Color.RED)
        elif name in SAFE_CLASSES:
            colors.append(sv.Color.GREEN)
        else:
            colors.append(sv.Color.WHITE)

    # supervision v0.19+ supports per-detection colors via ColorLookup
    annotated = box_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    # ── Dashboard overlay ─────────────────────────────────────────────────────
    # Red flash on violation
    if violation_flash > 0:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], annotated.shape[0]),
                      (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        violation_flash -= 1

    # Stats panel (top-left)
    panel_h, panel_w = 140, 320
    cv2.rectangle(annotated, (10, 10), (10+panel_w, 10+panel_h), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (10+panel_w, 10+panel_h), (80, 80, 80), 1)

    total_violations = len(violation_log)
    compliance_pct   = 100.0 if not (violations_this_frame or safe_this_frame) else \
                       round(len(safe_this_frame) /
                             max(len(safe_this_frame) + len(violations_this_frame), 1) * 100)

    cv2.putText(annotated, "PPE Compliance Monitor",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(annotated, f"Frame: {frame_count}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(annotated, f"Violations logged: {total_violations}",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

    comp_color = (80, 220, 80) if compliance_pct >= 80 else (80, 80, 255)
    cv2.putText(annotated, f"Compliance: {compliance_pct}%",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, comp_color, 1)

    # Last violation
    if violation_log:
        last = violation_log[-1]
        cv2.putText(annotated, f"Last: {last['time']} - {last['violation'][:28]}",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,255), 1)

    cv2.imshow("PPE Compliance Detector", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # Save current frame with timestamp
        fname = f"violation_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"Saved {fname}")

cap.release()
cv2.destroyAllWindows()

# ── Print violation report ─────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Session Report — {frame_count} frames processed")
print(f"Total violations detected: {len(violation_log)}")
print(f"\nViolation log:")
for v in violation_log[-10:]:  # last 10
    print(f"  [{v['time']}] Frame {v['frame']:5d} — {v['violation']}")