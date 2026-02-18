import torch
import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# ---------- Helper: draw only detections inside ROI ----------
def draw_detections_inside_roi(frame, boxes, class_names, roi_points, conf_threshold=0.5):
    """
    Draw bounding boxes only for detections whose center is inside the ROI.
    boxes: ultralytics engine.results.Boxes object
    """
    img = frame.copy()
    if boxes is None or len(boxes) == 0:
        return img

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        if conf < conf_threshold:
            continue

        # Compute center
        x1, y1, x2, y2 = xyxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Check if inside ROI (if ROI is defined)
        if roi_points is not None:
            if cv2.pointPolygonTest(roi_points, (cx, cy), False) < 0:
                continue  # outside, skip drawing

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put label
        label = f"{class_names[cls]} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw ROI polygon (if any)
    if roi_points is not None:
        cv2.polylines(img, [roi_points], True, (255, 0, 0), 3)

    return img

# ---------- ROI selection with resized view ----------
def select_roi(original_frame, max_width=1280, max_height=720,
               window_name="Draw polygon (press 'c' to confirm, 'q' to skip)"):
    """
    Let user draw a polygon on a resized view of the frame.
    Points are stored in original image coordinates.
    """
    points_orig = []
    roi = None

    h_orig, w_orig = original_frame.shape[:2]
    scale = min(max_width / w_orig, max_height / h_orig, 1.0)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    display_frame = cv2.resize(original_frame, (new_w, new_h))

    def mouse_callback(event, x, y, flags, param):
        nonlocal points_orig
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            points_orig.append((orig_x, orig_y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points_orig:
                points_orig.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = display_frame.copy()
        if len(points_orig) > 1:
            pts_disp = [(int(x * scale), int(y * scale)) for x, y in points_orig]
            cv2.polylines(display, [np.array(pts_disp)], False, (0, 255, 0), 2)
        for (x_orig, y_orig) in points_orig:
            x_disp = int(x_orig * scale)
            y_disp = int(y_orig * scale)
            cv2.circle(display, (x_disp, y_disp), 3, (0, 0, 255), -1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(points_orig) < 3:
                print("Need at least 3 points. Try again.")
                points_orig.clear()
                continue
            roi = np.array(points_orig, dtype=np.int32)
            break
        elif key == ord('q') or key == 27:
            roi = None
            break

    cv2.destroyWindow(window_name)
    return roi

# ---------- Paths ----------
video_path = r"D:\Downloads\0217.mp4"
model_path = r"D:\THESIS\Roboflow Model_Wireless Int. 90m\best (2).pt"
desktop = Path(os.path.expanduser("~")) / "Desktop"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
extract_folder = desktop / f"extracted_frames_{timestamp}"
detect_folder = desktop / f"detected_frames_{timestamp}"
extract_folder.mkdir(exist_ok=True)
detect_folder.mkdir(exist_ok=True)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# ---------- Open video and get first frame for ROI ----------
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Error opening video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    print("Could not read first frame")
    exit()

print("Select region of interest (left click to add points, right click to undo, 'c' to confirm, 'q' to skip).")
roi_points = select_roi(first_frame)
use_roi = roi_points is not None

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------- Load YOLO model ----------
model = YOLO(model_path)
class_names = model.names
target_classes = ['car', 'motorcycle', 'bus']
target_ids = [id for id, name in class_names.items() if name.lower() in target_classes]
print(f"Target classes: {[class_names[id] for id in target_ids]}")

# ---------- Prepare data storage ----------
data = {th: [] for th in thresholds}

interval_frames = int(round(5 * fps))
print(f"FPS: {fps}, extracting every {interval_frames} frames (~5 sec)")

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % interval_frames == 0:
        time_sec = frame_index / fps

        # Save raw frame
        frame_filename = f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(extract_folder / frame_filename), frame)

        # Run detection (use a low confidence to get all candidates)
        results = model(frame, conf=0.3)[0]   # conf=0.3 captures everything above lowest threshold
        boxes = results.boxes

        # Count inside ROI per threshold
        counts = {th: {'car': 0, 'motorcycle': 0, 'bus': 0} for th in thresholds}

        if boxes is not None and len(boxes) > 0:
            for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                cls = int(cls)
                conf = float(conf)
                if cls not in target_ids:
                    continue

                x1, y1, x2, y2 = xyxy.tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if use_roi:
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) < 0:
                        continue

                name = class_names[cls].lower()
                for th in thresholds:
                    if conf >= th:
                        counts[th][name] += 1

        for th in thresholds:
            car = counts[th]['car']
            motorcycle = counts[th]['motorcycle']
            bus = counts[th]['bus']
            data[th].append([time_sec, car, motorcycle, bus, car + motorcycle + bus])

        # --- Save annotated image: only detections INSIDE ROI with conf >= 0.5 ---
        # Use the same boxes but filter and draw manually
        annotated = draw_detections_inside_roi(frame, boxes, class_names, roi_points if use_roi else None, conf_threshold=0.5)
        cv2.imwrite(str(detect_folder / frame_filename), annotated)

    frame_index += 1

cap.release()
print("Frame extraction and detection completed.")

# ---------- Generate Excel report ----------
excel_path = desktop / f"vehicle_counts_{timestamp}.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for th in thresholds:
        df = pd.DataFrame(data[th],
                          columns=['Time (s)', 'Car', 'Motorcycle', 'Bus', 'Total'])
        sheet_name = f"conf_{th}".replace('.', '_')
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Excel report saved to: {excel_path}")