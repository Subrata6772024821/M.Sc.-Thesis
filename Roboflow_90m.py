import torch
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# Paths
video_path = r"D:\Downloads\0217.mp4"
model_path = r"D:\Downloads\best (2).pt"
desktop = Path(os.path.expanduser("~")) / "Desktop"

# Create output folders with timestamp to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
extract_folder = desktop / f"extracted_frames_{timestamp}"
detect_folder = desktop / f"detected_frames_{timestamp}"
extract_folder.mkdir(exist_ok=True)
detect_folder.mkdir(exist_ok=True)

# Confidence thresholds to evaluate
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Open video
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Error opening video")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

# Calculate frame interval for 5 seconds
interval_frames = int(round(5 * fps))
print(f"Video FPS: {fps}, extracting every {interval_frames} frames (~5 sec)")

# Load YOLO model
model = YOLO(model_path)
class_names = model.names
target_classes = ['car', 'motorcycle', 'bus']
target_ids = [id for id, name in class_names.items() if name.lower() in target_classes]
print(f"Target class IDs: {target_ids} -> {[class_names[id] for id in target_ids]}")

# Data structure: data[threshold] = list of [frame_time, car, motorcycle, bus, total]
data = {th: [] for th in thresholds}

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % interval_frames == 0:
        time_sec = frame_index / fps

        # Save extracted (raw) frame
        frame_filename = f"frame_{frame_index:06d}.jpg"
        frame_path = extract_folder / frame_filename
        cv2.imwrite(str(frame_path), frame)
        print(f"Extracted frame {frame_index} at {time_sec:.2f}s")

        # Run detection once (keep all detections, we will filter by confidence later)
        results = model(frame)[0]
        boxes = results.boxes

        if boxes is not None:
            # Count vehicles for each confidence threshold
            for th in thresholds:
                car_cnt = 0
                motorcycle_cnt = 0
                bus_cnt = 0
                for cls, conf in zip(boxes.cls, boxes.conf):
                    cls = int(cls)
                    conf = float(conf)
                    if conf >= th and cls in target_ids:
                        class_name = class_names[cls].lower()
                        if class_name == 'car':
                            car_cnt += 1
                        elif class_name == 'motorcycle':
                            motorcycle_cnt += 1
                        elif class_name == 'bus':
                            bus_cnt += 1
                total_cnt = car_cnt + motorcycle_cnt + bus_cnt
                data[th].append([time_sec, car_cnt, motorcycle_cnt, bus_cnt, total_cnt])

        # --- Save annotated image with detections at confidence >= 0.5 ---
        # Run inference again with conf=0.5 to show only boxes above that threshold
        save_results = model(frame, conf=0.5)[0]
        annotated = save_results.plot()  # draws bounding boxes
        cv2.imwrite(str(detect_folder / frame_filename), annotated)

    frame_index += 1

cap.release()
print("Frame extraction and detection completed.")

# --- Generate Excel report ---
excel_path = desktop / f"vehicle_counts_{timestamp}.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for th in thresholds:
        df = pd.DataFrame(data[th],
                          columns=['Time (s)', 'Car', 'Motorcycle', 'Bus', 'Total'])
        sheet_name = f"conf_{th}".replace('.', '_')
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Excel report saved to: {excel_path}")