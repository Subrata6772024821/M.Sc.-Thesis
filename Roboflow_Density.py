import torch
import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# ---------------- SETTINGS ---------------- #
DISPLAY_SCALE = 0.5

video_path = r"D:\THESIS\Charoenphool Intersection\DJI_20260119102531_0007_D.MP4"
model_path = r"D:\THESIS\Charoenphool Intersection\Model_30\Yolo 11s\best (5).pt"

desktop = Path(os.path.expanduser("~")) / "Desktop"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

extract_folder = desktop / f"extracted_frames_{timestamp}"
extract_folder.mkdir(exist_ok=True)

thresholds = [0.3,0.4,0.5,0.6,0.7,0.8]

# create detection folders per confidence
detect_folders = {}
for th in thresholds:
    folder = desktop / f"detected_conf_{str(th).replace('.','_')}_{timestamp}"
    folder.mkdir(exist_ok=True)
    detect_folders[th] = folder

# ---------------- ROI SELECTION ---------------- #
roi_points=[]

def select_roi(frame,name="ROI"):
    global roi_points
    roi_points=[]

    display_frame=cv2.resize(frame,None,
                             fx=DISPLAY_SCALE,
                             fy=DISPLAY_SCALE)

    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            roi_points.append(
                (int(x/DISPLAY_SCALE),
                 int(y/DISPLAY_SCALE))
            )

    cv2.namedWindow(name)
    cv2.setMouseCallback(name,mouse)

    print(f"Draw {name} → press 's'")

    while True:
        temp=display_frame.copy()

        for p in roi_points:
            cv2.circle(temp,
                       (int(p[0]*DISPLAY_SCALE),
                        int(p[1]*DISPLAY_SCALE)),
                       4,(0,255,0),-1)

        if len(roi_points)>1:
            pts=np.array([(int(p[0]*DISPLAY_SCALE),
                           int(p[1]*DISPLAY_SCALE))
                          for p in roi_points])
            cv2.polylines(temp,[pts],False,(0,255,0),2)

        cv2.imshow(name,temp)
        if cv2.waitKey(1)&0xFF==ord('s'):
            break

    cv2.destroyWindow(name)
    return np.array(roi_points,dtype=np.int32)

# ---------------- CALIBRATION ---------------- #
def select_calibration(frame):

    pts=[]

    display_frame=cv2.resize(frame,None,
                             fx=DISPLAY_SCALE,
                             fy=DISPLAY_SCALE)

    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x/DISPLAY_SCALE),
                        int(y/DISPLAY_SCALE)))

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration",click)

    print("Select TWO calibration points → press 's'")

    while True:
        temp=display_frame.copy()

        for p in pts:
            cv2.circle(temp,
                       (int(p[0]*DISPLAY_SCALE),
                        int(p[1]*DISPLAY_SCALE)),
                       5,(0,0,255),-1)

        if len(pts)==2:
            cv2.line(temp,
                     (int(pts[0][0]*DISPLAY_SCALE),
                      int(pts[0][1]*DISPLAY_SCALE)),
                     (int(pts[1][0]*DISPLAY_SCALE),
                      int(pts[1][1]*DISPLAY_SCALE)),
                     (0,0,255),2)

        cv2.imshow("Calibration",temp)

        if cv2.waitKey(1)&0xFF==ord('s'):
            break

    cv2.destroyWindow("Calibration")
    return pts

# ---------------- UTILITIES ---------------- #
def inside_roi(point,roi):
    return cv2.pointPolygonTest(roi,point,False)>=0

def roi_length(roi,mpp):
    xs=roi[:,0]
    return (max(xs)-min(xs))*mpp

# ---------------- OPEN VIDEO ---------------- #
cap=cv2.VideoCapture(video_path)
fps=cap.get(cv2.CAP_PROP_FPS)

interval_frames=int(fps)   # every 1 second

ret,first_frame=cap.read()

# ---------------- CALIBRATION ---------------- #
calib_pts=select_calibration(first_frame)
real_length=float(input("Calibration length (m): "))

pixel_dist=np.linalg.norm(
    np.array(calib_pts[0])-np.array(calib_pts[1])
)

meters_per_pixel=real_length/pixel_dist

# ---------------- ROI ---------------- #
roi1=select_roi(first_frame,"ROI 1")
roi2=select_roi(first_frame,"ROI 2")

lanes_roi1=int(input("ROI1 lanes: "))
lanes_roi2=int(input("ROI2 lanes: "))

roi1_len=roi_length(roi1,meters_per_pixel)
roi2_len=roi_length(roi2,meters_per_pixel)

cap.set(cv2.CAP_PROP_POS_FRAMES,0)

# ---------------- MODEL ---------------- #
model=YOLO(model_path)
class_names=model.names

target_classes=['car','motorcycle','bus']
target_ids=[i for i,n in class_names.items()
            if n.lower() in target_classes]

data={th:[] for th in thresholds}

# ---------------- PROCESS ---------------- #
frame_index=0
time_counter=0   # ✅ integer time

while True:

    ret,frame=cap.read()
    if not ret:
        break

    if frame_index%interval_frames==0:

        time_sec=time_counter
        frame_name=f"frame_{frame_index:06d}.jpg"

        # ---------- draw ROI on extracted ----------
        extracted_img=frame.copy()
        cv2.polylines(extracted_img,[roi1],True,(0,255,0),3)
        cv2.polylines(extracted_img,[roi2],True,(255,0,0),3)

        cv2.imwrite(str(extract_folder/frame_name),
                    extracted_img)

        results=model(frame)[0]
        boxes=results.boxes

        if boxes is not None:

            for th in thresholds:

                car_cnt=motorcycle_cnt=bus_cnt=0
                roi1_cnt=roi2_cnt=0

                for box,cls,conf in zip(
                        boxes.xyxy,
                        boxes.cls,
                        boxes.conf):

                    cls=int(cls)
                    conf=float(conf)

                    if conf<th or cls not in target_ids:
                        continue

                    x1,y1,x2,y2=map(int,box)
                    cx=int((x1+x2)/2)
                    cy=int((y1+y2)/2)

                    cname=class_names[cls].lower()

                    if cname=='car': car_cnt+=1
                    elif cname=='motorcycle': motorcycle_cnt+=1
                    elif cname=='bus': bus_cnt+=1

                    if inside_roi((cx,cy),roi1):
                        roi1_cnt+=1
                    if inside_roi((cx,cy),roi2):
                        roi2_cnt+=1

                total_cnt=car_cnt+motorcycle_cnt+bus_cnt

                density_roi1=(roi1_cnt*1000)/(roi1_len*lanes_roi1) if roi1_len>0 else 0
                density_roi2=(roi2_cnt*1000)/(roi2_len*lanes_roi2) if roi2_len>0 else 0

                data[th].append([
                    time_sec,car_cnt,motorcycle_cnt,bus_cnt,
                    total_cnt,roi1_cnt,roi2_cnt,
                    density_roi1,density_roi2
                ])

                # -------- detection image per confidence --------
                det=model(frame,conf=th)[0]
                annotated=det.plot()

                cv2.polylines(annotated,[roi1],True,(0,255,0),3)
                cv2.polylines(annotated,[roi2],True,(255,0,0),3)

                cv2.imwrite(
                    str(detect_folders[th]/frame_name),
                    annotated
                )

        print(f"Processed second {time_counter}")

        time_counter+=1   # ✅ integer increment

    frame_index+=1

cap.release()

# ---------------- EXCEL ---------------- #
excel_path=desktop/f"vehicle_density_{timestamp}.xlsx"

with pd.ExcelWriter(excel_path,engine='openpyxl') as writer:

    for th in thresholds:

        df=pd.DataFrame(data[th],columns=[
            'Time (s)','Car','Motorcycle','Bus','Total',
            'ROI1 Count','ROI2 Count',
            'Density ROI1 (veh/km/lane)',
            'Density ROI2 (veh/km/lane)'
        ])

        df.to_excel(writer,
                    sheet_name=f"conf_{str(th).replace('.','_')}",
                    index=False)

print("Excel saved:",excel_path)