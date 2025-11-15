"""
AgriVision Webcam Demo
-----------------------
Supports:
1️⃣ Local OpenCV webcam live detection
2️⃣ Streamlit Cloud browser camera snapshot mode

Press 'q' to quit in local mode.
"""

import os
import argparse
import time
import pandas as pd
from datetime import datetime

from ultralytics import YOLO

RUNNING_IN_STREAMLIT = os.getenv("STREAMLIT_RUNTIME") is not None

if RUNNING_IN_STREAMLIT:
    import streamlit as st
    from PIL import Image
    import numpy as np


# --------------------------------------------------
# 🔵 LOCAL MODE (OpenCV Live Webcam)
# --------------------------------------------------
def local_mode(source=0, model_path="model/best.pt", conf=0.25, imgsz=640,
               save_csv=False, out_csv="webcam_detections.csv"):

    import cv2

    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not access webcam/source: {source}")
        return

    detections = []
    frame_idx = 0

    print("[INFO] Starting live detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] No frame received from webcam.")
            time.sleep(0.1)
            continue

        # Run YOLO
        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)

        # Draw detections
        try:
            annotated = results[0].plot()
        except:
            annotated = frame

        cv2.imshow("AgriVision - Live Detection (Press 'q' to quit)", annotated)

        # Save detections to list
        if results and hasattr(results[0], "boxes"):
            res = results[0]
            names = res.names if hasattr(res, "names") else model.names

            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].item())
                confv = float(res.boxes.conf[i].item())
                xyxy = res.boxes.xyxy[i].tolist()

                detections.append({
                    "Frame": frame_idx,
                    "Timestamp": datetime.now(),
                    "Class": names.get(cls_id, str(cls_id)),
                    "Confidence": round(confv, 4),
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2)
                })

                print(f"[DETECTED] {names.get(cls_id)} ({confv:.2f})")

        frame_idx += 1

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Webcam closed.")
    cap.release()
    cv2.destroyAllWindows()

    # Save detections
    if save_csv and detections:
        df = pd.DataFrame(detections)
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved detections to {out_csv}")


# --------------------------------------------------
# 🔵 STREAMLIT MODE (Browser Camera Snapshot)
# --------------------------------------------------
def streamlit_mode(model_path="model/best.pt", conf=0.25, imgsz=640):
    st.title("📸 AgriVision – Browser Camera Mode")

    model = YOLO(model_path)

    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

        # Convert to NumPy array (RGB) → then to BGR
        img_array = np.array(image)
        img_bgr = img_array[:, :, ::-1]

        results = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)

        annotated = results[0].plot()
        annotated_rgb = annotated[:, :, ::-1]

        st.image(annotated_rgb, caption="Detections", use_column_width=True)

        # Show detection details
        res = results[0]
        names = res.names if hasattr(res, "names") else model.names

        if res.boxes:
            st.subheader("📌 Detected Issues")
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].item())
                confv = float(res.boxes.conf[i].item())
                st.write(f"**{names.get(cls_id)}** — {confv:.2f}")
        else:
            st.info("No issues detected.")


# --------------------------------------------------
# 🔵 ENTRY POINT
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or RTSP/HTTP URL")
    parser.add_argument("--model", type=str, default="model/best.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", help="Save detections to CSV (local only)")
    args = parser.parse_args()

    if RUNNING_IN_STREAMLIT:
        streamlit_mode(args.model, args.conf, args.imgsz)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        local_mode(src, args.model, args.conf, args.imgsz, args.save)


if __name__ == "__main__":
    main()
