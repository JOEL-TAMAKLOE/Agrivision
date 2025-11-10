"""
AgriVision Webcam Demo
-----------------------
This script supports both:
1️⃣ Local OpenCV-based webcam streaming
2️⃣ Streamlit Cloud-friendly mode using browser camera input

Press 'q' to quit in local mode.
"""

import os
import argparse
import time
import pandas as pd
from datetime import datetime

# Import model
from ultralytics import YOLO

# Detect environment
RUNNING_IN_STREAMLIT = os.getenv("STREAMLIT_RUNTIME") is not None

if RUNNING_IN_STREAMLIT:
    import streamlit as st
    from PIL import Image
    import numpy as np

def local_mode(source=0, model_path="best.pt", conf=0.25, imgsz=640, save_csv=False, out_csv="webcam_detections.csv"):
    import cv2
    model = YOLO("model/best.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    detections = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
            try:
                overlay = results[0].plot()
            except Exception:
                overlay = frame

            cv2.imshow("AgriVision - Live Demo (press q to quit)", overlay)

            if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                res = results[0]
                names = res.names if hasattr(res, "names") else model.names
                for i in range(len(res.boxes)):
                    cls_id = int(res.boxes.cls[i].item())
                    confv = float(res.boxes.conf[i].item())
                    xyxy = res.boxes.xyxy[i].tolist()
                    detections.append({
                        "Frame": frame_idx,
                        "Timestamp": datetime.now(),
                        "Class": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id],
                        "Confidence": round(confv, 4),
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2)
                    })
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_csv and len(detections) > 0:
            df = pd.DataFrame(detections)
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved detections to {out_csv}")

def streamlit_mode(model_path="best.pt", conf=0.25, imgsz=640):
    st.title("📸 AgriVision – Streamlit Camera Mode")
    st.write("Use your browser camera to capture and analyze an image.")

    model = YOLO(model_path)
    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        # Run inference
        results = model.predict(img_array, conf=conf, imgsz=imgsz, verbose=False)
        annotated = results[0].plot()

        st.image(annotated, caption="Prediction", use_column_width=True)

        st.success("✅ Detection complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or RTSP/HTTP URL")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--save", action="store_true", help="Save detections to CSV (local only)")
    args = parser.parse_args()

    if RUNNING_IN_STREAMLIT:
        streamlit_mode(args.model, args.conf, args.imgsz)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        local_mode(src, args.model, args.conf, args.imgsz, args.save)

if __name__ == "__main__":
    main()
