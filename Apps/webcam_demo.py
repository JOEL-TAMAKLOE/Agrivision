# webcam_demo.py
"""
Low-latency local demo for AgriVision.
Runs YOLOv8 inference on a webcam or RTSP/HTTP stream using OpenCV windows (cv2.imshow).
Press 'q' to quit. Optionally save detections to CSV with --save.
"""

from ultralytics import YOLO
import cv2
import time
import pandas as pd
from datetime import datetime
import argparse
import os

def main(source=0, model_path="models/best.pt", conf=0.25, imgsz=640, save_csv=False, out_csv="webcam_detections.csv"):
    # Load model
    model = YOLO(model_path)

    # Open video source
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
                # Wait briefly for network cameras or hiccups
                time.sleep(0.1)
                continue

            # Run YOLO inference (frame is BGR numpy array)
            results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)

            # Annotated overlay (BGR)
            try:
                overlay = results[0].plot()
            except Exception:
                overlay = frame

            # Show overlay window
            cv2.imshow("AgriVision - Live Demo (press q to quit)", overlay)

            # Collect detection rows (if any)
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

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_csv and len(detections) > 0:
            df = pd.DataFrame(detections)
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved detections to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or RTSP/HTTP URL")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save", action="store_true", help="Save detections CSV on exit")
    args = parser.parse_args()

    # If source looks like a digit, convert to int for webcam
    src = int(args.source) if args.source.isdigit() else args.source
    main(source=src, model_path=args.model, conf=args.conf, imgsz=args.imgsz, save_csv=args.save)
