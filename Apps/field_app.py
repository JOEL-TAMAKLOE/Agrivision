import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
import streamlit as st
from ultralytics import YOLO
import cv2, time, pandas as pd, numpy as np, os, re, platform, subprocess
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime
from PIL import Image
from tempfile import NamedTemporaryFile

# ----------------------------
# Inject PWA manifest + service worker
# ----------------------------
st.markdown("""
<link rel="manifest" href="/manifest.json">
<script>
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register("/service-worker.js")
    .then(() => console.log("✅ Service Worker Registered"));
}
</script>
""", unsafe_allow_html=True)

# ----------------------------
# Load YOLO model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        return YOLO("models/best.pt")
    except Exception as e:
        st.error(f"❌ Could not load YOLO model at models/best.pt. Please add it. Error: {e}")
        st.stop()

model = load_model()

# ----------------------------
# App title
# ----------------------------
st.title("🌾 AgriVision: Smart Detection of Crop Stress & Pests")
st.write("Detects abiotic stress, insects, and diseases from field **images**, **drone footage**, or **live webcam feeds**.")

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("⚙️ Inference Settings")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
frame_skip = st.sidebar.number_input("Sample every Nth frame (video)", min_value=1, max_value=30, value=3)

# ----------------------------
# Helper: Convert YOLO results -> DataFrame rows
# ----------------------------
def results_to_rows(results, frame_idx=0, gps=None):
    rows = []
    boxes = results[0].boxes
    names = results[0].names if hasattr(results[0], "names") else model.names
    lat, lon = (None, None)
    if gps is not None:
        lat, lon = gps

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].tolist()
            rows.append([
                frame_idx,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                names.get(cls_id, str(cls_id)),
                round(conf, 3),
                *[round(v, 2) for v in xyxy],
                lat, lon
            ])
    return rows

# ----------------------------
# Mode selection
# ----------------------------
mode = st.radio("Choose input type:", ["📸 Image", "🎥 Video", "🎦 Live Webcam"], horizontal=True)

# ----------------------------
# IMAGE MODE
# ----------------------------
if mode == "📸 Image":
    image_source = st.file_uploader("📷 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if image_source:
        img = Image.open(image_source).convert("RGB")
        results = model.predict(img, conf=conf_thres, imgsz=imgsz, verbose=False)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_column_width=True)

        plotted = results[0].plot()
        col2.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)

        rows = results_to_rows(results, frame_idx=0)
        if rows:
            df_img = pd.DataFrame(rows, columns=[
                "Frame", "Timestamp", "Class", "Confidence", "x1", "y1", "x2", "y2", "Latitude", "Longitude"
            ])
            st.subheader("📑 Detections")
            st.dataframe(df_img, use_container_width=True)
            st.download_button("⬇️ Download detections (CSV)",
                               df_img.to_csv(index=False).encode("utf-8"),
                               file_name="image_detections.csv",
                               mime="text/csv")
        else:
            st.info("No detections above the confidence threshold.")

# ----------------------------
# VIDEO MODE
# ----------------------------
elif mode == "🎥 Video":
    video_file = st.file_uploader("📂 Upload drone video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    srt_file = st.file_uploader("(Optional) Upload GPS sidecar (SRT file)", type=["srt"])
    gps_per_frame = []

    # Parse GPS from SRT if available
    if srt_file:
        try:
            srt_text = srt_file.read().decode("utf-8", errors="ignore")
            gps_per_frame = []
            for match in re.finditer(r"GPS:\\s*([-\d\\.]+),\\s*([-\d\\.]+)", srt_text):
                gps_per_frame.append((float(match.group(1)), float(match.group(2))))
            st.sidebar.success(f"Parsed {len(gps_per_frame)} GPS entries from SRT.")
        except Exception as e:
            st.sidebar.warning(f"Could not parse SRT file: {e}")

    if video_file:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        frames, detections, frame_idx = [], [], 0
        process_status = st.empty()
        process_status.info(f"⏳ Processing video (every {frame_skip} frame)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                results = model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)
                overlay = results[0].plot()
                frames.append(overlay)
                gps_tuple = gps_per_frame[frame_idx] if frame_idx < len(gps_per_frame) else None
                detections.extend(results_to_rows(results, frame_idx, gps_tuple))
            frame_idx += 1

        cap.release()
        os.remove(tmp_path)
        process_status.empty()

        if not detections:
            st.warning("No detections found above the threshold.")
            st.stop()

        df = pd.DataFrame(detections, columns=[
            "Frame", "Timestamp", "Class", "Confidence", "x1", "y1", "x2", "y2", "Latitude", "Longitude"
        ])

        st.subheader("🎥 Replay Mode (Video + Map Sync)")
        max_frame = int(df["Frame"].max())
        replay_frame = st.slider("Replay Frame", 0, max_frame, 0, 1)
        play = st.checkbox("▶️ Auto Play")

        video_col, map_col = st.columns([1.5, 1.5])

        if replay_frame < len(frames):
            video_col.image(cv2.cvtColor(frames[replay_frame], cv2.COLOR_BGR2RGB),
                            caption=f"Frame {replay_frame}", use_column_width=True)

        df_replay = df[df["Frame"] <= replay_frame]
        if not df_replay.empty and df_replay["Latitude"].notna().any():
            m = folium.Map(
                location=[df_replay["Latitude"].dropna().mean(), df_replay["Longitude"].dropna().mean()],
                zoom_start=15
            )
            for _, row in df_replay.dropna(subset=["Latitude", "Longitude"]).iterrows():
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=6, color="red", fill=True, fill_opacity=0.6,
                    popup=f"{row['Class']} ({row['Confidence']})"
                ).add_to(m)
            HeatMap(df_replay.dropna(subset=["Latitude", "Longitude"])[["Latitude", "Longitude"]].values.tolist(),
                    radius=15, blur=10).add_to(m)
            st_folium(m, width=600, height=400)

        if play:
            for f in range(replay_frame, max_frame + 1):
                if f < len(frames):
                    video_col.image(cv2.cvtColor(frames[f], cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {f}", use_column_width=True)
                time.sleep(0.15)

        st.subheader("📑 Detections Table")
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download detections (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="video_detections.csv",
                           mime="text/csv")

# ----------------------------
# 🎦 LIVE WEBCAM MODE
# ----------------------------
else:
    st.subheader("🎦 Live Webcam Detection Mode")
    st.markdown("Detect crop issues in real-time using your webcam.")

    # Check if running locally (no cloud environment variable)
    is_local = not any(env in os.environ for env in ["STREAMLIT_SERVER_PORT", "STREAMLIT_RUNTIME"])
    
    # 1️⃣ Cloud mode – use browser camera
    if not is_local:
        st.info("🌐 Running in cloud mode. Using browser camera snapshots for detection.")
        camera_input = st.camera_input("Take a snapshot")
        if camera_input:
            img = Image.open(camera_input)
            results = model.predict(img, conf=conf_thres, imgsz=imgsz, verbose=False)
            plotted = results[0].plot()
            st.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

    # 2️⃣ Local mode – offer full live feed + offline script
    else:
        st.success("💻 Running locally. You can use your webcam in live feed mode or open the offline live cam app.")

        colA, colB = st.columns(2)
        with colA:
            if st.button("🎥 Start Live Feed (in-app)"):
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)
                    overlay = results[0].plot()
                    stframe.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")
                    if st.button("⏹ Stop"):
                        break
                cap.release()

        with colB:
            st.markdown("Or open the dedicated offline mode app below:")
            if st.button("🧭 Run Offline LiveCam (webcam_demo.py)"):
                script_path = os.path.join(os.getcwd(), "webcam_demo.py")
                if os.path.exists(script_path):
                    subprocess.Popen(["python", script_path], shell=True)
                    st.success("✅ Offline LiveCam opened in a new window.")
                else:
                    st.error("❌ webcam_demo.py not found. Please ensure it’s in the same directory.")
