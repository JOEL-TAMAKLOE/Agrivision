# Apps/field_app.py
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"


# ----------------------------
# SAFE global fix to prevent NameError
# ----------------------------
chosen = None
# ----------------------------

import streamlit as st
from ultralytics import YOLO
import cv2
import time
import pandas as pd
import numpy as np
import re
import subprocess
import platform
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime
from PIL import Image
from tempfile import NamedTemporaryFile
import html
import pathlib
import base64
from PIL import Image
import os
from io import BytesIO

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def get_app_dir():
    """Return directory where this file lives (reliable for script-relative resources)."""
    return os.path.dirname(__file__)

def is_running_locally():
    """
    Decide if app is running locally (developer machine) or in Streamlit Cloud.
    Conservative heuristic.
    """
    home = os.environ.get("HOME", "")
    if "appuser" in home or "streamlit" in home:
        return False
    if platform.system().lower() in ("windows", "darwin"):
        return True
    return True

def results_to_rows(results, frame_idx=0, gps=None, source_file=None):
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
            row = [
                frame_idx,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                names.get(cls_id, str(cls_id)),
                round(conf, 3),
                *[round(v, 2) for v in xyxy],
                lat, lon
            ]
            if source_file:
                row.append(source_file)
            rows.append(row)
    return rows

def parse_srt_for_gps_and_altitude(srt_text):
    """
    Parse SRT text for GPS and optional Altitude.
    Returns list of tuples (lat, lon, alt_or_none) indexed by caption order.
    """
    gps_entries = []
    latlon_pattern = re.compile(r"GPS[:\s]*([-\d\.]+)[,\s]+([-\d\.]+)")
    alt_pattern = re.compile(r"Altitude[:\s]*([-\d\.]+)")
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for blk in blocks:
        lat, lon, alt = None, None, None
        m = latlon_pattern.search(blk)
        if m:
            try:
                lat = float(m.group(1)); lon = float(m.group(2))
            except:
                lat, lon = None, None
        m2 = alt_pattern.search(blk)
        if m2:
            try:
                alt = float(m2.group(1))
            except:
                alt = None
        if lat is not None and lon is not None:
            gps_entries.append((lat, lon, alt))
    return gps_entries

def height_advisory_bar(altitude_m: float):
    """Render a small HTML bar to advise about altitude."""
    if altitude_m is None:
        st.info("No altitude metadata found for this video.")
        return
    max_height = 60.0
    pct = min(max(altitude_m / max_height * 100.0, 0.0), 100.0)
    if altitude_m < 10:
        color = "#e63946"  # red
        status = f"Too Low — may miss context ({altitude_m:.1f} m)"
    elif altitude_m > 30:
        color = "#f4a261"  # orange
        status = f"Too High — may reduce fine-detail detection ({altitude_m:.1f} m)"
    else:
        color = "#2a9d8f"  # green
        status = f"Optimal altitude for crop-level detection ({altitude_m:.1f} m)"
    st.markdown(f"""
    <div style='width:100%;background:#e9ecef;border-radius:8px;height:14px;'>
      <div style='width:{pct}%;background:{color};height:14px;border-radius:8px;'></div>
    </div>
    <div style='text-align:center;font-weight:600;margin-top:6px;color:{color}'>{status}</div>
    """, unsafe_allow_html=True)

def slugify_filename(name: str):
    """Create a safe filename base from the provided name (without extension)."""
    base = pathlib.Path(name).stem
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', base)
    return safe

def make_csv_name(source_name: str, suffix="_detections.csv"):
    """Return csv filename derived from source_name (or default timestamp)."""
    if source_name:
        base = slugify_filename(source_name)
    else:
        base = f"agri_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"{base}{suffix}"

# ------------------------------------------------------------------
# Color palette + styles (light + dark)
# ------------------------------------------------------------------
PALETTE = {
    "primary": "#2a9d8f",
    "primary_dark": "#21867a",
    "accent": "#f4a261",
    "danger": "#e63946",
    "muted": "#6c757d",
    "bg_light": "#f6f8fa",
    "card": "#ffffff",
    "text": "#111827",
    "dark_bg": "#0b1220",
    "dark_card": "#0f1724",
    "dark_text": "#e6eef6"
}

def inject_css(dark_mode: bool):
    if dark_mode:
        css = f"""
        <style>
        :root{{ --ag-primary: {PALETTE['primary']}; --bg: {PALETTE['dark_bg']}; --card: {PALETTE['dark_card']}; --text: {PALETTE['dark_text']}; }}
        [data-testid="stSidebar"] {{ background-color: #071122; border-right: 1px solid rgba(255,255,255,0.03); }}
        .stApp {{ background: linear-gradient(180deg, #071122 0%, #071a2a 100%); color: var(--text); }}
        .card {{ background: var(--card); border-radius: 12px; padding: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }}
        .logo-title {{"font-family":"Segoe UI", sans-serif}}
        .stButton > button {{ background: {PALETTE['primary']}; color: white; border-radius: 10px; }}
        .uploader {{ border: 2px dashed {PALETTE['primary']}; border-radius: 12px; padding: 12px; }}
        hr{{ border-color: rgba(255,255,255,0.06) }}
        </style>
        """
    else:
        css = f"""
        <style>
        :root{{ --ag-primary: {PALETTE['primary']}; --bg: {PALETTE['bg_light']}; --card: {PALETTE['card']}; --text: {PALETTE['text']}; }}
        [data-testid="stSidebar"] {{ background-color: {PALETTE['bg_light']}; border-right: 1px solid #e6e6e6; }}
        .stApp {{ background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); color: var(--text); }}
        .card {{ background: var(--card); border-radius: 12px; padding: 12px; box-shadow: 0 6px 18px rgba(16,24,40,0.04); }}
        .stButton > button {{ background: {PALETTE['primary']}; color: white; border-radius: 10px; }}
        .uploader {{ border: 2px dashed {PALETTE['primary']}; border-radius: 12px; padding: 12px; }}
        hr{{ border-color: #eee }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(get_app_dir(), "..", "model", "best.pt")
        model_path = os.path.normpath(model_path)
        if not os.path.exists(model_path):
            alt = os.path.join(os.getcwd(), "model", "best.pt")
            if os.path.exists(alt):
                model_path = alt
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Could not load YOLO model at model/best.pt. Please add it. Error: {e}")
        st.stop()

# load once
model = load_model()

# ------------------------------------------------------------------
# Page config + header
# ------------------------------------------------------------------
st.set_page_config(page_title="AgriVision", layout="wide", initial_sidebar_state="expanded")

# Theme control (Dark / Light)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Top bar with logo + animation
def show_header():
    # Folder where your images live
    img_dir = os.path.join(os.getcwd(), "images")

    # Pick the image you want
    logo_filename = "image2.png"   # ← change this when needed
    logo_path = os.path.join(img_dir, logo_filename)

    if not os.path.exists(logo_path):
        st.error(f"⚠️ Logo not found: {logo_path}")
        return

    # Load logo image
    img = Image.open(logo_path)

    # Resize to match header height
    img = img.resize((70, 70))  # Adjust if needed (height 70px)

    # Convert to base64 for embedding
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode()

    # HTML block with round logo + fade-in animation
    header_html = f"""
    <div style="display:flex;align-items:center;gap:16px;animation:fadeIn 0.9s ease-in-out;">
        <img src="data:image/png;base64,{base64_img}" 
             style="height:70px;width:70px;border-radius:50%;object-fit:cover;
                    box-shadow:0 4px 12px rgba(0,0,0,0.15);" />

        <div style="line-height:1.05">
            <h1 style="margin:0;padding:0;color:var(--ag-primary);">🌾 AgriVision</h1>
            <div style="color:var(--text);font-weight:600">
                Smart Detection of Crop Stress & Pests
            </div>
        </div>
    </div>

    <style>
    @keyframes fadeIn {{
        from {{ opacity:0; transform: translateY(-6px); }}
        to {{ opacity:1; transform: translateY(0); }}
    }}
    </style>
    """

    st.markdown(header_html, unsafe_allow_html=True)


# inject CSS theme
inject_css(st.session_state.dark_mode)
show_header()

# small UI controls row
col_a, col_b, col_c = st.columns([0.6, 0.6, 2.8])
with col_a:
    # Dark mode toggle
    dm = st.checkbox("Dark mode", value=st.session_state.dark_mode)
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        inject_css(st.session_state.dark_mode)
        st.experimental_rerun()
with col_b:
    st.markdown("")  # placeholder
with col_c:
    st.markdown("<div style='text-align:right;color:var(--muted)'>Built for field use • Offline capable • Lightweight</div>", unsafe_allow_html=True)

st.write("")  # spacing

st.write("Detects abiotic stress, insects, and diseases from field **images**, **drone footage**, or **live webcam feeds**.")

# Sidebar settings
st.sidebar.header("⚙️ Inference Settings")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
frame_skip = st.sidebar.number_input("Sample every Nth frame (video)", min_value=1, max_value=30, value=3)

# small description under settings
st.sidebar.markdown("<small style='color:var(--muted)'>Lower confidence finds more candidates; lower image size is faster.</small>", unsafe_allow_html=True)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Mode
mode = st.radio("Choose input type:", ["📸 Image", "🎥 Video", "🎦 Live Webcam"], horizontal=True)

# ----------------------------
# IMAGE MODE
# ----------------------------
if mode == "📸 Image":
    image_source = st.file_uploader("📷 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"], help="Drag & drop or click to upload")
    if image_source:
        try:
            pil_img = Image.open(image_source).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        # Convert PIL -> NumPy (RGB) -> BGR (expected by YOLO)
        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Run prediction
        with st.spinner("Running detection..."):
            results = model.predict(img_np, conf=conf_thres, imgsz=imgsz, verbose=False)

        # Show original and detections
        col1, col2 = st.columns(2)
        col1.image(pil_img, caption=f"Original — {image_source.name}", use_column_width=True)

        try:
            plotted = results[0].plot()  # BGR image
            display_img = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            col2.image(display_img, caption="Detections", use_column_width=True)
        except Exception:
            col2.info("Could not render overlay; showing original.")
            col2.image(pil_img, use_column_width=True)

        # Export detections
        file_name_used = getattr(image_source, "name", None)
        rows = results_to_rows(results, frame_idx=0, source_file=file_name_used)
        if rows:
            df_img = pd.DataFrame(rows, columns=[
                "Frame", "Timestamp", "Class", "Confidence", "x1", "y1", "x2", "y2", "Latitude", "Longitude", "Source_File"
            ])
            st.subheader("📑 Detections")
            st.dataframe(df_img, use_container_width=True)
            # create download filename
            csv_name = make_csv_name(file_name_used)
            st.download_button("⬇️ Download detections (CSV)", df_img.to_csv(index=False).encode("utf-8"),
                               file_name=csv_name, mime="text/csv")
        else:
            st.info("No detections above the confidence threshold.")

# ----------------------------
# VIDEO MODE
# ----------------------------
elif mode == "🎥 Video":
    video_file = st.file_uploader("📂 Upload drone video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    srt_file = st.file_uploader("(Optional) Upload GPS sidecar (SRT file)", type=["srt"])
    gps_per_frame = []

    # Parse SRT if available for GPS & altitude
    avg_altitude = None
    file_name_used = None
    if srt_file:
        try:
            srt_text = srt_file.read().decode("utf-8", errors="ignore")
            gps_entries = parse_srt_for_gps_and_altitude(srt_text)
            if gps_entries:
                gps_per_frame = [(lat, lon) for (lat, lon, alt) in gps_entries]
                alts = [alt for (_, _, alt) in gps_entries if alt is not None]
                if alts:
                    avg_altitude = float(np.mean(alts))
                st.sidebar.success(f"Parsed {len(gps_entries)} GPS entries from SRT.")
        except Exception as e:
            st.sidebar.warning(f"Could not parse SRT file: {e}")

    if video_file:
        file_name_used = getattr(video_file, "name", None)
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        frames = []
        detections = []
        frame_idx = 0
        process_status = st.empty()
        process_status.info(f"⏳ Processing video (every {frame_skip} frame)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                # frame is BGR from OpenCV — ok to pass directly
                results = model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)
                try:
                    overlay = results[0].plot()  # BGR image
                except Exception:
                    overlay = frame.copy()
                frames.append(overlay)
                gps_tuple = gps_per_frame[frame_idx] if frame_idx < len(gps_per_frame) else None
                rows = results_to_rows(results, frame_idx, gps_tuple, source_file=file_name_used)
                detections.extend(rows)
            frame_idx += 1

        cap.release()
        os.remove(tmp_path)
        process_status.empty()

        if not detections:
            st.warning("No detections found above the threshold.")
            st.stop()

        df = pd.DataFrame(detections, columns=[
            "Frame", "Timestamp", "Class", "Confidence", "x1", "y1", "x2", "y2", "Latitude", "Longitude", "Source_File"
        ])

        # Show optional altitude advisory
        if avg_altitude is not None:
            st.subheader("📈 Flight altitude advisory")
            height_advisory_bar(avg_altitude)

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
        csv_name = make_csv_name(file_name_used if file_name_used else "video")
        st.download_button("⬇️ Download detections (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=csv_name, mime="text/csv")

# ----------------------------
# LIVE WEBCAM MODE
# ----------------------------
else:
    st.subheader("🎦 Live Webcam Detection Mode")
    st.markdown("Detect crop issues in real-time using your webcam.")

    local = is_running_locally()

    if not local:
        # Cloud mode — browser snapshot
        st.info("🌐 Running in cloud mode. Use browser camera snapshots for detection.")
        camera_input = st.camera_input("Take a snapshot")
        if camera_input:
            try:
                pil_img = Image.open(camera_input).convert("RGB")
                img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                with st.spinner("Running detection..."):
                    results = model.predict(img_np, conf=conf_thres, imgsz=imgsz, verbose=False)
                plotted = results[0].plot()
                st.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
            except Exception as e:
                st.error(f"Error during snapshot detection: {e}")
        st.warning("⚠️ Full live webcam feed is only available when running locally. Offline LiveCam is disabled here.")

    else:
        # Local mode — allow in-app live feed and offline script
        st.success("💻 Running locally. You can use your webcam in live feed mode or open the offline LiveCam app.")

        # Initialize session state for streaming
        if "streaming" not in st.session_state:
            st.session_state.streaming = False

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Start in-app Live Feed:**")
            if not st.session_state.streaming:
                if st.button("🎥 Start Live Feed (in-app)"):
                    st.session_state.streaming = True
                    st.experimental_rerun()  # re-render to show stop button
            else:
                if st.button("⏹ Stop Live Feed"):
                    st.session_state.streaming = False
                    st.experimental_rerun()

            # If streaming, display frames
            if st.session_state.streaming:
                placeholder = st.empty()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.warning("⚠️ Could not access webcam.")
                else:
                    try:
                        while st.session_state.streaming and cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("⚠️ Could not read frame from webcam.")
                                break
                            # frame is BGR already
                            results = model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)
                            try:
                                overlay = results[0].plot()
                            except Exception:
                                overlay = frame
                            placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
                            time.sleep(0.03)
                    finally:
                        cap.release()
                        placeholder.empty()

        with colB:
            st.markdown("**Or open the dedicated offline LiveCam app:**")
            script_path = os.path.join(get_app_dir(), "webcam_demo.py")
            if os.path.exists(script_path):
                if st.button("🧭 Run Offline LiveCam (webcam_demo.py)"):
                    # spawn a new process locally (works only on local machines)
                    try:
                        subprocess.Popen(["python", script_path])
                        st.success("✅ Offline LiveCam opened in a new window (local only).")
                    except Exception as e:
                        st.error(f"Could not launch offline app: {e}")
            else:
                st.info("❌ webcam_demo.py not found in Apps/. Place webcam_demo.py alongside this file for offline mode.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<hr style='margin-top:32px;'>
<div style='display:flex;justify-content:space-between;align-items:center'>
  <div style='font-size:14px;color:#6b7280'>Developed & Built by <strong>Joel Tamakloe</strong> ❤️</div>
</div>
<br>
""", unsafe_allow_html=True)
# End of file
