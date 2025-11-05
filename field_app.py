import streamlit as st
from ultralytics import YOLO
import cv2, time, pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime

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
model = YOLO("models/best.pt")

st.title("🌾 AgriVision: Smart Detection of Crop Stress & Pests")
st.write("Detects abiotic stress, insects, and diseases from field images or drone footage.")

# ----------------------------
# Upload video
# ----------------------------
video_source = st.file_uploader("📂 Upload drone video (mp4/avi)", type=["mp4", "avi", "mov"])
gps_per_frame = []  # Placeholder for GPS data if available

# ----------------------------
# Run detection + store results
# ----------------------------
if video_source:
    cap = cv2.VideoCapture(video_source.name)
    detections, frames, frame_index = [], [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        frames.append(frame)

        # Optional GPS (extend later)
        latitude, longitude = (None, None)
        if frame_index < len(gps_per_frame):
            latitude, longitude, alt = gps_per_frame[frame_index]

        # Save detections
        for r in results[0].boxes:
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])
            detections.append([
                frame_index,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model.names[cls_id],
                round(conf, 2),
                latitude, longitude
            ])
        frame_index += 1
    cap.release()

    # Convert to DataFrame
    df = pd.DataFrame(detections, columns=["Frame", "Timestamp", "Class", "Confidence", "Latitude", "Longitude"])

    # ----------------------------
    # Replay Mode: Video + Map
    # ----------------------------
    st.subheader("🎥 Replay Mode (Video + Map Sync)")

    max_frame = df["Frame"].max()
    replay_frame = st.slider("Replay Frame", 0, int(max_frame), 0, 1)
    play = st.checkbox("▶️ Auto Play")

    video_col, map_col = st.columns([1.5, 1.5])

    # Show video frame
    if replay_frame < len(frames):
        video_col.image(frames[replay_frame], channels="BGR", caption=f"Frame {replay_frame}")

    # Show detections on map
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

    # Auto-play loop
    if play:
        for f in range(replay_frame, max_frame):
            if f < len(frames):
                video_col.image(frames[f], channels="BGR", caption=f"Frame {f}")
            time.sleep(0.2)
