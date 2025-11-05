🌾 AgriVision

AI-Powered Smart Detection of Crop Stress & Pests
AgriVision uses YOLOv8 to detect abiotic stress, insects, and plant diseases in agricultural images and drone videos.
It runs on laptop & mobile (PWA), works offline, and can be deployed on the cloud (Streamlit Sharing / Docker).
---
📂 Project Structure
agrivision/
│── field_app.py          # Streamlit app (image + video detection)
│── requirements.txt      # Python dependencies
│── Dockerfile            # Container build file
│── .dockerignore         # Ignore unnecessary files in Docker builds
│── manifest.json         # PWA manifest
│── service-worker.js     # PWA service worker
│── models/
│   └── best.pt           # Trained YOLOv8 model (add this after training)
│── assets/
│   ├── icon-192.png      # PWA icon (192px)
│   └── icon-512.png      # PWA icon (512px)
│── notebooks/
│   └── training.ipynb    # Training process (YOLOv8 notebook)
---
🧠 Add Your Trained Model (best.pt)

The app needs the trained YOLOv8 weights to make detections.

Option 1: Download from Colab

Train the model using notebooks/training.ipynb.

After training, YOLO saves best.pt (usually at /content/runs/detect/trainX/weights/best.pt).

Download it from Colab:

from google.colab import files
files.download('/content/runs/detect/train7/weights/best.pt')


Move it into the models/ folder:

agrivision/models/best.pt

Option 2: Copy Directly in Colab
!cp /content/runs/detect/train7/weights/best.pt /content/agrivision/models/
---
⚙️ Installation
Local Setup
git clone https://github.com/yourusername/agrivision.git
cd agrivision
pip install -r requirements.txt
streamlit run field_app.py


Open in your browser at: http://localhost:8501
---
🐳 Run with Docker

Build the image:

docker build -t agrivision .


Run the container:

docker run -p 8501:8501 agrivision


App will be available at: http://localhost:8501

📱 Install as PWA (Offline Support)

AgriVision is PWA-enabled, so you can install it like an app.

Run locally or deploy (e.g., Streamlit Cloud / Docker server).

Open the app in Chrome/Edge/Brave.

Click “Install App” (in browser menu).

The app now works like a native app with offline caching.
---
🎥 Features

✅ Detects crop diseases, pests, abiotic stress
✅ Works on images & videos
✅ Replay mode with video + map sync
✅ GPS integration (parse .srt sidecar files from drones)
✅ CSV export of detections
✅ Mobile + Laptop support (PWA responsive design)
✅ Runs offline (PWA + Docker)

📸 Screenshots

🔹 App Home


🔹 Image Detection


🔹 Video + Map Replay


📌 Roadmap

 Add real-time drone video streaming

 Train with larger datasets for better accuracy

 Integrate weather & soil sensor data

 Deploy on mobile edge devices
---
🤝 Contributing

Pull requests are welcome!

📜 License

MIT License 