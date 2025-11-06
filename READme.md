# 🌾 AgriVision

## Overivew 
AgriVision is an AI-powered field application that uses computer vision (YOLO) to analyze crop health and provide real-time insights
AI-Powered Smart Detection of Crop Stress & Pests
AgriVision uses YOLOv8 to detect abiotic stress, insects, and plant diseases in agricultural images and drone videos.
It runs on laptop & mobile (PWA), works offline, and can be deployed on the cloud (Streamlit Sharing / Docker).

---
## 📂 Project Structure
```plaintext
Agrivision/
│── Apps/                  
     └──                  # Contains the app.py files (image + video detection)
│── requirements.txt      # Python dependencies
│── Dockerfile            # Container build file
│── .dockerignore         # Ignore unnecessary files in Docker builds
│── manifest.json         # PWA manifest
│── service-worker.js     # PWA service worker
│── model/
│   └── best.pt          # Trained YOLOv8 model 
│── images/
│     └──                # contains images
│── notebook/
│   └── training.ipynb    # Training process (YOLOv8 notebook)
```
---

## ⚙️ Installation
Local Setup
**Clone the Repository**:
   ```bash
git clone https://github.com/JOEL-TAMAKLOE/agrivision.git
cd agrivision
pip install -r requirements.txt
streamlit run field_app.py
```

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
## 🎥 Features

✅ Detects crop diseases, pests, abiotic stress
✅ Works on images & videos
✅ Replay mode with video + map sync
✅ GPS integration (parse .srt sidecar files from drones)
✅ CSV export of detections
✅ Mobile + Laptop support (PWA responsive design)
✅ Runs offline (PWA + Docker)


---

## 📸 Screenshots

🔹 App Home


🔹 Image Detection


🔹 Video + Map Replay

---

## 📌 Roadmap
![roadmap](images/visionflow.png)


 Train with larger datasets for better accuracy

 Integrate weather & soil sensor data

 Deploy on mobile edge devices
---
## 🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

---
## 📜 License

MIT License 
