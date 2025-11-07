# --------------------------------------------------------
# 🐍 AgriVision Dockerfile
# --------------------------------------------------------
# Use an official lightweight Python image
FROM python:3.10-slim

# --------------------------------------------------------
# 📁 Set the working directory inside the container
# --------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------
# ⚙️ Install system dependencies required by OpenCV, Ultralytics & Pillow
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# 📦 Copy only requirements file first (for better caching)
# --------------------------------------------------------
COPY requirements.txt .

# --------------------------------------------------------
# 🧰 Upgrade pip safely
# --------------------------------------------------------
RUN pip install --upgrade pip

# --------------------------------------------------------
# 🔧 Install Python dependencies
# --------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# 📂 Copy the rest of the project files
# --------------------------------------------------------
COPY . .

# --------------------------------------------------------
# 🌐 Expose Streamlit port
# --------------------------------------------------------
EXPOSE 8501

# --------------------------------------------------------
# 🚀 Run the Streamlit app
# --------------------------------------------------------
CMD ["streamlit", "run", "field_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
