# --------------------------------------------------------
# 🐍 AgriVision Dockerfile
# --------------------------------------------------------
# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# --------------------------------------------------------
# 🧱 Install system dependencies required by OpenCV & Ultralytics
# --------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# 📦 Install Python dependencies
# --------------------------------------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# 🌐 Expose Streamlit port
# --------------------------------------------------------
EXPOSE 8501

# --------------------------------------------------------
# 🚀 Run the Streamlit app
# --------------------------------------------------------
CMD ["streamlit", "run", "field_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
