# -----------------------
# 🌾 AgriVision Dockerfile
# -----------------------
FROM python:3.10-slim

# Prevents OpenCV/libGL errors
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "Apps/field_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
