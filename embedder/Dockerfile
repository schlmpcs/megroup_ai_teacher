# CUDA 12.1 + PyTorch 2.3.1 runtime, suitable for Ampere sm_86 (RTX 3060).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Cache Hugging Face model downloads here (mount as a volume to persist).
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
