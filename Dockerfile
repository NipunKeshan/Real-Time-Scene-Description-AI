# Real-Time AI Scene Description System
# Multi-stage Docker build for optimized production deployment

# Base stage with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Set up Python environment
ENV PATH="/home/app/.local/bin:$PATH"
RUN python3 -m pip install --upgrade pip

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Install development dependencies
RUN pip install --user \
    jupyter \
    notebook \
    ipywidgets \
    jupyter-contrib-nbextensions

# Expose ports
EXPOSE 8501 8000 8888

# Default command for development
CMD ["python3", "-m", "src.main", "--mode", "streamlit"]

# Production stage
FROM base as production

# Copy requirements and install production dependencies only
COPY --chown=app:app requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Copy application code
COPY --chown=app:app src/ src/
COPY --chown=app:app config/ config/
COPY --chown=app:app README.md .

# Create necessary directories
RUN mkdir -p logs outputs models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "src.main", "--mode", "api"]

# API-only stage (minimal footprint)
FROM python:3.9-slim as api-only

ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Install Python dependencies
COPY --chown=app:app requirements.txt .
RUN pip install --user --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pillow \
    numpy \
    torch \
    transformers \
    && python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Copy only API-related code
COPY --chown=app:app src/api/ src/api/
COPY --chown=app:app src/models/ src/models/
COPY --chown=app:app src/utils/ src/utils/
COPY --chown=app:app config/ config/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
