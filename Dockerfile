FROM python:3.12-slim

# Avoid Python bytecode & buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create workdir
WORKDIR /app

# ✅ Install OS dependencies for OpenCV
# - libgl1: fixes libGL.so.1 missing error
# - libglib2.0-0: some codecs & image ops
# - libsm6 libxext6 libxrender1: common X11 image libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create venv
RUN python -m venv /opt/venv

# Use venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command
CMD ["python", "main.py"]