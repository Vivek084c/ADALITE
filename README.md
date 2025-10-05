# ADALITE: Real-Time Monocular Depth Estimation on Raspberry Pi 5 with PREEMPT_RT Kernel

![Depth Prediction](assets/depth.gif)

![Actual Depth Prediction](assets/actual_depth.gif)

## Overview

ADALITE is an end-to-end deep learning pipeline for **real-time monocular depth estimation** optimized for edge devices, specifically the **Raspberry Pi 5**. The project combines knowledge distillation for model training with deployment on a custom **PREEMPT_RT (Real-Time) kernel** to achieve deterministic, low-latency depth prediction from single RGB camera frames.

The system captures live video from the Raspberry Pi Camera Module 3, performs depth estimation using a TensorFlow Lite model, and displays the results directly on the Linux framebuffer for real-time visualization. This project was developed as part of the **CATERPILLAR TECH CHALLENGE 2025**, where it secured a winning position.

### Key Achievements
- **Real-Time Performance**: Achieves ~10-15 FPS on Raspberry Pi 5 with deterministic latency (<200µs under load).
- **Edge-Optimized**: Uses TensorFlow Lite for efficient inference on ARM64 architecture.
- **Real-Time Kernel**: Deployed on a custom PREEMPT_RT kernel ensuring microsecond-level scheduling.
- **Hardware Acceleration**: Leverages Raspberry Pi 5's capabilities for low-power, high-performance depth sensing.

---

## Architecture and Workflow

### Training Pipeline (Knowledge Distillation)
The training phase uses knowledge distillation to create an efficient student model:

1. **Data Ingestion**: Downloads and extracts image datasets from Google Drive.
2. **Teacher Model**: Downloads pre-trained MiDaS TFLite model from KaggleHub as the teacher.
3. **Soft Label Generation**: Teacher model generates depth maps (soft labels) for training images.
4. **Dataset Preparation**: Stores preprocessed images and soft labels in HDF5 format.
5. **Student Model Training**: Trains a custom lightweight Keras model using soft labels.
6. **Model Export**: Exports trained model to TFLite format for deployment.

### Deployment Pipeline (Real-Time Inference)
The deployment script (`Raspberry_Pi_5/18TH_JUNE_TEST_1_NO_HW.py`) handles real-time processing:

1. **Camera Capture**: Uses Picamera2 to capture RGB frames at 640x480, cropped to square.
2. **Preprocessing**: Resizes to 256x256, normalizes with ImageNet mean/std.
3. **Inference**: Runs TFLite model on Raspberry Pi 5 (ARM64).
4. **Postprocessing**: Aligns depth map with calibration constants (m, c).
5. **Visualization**: Generates matplotlib plots with depth points overlay.
6. **Display**: Renders combined original + depth image on Linux framebuffer.

---

## Technical Details

### Model Specifications
- **Model File**: `ADALITE_TFLITE.tflite` (custom distilled model)
- **Input Shape**: 256x256x3 (RGB normalized)
- **Output Shape**: 256x256x1 (depth map)
- **Framework**: TensorFlow Lite 2.18.0
- **Quantization**: FP32 (for precision; can be quantized for further optimization)
- **Inference Threads**: 4 (optimized for Raspberry Pi 5's 4 cores)

### Preprocessing Pipeline
```python
# Color conversion and resizing
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)

# Normalization (ImageNet stats)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
```

### Depth Alignment
```python
# Calibration constants (derived from dataset)
m, c = (-0.37965089082717896, 14.945058822631836)
aligned_depth = m * pred_depth + c
```

### Visualization
- **Colormap**: 'magma' for depth maps
- **Points Overlay**: 6x6 grid of depth values at key locations
- **Display Resolution**: Matches framebuffer (e.g., 1920x1080 for HDMI)
- **Format**: RGB888 for camera, BGRA/BGR565 for framebuffer

### Performance Metrics
- **Target FPS**: 10-15 FPS (measured in real-time)
- **Latency**: <200µs deterministic (with PREEMPT_RT kernel)
- **Memory Usage**: ~200MB RAM during inference
- **CPU Utilization**: ~60-80% on 4 cores

### Hardware Acceleration
- **No Hardware Acceleration in Script**: Current version uses CPU inference (NO_HW variant)
- **Potential Optimizations**: NEON SIMD, OpenCL, or Coral TPU integration for future versions

---

## Requirements

### Hardware
- **Raspberry Pi 5** (8GB RAM recommended)
- **Raspberry Pi Camera Module 3** (with autofocus)
- **64GB microSD Card** (Class 10, UHS-I)
- **Official Raspberry Pi Active Cooler**
- **HDMI Display** (for framebuffer output and debugging)
- **Power Supply**: 27W USB-C PD (official recommended)

### Software
- **Operating System**: Raspberry Pi OS (64-bit, Debian Bookworm)
- **Kernel**: Custom PREEMPT_RT kernel (6.15.0-rc7-v8-16k-NTP+)
- **Python**: 3.12
- **Key Libraries**:
  - TensorFlow Lite Runtime: 2.18.0
  - OpenCV: 4.12.0
  - NumPy: 2.2.6
  - Matplotlib: 3.10.6 (headless mode)
  - Picamera2: Latest
  - Pillow: 11.3.0

### RTOS Kernel Requirements
The project requires a custom PREEMPT_RT kernel for real-time performance. Refer to the dedicated repository for kernel compilation and deployment.

---

## Installation and Setup

### 1. Prepare the RTOS Kernel
**Important**: The real-time performance depends on the PREEMPT_RT kernel. Follow the instructions in the RTOS repository to build and deploy the kernel.

- **RTOS Repository**: [https://github.com/ShekharShwetank/RTOS](https://github.com/ShekharShwetank/RTOS)
- **Kernel Version**: 6.15.0-rc7-v8-16k-NTP+
- **Key Features**: Full preemption, 1000Hz timer, performance governor, NTP/PPS support

Clone and build the RTOS kernel:
```bash
git clone https://github.com/ShekharShwetank/RTOS
cd RTOS
chmod +x build_rt_kernel.sh
./build_rt_kernel.sh
```

### 2. Clone the Main Repository
```bash
git clone <your-adalite-repo-url>
cd ADALITE
```

### 3. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv cat_venv
source cat_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download the Model
Place the trained TFLite model in `Raspberry_Pi_5/ADALITE_TFLITE.tflite`.

### 5. Configure Camera and Permissions
```bash
# Enable camera in raspi-config
sudo raspi-config

# Add user to video group
sudo usermod -a -G video $USER

# Reboot
sudo reboot
```

---

## Usage

### Running the Real-Time Depth Estimation
Navigate to the Raspberry Pi 5 directory and execute the script:
```bash
cd Raspberry_Pi_5
python3 18TH_JUNE_TEST_1_NO_HW.py
```

- **Controls**: Press `Ctrl+C` for graceful shutdown.
- **Output**: Displays live camera feed with depth overlay on HDMI-connected display.
- **Logs**: FPS and status printed to console.

### Training the Model (Optional)
For retraining or customization:
```bash
# Using Docker for training
docker build -t adalite-pipeline .
docker run --rm -v $(pwd)/logs:/app/logs adalite-pipeline

# Or directly
python main.py
```

### Testing with Video File
Modify the script to use `cv2.VideoCapture('input_road.mp4')` instead of Picamera2 for offline testing.

---

## Project Structure
```
ADALITE/
├── assets/                          # Documentation assets
├── config/                          # YAML configurations
├── models/                          # Model architectures
├── models_trained_tflite/           # Exported TFLite models
├── Raspberry_Pi_5/                  # Deployment scripts and models
│   ├── 18TH_JUNE_TEST_1_NO_HW.py    # Main inference script
│   ├── ADALITE_TFLITE.tflite        # Deployed model
│   └── cat_venv/                    # Python virtual environment
├── stages/                          # Training pipeline stages
├── utils/                           # Utility functions
├── Dockerfile                       # Containerization
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Development Log and Full Documentation
For complete development logs, training notebooks, test data, and additional documentation:

- **Full Development Repository**: [https://github.com/ShekharShwetank/PixelPac](https://github.com/ShekharShwetank/PixelPac)
- Includes Jupyter notebooks, test scripts, encoder-decoder experiments, and Caterpillar Tech Challenge documentation.

---

## Troubleshooting

### Common Issues
- **Camera Not Detected**: Ensure Picamera2 is installed and camera is enabled in `raspi-config`.
- **Framebuffer Errors**: Check HDMI connection and resolution settings.
- **High Latency**: Verify PREEMPT_RT kernel is active (`uname -a` should show "PREEMPT_RT").
- **Import Errors**: Activate virtual environment: `source cat_venv/bin/activate`.

### Performance Tuning
- **CPU Affinity**: Pin process to specific cores for reduced jitter.
- **Governor**: Ensure CPU governor is set to "performance".
- **Memory**: Monitor with `htop` or `free -h`.

### Kernel Verification
```bash
uname -a  # Should show PREEMPT_RT
cyclictest -Sp90 -i200 -n -l100000  # Latency test (<200µs expected)
```

---

## Contributing
Contributions are welcome! Please refer to the PixelPac repository for development guidelines and submit pull requests for improvements.

---

## License
This project is licensed under the MIT License. See LICENSE file for details.

---

## Acknowledgments
- **CATERPILLAR TECH CHALLENGE 2025**: For the competition platform.
- **Raspberry Pi Foundation**: For the hardware platform.
- **TensorFlow Lite**: For efficient edge inference.
- **PREEMPT_RT Community**: For real-time kernel patches.

For more details, visit the [RTOS](https://github.com/ShekharShwetank/RTOS) and [PixelPac](https://github.com/ShekharShwetank/PixelPac) repositories.
