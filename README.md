# ADALITE: Depth Estimation Pipeline
![Depth Prediction](assets/depth.gif)

![Actual Depth Prediction](assets/actual_depth.gif)

## What We Are Doing

ADALITE is a deep learning pipeline for monocular depth estimation using knowledge distillation for Edge-ai device. The project leverages a pre-trained teacher model (TFLite) to generate soft labels, which are then used to train a custom student model for efficient and accurate depth prediction from single RGB images.

---

## How We Are Doing It

The pipeline consists of the following stages:

1. **Data Ingestion**: Downloads and extracts the image dataset from Google Drive.
2. **Teacher Model Download**: Downloads a pre-trained MiDaS TFLite model from KaggleHub.
3. **Soft Label Generation**: Uses the teacher model to generate depth maps (soft labels) for the training images.
4. **Dataset Preparation**: Stores preprocessed images and corresponding soft labels in an HDF5 file for efficient loading.
5. **Student Model Training**: Trains a custom Keras-based depth estimation model using the generated dataset, with cyclical learning rate scheduling and checkpointing.
6. **Model Export**: Saves the trained student model and exports it to TFLite format for deployment.

---

## What Parts I Have Added

- **Custom Student Model**: Implemented in [`models/DepthEstimationModel.py`](models/DepthEstimationModel.py), designed for efficient depth estimation.
- **Data Loader & Soft Label Generation**: Scripts in [`utils/data_loader.py`](utils/data_loader.py) and [`stages/generate_dataset.py`](stages/generate_dataset.py) for generating and storing soft labels.
- **Training Pipeline**: Modular training loop with metric logging, checkpointing, and cyclical learning rate in [`stages/model_training.py`](stages/model_training.py) and [`stages/configure_optimizer_metrices.py`](stages/configure_optimizer_metrices.py).
- **Configuration Management**: YAML-based configuration in [`config/config.yaml`](config/config.yaml).
- **Logging**: Centralized logging via [`utils/logger.py`](utils/logger.py).
- **Dockerization**: Dockerfile for reproducible environment setup.

---

## Pipeline Overview

1. **Download Data**: Fetches and unzips the dataset.
2. **Download Teacher Model**: Downloads MiDaS TFLite model.
3. **Generate Soft Labels**: Runs teacher model inference on images, saves results to HDF5.
4. **Prepare Dataset**: Loads HDF5 data into TensorFlow datasets for training/validation.
5. **Train Student Model**: Trains the student model using the soft labels.
6. **Export Model**: Saves the trained model and exports to TFLite.

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repo-url>
cd ADALITE
```

### 2. Install Dependencies
Using the Docker
```sh
docker build -t adalite-pipeline .
docker run --rm adalite-pipeline
```

### 3. Configure
Edit [`config/config.yaml`](config/config.yaml) to set dataset URLs, paths, and training parameters as needed.
```sh
docker build -t adalite-pipeline .
docker run --rm adalite-pipeline
```

### 4. Run the pipeline
```sh
python main.py
```

All logs will be saved in the [`logs/`](logs/)directory.