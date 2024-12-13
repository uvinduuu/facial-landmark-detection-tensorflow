# Facial Landmark Detection with TensorFlow

This repository provides an end-to-end pipeline for detecting 68 facial landmarks using TensorFlow and Keras. The project includes dataset preprocessing, model training, inference, and result visualization.

---

## Features
- End-to-end implementation for facial landmark detection.
- Preprocessing pipeline for converting datasets into `TFRecord` format.
- Training with checkpointing and TensorBoard monitoring.
- Inference support with visualization of predicted landmarks.

---

## Dataset
The **300W-LP** dataset is used, containing:
- **Images**: Facial images in `.jpg` format.
- **Landmarks**: Corresponding `.mat` files containing 68 key points.

### Dataset Preprocessing
The dataset is preprocessed by:
1. Extracting landmarks from `.mat` files and normalizing them relative to image dimensions.
2. Converting the dataset into `TFRecord` format.

#### Preprocessing Script
Run the following script to preprocess the dataset:

```bash
python preprocess.py --image_dir data/images --mat_dir data/mats --output_dir data/tfrecords
```

This will create:
- `300w_train.record`: Training data.
- `300w_validation.record`: Validation data.

---

## Training the Model

### Training Script
Run the following command to train the model:

```bash
python train.py
```

### Training Configuration
- **Input size**: 128x128x3
- **Batch size**: 16
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Checkpoints**: Saved in the `checkpoints/` directory.

### Monitoring Training
Use TensorBoard to monitor the training progress:

1. Start TensorBoard:

    ```bash
    tensorboard --logdir logs
    ```

2. Open [http://localhost:6006](http://localhost:6006) in your browser.

---

## Inference
Run the inference script to visualize predicted landmarks on test images:

```bash
python infer.py --image path_to_test_image.jpg
```

The script will load the trained model, predict the landmarks, and display the image with overlaid key points.

---

## Directory Structure
The repository should be organized as follows:

```
.
├── checkpoints/             # Model checkpoints
├── exported/                # Trained model directory
├── logs/                    # TensorBoard logs
├── data/                    # Dataset files (images and .mat files)
├── preprocess.py            # Dataset preprocessing script
├── train.py                 # Training script
├── infer.py                 # Inference script
├── 300w_train.record        # Training data in TFRecord format
├── 300w_validation.record   # Validation data in TFRecord format
├── README.md                # Documentation
```

---

## Workflow

### Step 1: Preprocess the Dataset

```bash
python preprocess.py --image_dir data/images --mat_dir data/mats --output_dir data/tfrecords
```

### Step 2: Train the Model

```bash
python train.py
```

### Step 3: Monitor Training Progress

```bash
tensorboard --logdir logs
```

### Step 4: Perform Inference

```bash
python infer.py --image path_to_test_image.jpg
```

---

## Example Results
Below is an example visualization of predicted landmarks:

(Add visualization images here)

---

## Requirements

### Dependencies
Install the required dependencies:

```bash
pip install tensorflow opencv-python numpy
```

### Hardware
- **GPU**: Recommended for training.
- **CPU**: Sufficient for inference.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
