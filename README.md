# ♻️ Discarded Material Identification System (DMIS)

> **Real-Time Waste Detection in Images and Live Video using YOLOv5**
>
> An end-to-end computer vision system that detects and localises discarded materials across **13 waste categories** using a custom-trained YOLOv5s model — with a 3-stage training pipeline, a Flask web interface for image upload and live webcam detection, and automated CI/CD deployment to AWS via GitHub Actions.

---

<div align="center">

[![Python 3.7](https://img.shields.io/badge/Python-3.7-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%2B%20YOLOv5-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/API-Flask-black?logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20EC2-orange?logo=amazonaws)](https://aws.amazon.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want the visual overview first?** The deck covers the detection task, YOLOv5 architecture, pipeline, Flask API design, and deployment in 12 slides.

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1dptiVv-4iGHVXV5HiD78hMibXK-_N6Ru/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Business Problem](#1-business-problem) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset & Waste Categories](#6-dataset--waste-categories) |
| 7 | [ML Pipeline — Step by Step](#7-ml-pipeline--step-by-step) |
| 8 | [YOLOv5 Model — How It Works](#8-yolov5-model--how-it-works) |
| 9 | [Flask Web Application](#9-flask-web-application) |
| 10 | [How to Replicate — Full Setup Guide](#10-how-to-replicate--full-setup-guide) |
| 11 | [Running the Application](#11-running-the-application) |
| 12 | [CI/CD & Cloud Deployment](#12-cicd--cloud-deployment) |
| 13 | [Business Applications & Other Domains](#13-business-applications--other-domains) |
| 14 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 15 | [Troubleshooting](#15-troubleshooting) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Business Problem

### What problem are we solving?

Global waste mismanagement is a major environmental and public health crisis. Manual waste identification and sorting is slow, labour-intensive, error-prone, and not scalable for the volumes produced by modern cities, industrial facilities, and recycling plants. Cameras are already deployed at many waste collection and sorting points — but without computer vision, they are passive observers.

Core pain points:

- 🏭 **Manual sorting bottleneck** — human sorters at recycling facilities cannot keep up with volume; errors contaminate recycling streams
- 🌍 **Illegal dumping** — discarded materials in public spaces, waterways, and green areas are hard to monitor at scale
- 📸 **Cameras without intelligence** — CCTV and inspection cameras generate vast footage that is never analysed for waste events
- 💰 **Cost of misclassification** — incorrectly sorted materials (e.g. non-recyclables in recycling bins) significantly reduce the value of recycled output

### What does DMIS answer?

> *"Given an image or live video frame, which discarded materials are present — and exactly where in the frame are they located?"*

This is an **object detection** problem: the model draws bounding boxes around each detected waste item, assigns it to one of 13 waste categories, and reports the confidence score for each detection.

### Objectives

1. Train a custom YOLOv5s detector on a labelled waste detection dataset (13 classes)
2. Build a 3-stage modular training pipeline: data ingestion, validation, and model training
3. Serve detections via a Flask web API accepting base64-encoded images
4. Support live webcam inference via the `/live` route
5. Containerise with Docker and deploy to AWS EC2 via automated GitHub Actions CI/CD

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Task** | Object detection — bounding box localisation + classification |
| **Model** | YOLOv5s (small) — custom-trained on waste dataset |
| **Number of classes** | 13 waste categories (`nc=13`) |
| **Model architecture** | Custom YOLOv5s (`custom_yolov5s.yaml`) — `depth_multiple=0.33`, `width_multiple=0.5` |
| **Pretrained weights** | `yolov5s.pt` (COCO-pretrained) — fine-tuned on waste data |
| **Training** | 1 epoch (configurable), batch size 16, image size 416×416 |
| **Inference confidence** | `--conf 0.5` (50% minimum confidence threshold) |
| **Dataset source** | Google Drive (downloaded via `gdown`) |
| **Dataset format** | YOLO format: `train/`, `valid/`, `data.yaml` |
| **Inference modes** | Image upload (base64) + live webcam (source=0) |
| **Web framework** | Flask on port 8080 |
| **Frontend** | Bootstrap 4 HTML — image upload + annotated result display |
| **CI/CD** | GitHub Actions → Docker → AWS ECR → AWS EC2 |
| **Python version** | 3.7 |

---

## 3. Tech Stack

### Complete Technology Map

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.7 | Core language |
| **Deep Learning** | PyTorch (≥1.7.0) + Torchvision | YOLOv5 model training and inference |
| **Object Detection** | YOLOv5s (Ultralytics) | YOLO architecture — single-pass object detector |
| **Computer Vision** | OpenCV (`opencv-python`) | Image reading, pre/post processing |
| **Image Processing** | Pillow, NumPy | Image transformations and array ops |
| **Dataset Download** | `gdown` | Downloads labelled dataset from Google Drive |
| **Config Management** | PyYAML | Reads/writes `data.yaml` and model config YAMLs |
| **Web Framework** | Flask + Flask-CORS | REST API: image prediction + training trigger |
| **Frontend** | HTML5, Bootstrap 4, jQuery | Upload image, show annotated result side-by-side |
| **Image I/O** | Base64 (stdlib) | Encodes/decodes images for JSON API transport |
| **Containerisation** | Docker (`python:3.7-slim-buster`) | Packages app + YOLOv5 + AWS CLI |
| **Cloud Compute** | AWS EC2 (Ubuntu) | Hosts Flask app as Docker container |
| **Container Registry** | AWS ECR | Stores Docker images |
| **CI/CD** | GitHub Actions (3-job pipeline) | Build → push → deploy on every `main` push |
| **Logging** | Python `logging` via `from_root` | Timestamped logs to `log/` directory |
| **Error Handling** | Custom `AppException` | Captures filename + line number for all exceptions |
| **Package** | `setuptools` (`setup.py`) | Installs `wasteDetection` as editable package |

---

## 4. High-Level Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  Google Drive  ──(gdown)──►  data.zip                              │
│                                  │                                  │
│                         unzip ───┤                                  │
│                                  ▼                                  │
│                    artifacts/data_ingestion/                        │
│                    feature_store/                                   │
│                       ├── train/      (YOLO format images+labels)  │
│                       ├── valid/      (YOLO format images+labels)  │
│                       └── data.yaml   (class names, nc, paths)     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                              │
│                                                                     │
│  [Stage 1: Data Ingestion]                                          │
│    gdown → data.zip → extract → feature_store/                     │
│                                                                     │
│  [Stage 2: Data Validation]                                         │
│    Check: train/ + valid/ + data.yaml exist → status.txt           │
│                                                                     │
│  [Stage 3: Model Trainer]                                           │
│    Read nc from data.yaml → patch custom_yolov5s.yaml              │
│    python yolov5/train.py --weights yolov5s.pt --epochs 1          │
│    Copy best.pt → yolov5/my_model.pt                               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER  (app.py)                       │
│                                                                     │
│  Flask  port 8080                                                   │
│                                                                     │
│  GET  /         → index.html  (Bootstrap 4 upload UI)              │
│  GET  /train    → TrainPipeline().run_pipeline()                    │
│  POST /predict  → base64 decode → detect.py → base64 encode        │
│                   returns annotated image as JSON                   │
│  GET  /live     → detect.py --source 0  (webcam)                   │
│                                                                     │
│  [Docker Container]  ←──  [AWS EC2 :8080]                          │
│         ▲                                                           │
│  [GitHub push] → [Actions CI/CD] → [ECR] → [EC2 deploy]           │
└─────────────────────────────────────────────────────────────────────┘
```

### Data & Artifact Flow

| Stage | Input | Output |
|-------|-------|--------|
| Data Ingestion | Google Drive URL (`DATA_DOWNLOAD_URL`) | `artifacts/data_ingestion/data.zip` → `feature_store/train/` + `valid/` + `data.yaml` |
| Data Validation | `feature_store/` directory | `artifacts/data_validation/status.txt` (True/False) |
| Model Training | `data.yaml` + `yolov5s.pt` weights | `yolov5/my_model.pt` (best.pt after training) |
| Prediction | Base64 image via POST | Annotated image with bounding boxes returned as base64 JSON |

---

## 5. Repository Structure

```
Discarded-Material-Identification-System/
│
├── wasteDetection/                        # Core Python package
│   ├── __init__.py
│   ├── components/                        # Pipeline stage implementations
│   │   ├── data_ingestion.py              # gdown download + ZIP extract
│   │   ├── data_validation.py             # Check train/valid/data.yaml exist
│   │   └── model_trainer.py               # Patch YAML, run YOLOv5 train.py, copy best.pt
│   ├── constant/
│   │   ├── __init__.py
│   │   ├── application.py                 # APP_HOST="0.0.0.0", APP_PORT=8080
│   │   └── training_pipeline/__init__.py  # All training constants (URLs, epochs, batch size)
│   ├── entity/
│   │   ├── config_entity.py              # Dataclasses: DataIngestionConfig, ModelTrainerConfig
│   │   └── artifacts_entity.py           # Dataclasses: DataIngestionArtifact, ModelTrainerArtifact
│   ├── exception/__init__.py             # AppException with file+line error details
│   ├── logger/__init__.py                # Timestamped log files in log/
│   ├── pipeline/
│   │   └── training_pipeline.py          # TrainPipeline — chains all 3 stages, checks validation
│   └── utils/
│       └── main_utils.py                 # read_yaml_file, write_yaml_file, decodeImage, encodeImageIntoBase64
│
├── yolov5/                                # YOLOv5 framework (Ultralytics)
│   ├── train.py                           # YOLOv5 training script
│   ├── detect.py                          # YOLOv5 inference script
│   ├── models/
│   │   ├── yolov5s.yaml                   # YOLOv5s base architecture
│   │   └── custom_yolov5s.yaml            # Custom config — nc=13 classes
│   └── data/hyps/                         # Hyperparameter configs
│
├── flowcharts/                            # Architecture diagram images
│   ├── Data Ingetions.png
│   ├── Data validation.png
│   └── Model trainer.png
│
├── templates/
│   └── index.html                         # Bootstrap 4 UI — image upload + result display
│
├── artifacts/                             # Pipeline outputs (auto-created)
│   ├── data_ingestion/                    # data.zip + feature_store/
│   └── data_validation/                   # status.txt
│
├── log/                                   # Timestamped execution logs
│
├── app.py                                 # Flask app — 4 routes including live webcam
├── Dockerfile                             # python:3.7-slim-buster, installs awscli + ffmpeg
├── .github/workflows/main.yaml           # 3-job GitHub Actions CI/CD pipeline
├── requirements.txt                       # All deps: Flask, PyTorch, OpenCV, gdown, etc.
└── setup.py                               # Package: wasteDetection
```

---

## 6. Dataset & Waste Categories

### Dataset Format

The training data uses **YOLO format** — the standard for YOLOv5 training:

```
feature_store/
├── train/
│   ├── images/        ← JPEG/PNG images
│   └── labels/        ← .txt files: one line per object = [class_id cx cy w h]
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml          ← class names, nc, train/val paths
```

Each label file line encodes one object: `<class_id> <cx> <cy> <width> <height>` — all values normalised to [0, 1] relative to image dimensions.

### Waste Categories

The model detects **13 categories** of discarded materials (`nc=13` in `custom_yolov5s.yaml`). The specific class names are defined in `data.yaml` within the dataset, which is downloaded at training time from Google Drive. The 13-class scope covers the major discarded material types present at waste collection, sorting, and disposal sites:

> Typical waste detection datasets of this type cover categories such as: cardboard, glass bottles, plastic bottles, plastic bags, metal cans, paper, food waste, foam/polystyrene, clothing/textiles, electronic waste, wood, rubber, and general mixed waste — though the exact class names are defined in the downloaded `data.yaml`.

### Data Validation

Before training begins, the validation stage confirms the following three items exist in the extracted dataset:

```python
DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "valid", "data.yaml"]
```

If any is missing, `status.txt` is written as `False` and the pipeline raises an exception before training starts.

---

## 7. ML Pipeline — Step by Step

The training pipeline is triggered by visiting `http://localhost:8080/train` or by calling `TrainPipeline().run_pipeline()` programmatically. The pipeline checks validation status before allowing model training to proceed.

---

### Stage 1 — Data Ingestion

**Component:** `wasteDetection/components/data_ingestion.py`
**Config:** `DataIngestionConfig`

1. Reads `DATA_DOWNLOAD_URL` (Google Drive share link) from constants
2. Extracts the Google Drive file ID from the URL
3. Constructs gdown URL: `https://drive.google.com/uc?/export=download&id=<file_id>`
4. Downloads `data.zip` to `artifacts/data_ingestion/`
5. Extracts ZIP contents to `artifacts/data_ingestion/feature_store/`
6. Returns `DataIngestionArtifact(data_zip_file_path, feature_store_path)`

---

### Stage 2 — Data Validation

**Component:** `wasteDetection/components/data_validation.py`
**Config:** `DataValidationConfig`

1. Lists all files/folders in `feature_store/`
2. Checks each item against `required_file_list = ["train", "valid", "data.yaml"]`
3. Writes `True` or `False` to `artifacts/data_validation/status.txt`
4. If valid: copies `data.zip` to the project root (needed by model trainer's unzip step)
5. Returns `DataValidationArtifact(validation_status: bool)`

> **Important:** Training only proceeds if `validation_status == True`. If the dataset structure is wrong, the pipeline raises an exception: `"Your data is not in correct format"`.

---

### Stage 3 — Model Training

**Component:** `wasteDetection/components/model_trainer.py`
**Config:** `ModelTrainerConfig`

1. Unzips `data.zip` in the project root (extracts `train/`, `valid/`, `data.yaml`)
2. Reads `nc` (number of classes) from `data.yaml`
3. Reads the base `yolov5s.yaml` architecture config
4. **Patches `nc`** in the config with the actual number of classes from the dataset
5. Writes the patched config as `yolov5/models/custom_yolov5s.yaml`
6. Runs YOLOv5 training:
   ```bash
   cd yolov5/ && python train.py \
     --img 416 \
     --batch 16 \
     --epochs 1 \
     --data ../data.yaml \
     --cfg ./models/custom_yolov5s.yaml \
     --weights yolov5s.pt \
     --name yolov5s_results \
     --cache
   ```
7. Copies `best.pt` → `yolov5/best.pt` and `artifacts/model_trainer/best.pt`
8. Cleans up run artifacts, training data directories, and `data.yaml`
9. Returns `ModelTrainerArtifact(trained_model_file_path="yolov5/best.pt")`

### Training Constants

```python
MODEL_TRAINER_PRETRAINED_WEIGHT_NAME = "yolov5s.pt"   # COCO pretrained
MODEL_TRAINER_NO_EPOCHS              = 1               # Increase for better mAP
MODEL_TRAINER_BATCH_SIZE             = 16
# Image size: 416×416 (hardcoded in os.system call)
# Confidence threshold at inference: 0.5
```

---

## 8. YOLOv5 Model — How It Works

### Why YOLOv5?

YOLO (You Only Look Once) is a single-pass object detection architecture — it processes the entire image in one forward pass, making it extremely fast compared to two-stage detectors (like Faster R-CNN). YOLOv5s is the **small** variant, balancing inference speed and accuracy — ideal for deployment on standard hardware without GPU.

### Custom YOLOv5s Architecture (`custom_yolov5s.yaml`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `nc` | 13 | Number of waste classes |
| `depth_multiple` | 0.33 | Layer depth scaling — controls number of repeated blocks |
| `width_multiple` | 0.5 | Channel width scaling — controls feature map dimensions |
| Anchors | 3 scales × 3 boxes | 9 anchor boxes for detecting small, medium, and large waste items |
| Input size | 416 × 416 | Standard YOLO training resolution |

The backbone uses a **CSP (Cross Stage Partial)** network with `C3` modules and `SPPF` (Spatial Pyramid Pooling Fast) for multi-scale feature extraction. The head uses a **PANet** (Path Aggregation Network) structure with feature fusion at 3 scales — important for detecting waste items at different sizes and distances.

### Training Strategy

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Transfer learning** | Start from `yolov5s.pt` (COCO) | COCO pretraining provides strong visual features — significantly reduces epochs needed |
| **Fine-tuning** | All layers (default YOLOv5) | Full fine-tuning adapts the model to waste-specific appearance |
| **`--cache`** flag | Enabled | Caches images in RAM for faster epoch processing |
| **Epochs** | 1 (default, configurable) | Set `MODEL_TRAINER_NO_EPOCHS` higher (50–100) for production accuracy |

### Inference Pipeline (at `/predict`)

```
POST /predict
  ├── Receive: {"image": "<base64_string>"}
  ├── decodeImage()  →  saves to data/inputImage.jpg
  ├── os.system("cd yolov5/ && python detect.py
  │               --weights my_model.pt
  │               --img 416
  │               --conf 0.5
  │               --source ../data/inputImage.jpg")
  ├── Output: yolov5/runs/detect/exp/inputImage.jpg  (annotated with boxes)
  ├── encodeImageIntoBase64()  →  base64 string
  ├── os.system("rm -rf yolov5/runs")  →  cleanup
  └── Return: {"image": "<annotated_base64>"}
```

The frontend JavaScript renders the returned base64 image alongside the uploaded original, giving side-by-side comparison.

---

## 9. Flask Web Application

The Flask app (`app.py`) serves all four routes from a single process on port 8080.

### Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Renders `index.html` — Bootstrap 4 upload UI |
| `GET` | `/train` | Triggers `TrainPipeline().run_pipeline()` — all 3 stages |
| `POST` | `/predict` | Accepts `{"image": "<base64>"}` JSON → returns annotated image as `{"image": "<base64>"}` |
| `GET` | `/live` | Starts live webcam detection: `detect.py --source 0` |

### Frontend (`templates/index.html`)

The UI provides a split-panel layout:
- **Left panel**: Upload button → file picker → shows selected image preview
- **Right panel**: "Prediction Results" → renders annotated image returned from `/predict`
- **Predict button**: Reads the selected image as base64 via `FileReader`, POSTs to `/predict`, displays annotated result
- **Loading spinner**: Shown during the inference call to indicate processing

### Image API Flow

```javascript
// Frontend converts uploaded image to base64
reader.readAsDataURL(file);
base_data = canvas.toDataURL('image/jpeg').replace(/^data:image.+;base64,/, '');

// POST to /predict
$.ajax({ url: '../predict', type: 'post',
  data: JSON.stringify({ image: base_data }),
  success: function(res) {
    $('.res-part2').append("<img src='data:image/jpeg;base64," + res.image + "' />");
  }
});
```

### Live Webcam Mode

`GET /live` runs `detect.py --source 0` — instructs YOLOv5 to open the default system camera (device index 0) and run continuous detection. This requires the Docker container to have access to the host camera device.

---

## 10. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.7
- Git
- Conda (recommended)
- Docker Desktop (optional)
- AWS account (for deployment only)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sahatanmoyofficial/Discarded-Material-Identification-System.git
cd Discarded-Material-Identification-System
```

---

### Step 2 — Set Up Python Environment

```bash
conda create -n waste python=3.7 -y
conda activate waste
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# Installs: Flask, PyTorch, OpenCV, YOLOv5 deps, gdown, dill, from-root
# Also installs wasteDetection package in editable mode (-e .)
```

> ⚠️ **GPU note:** For GPU training, install the CUDA-compatible PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally) before running `pip install -r requirements.txt`.

---

### Step 4 — Verify YOLOv5 Submodule

The `yolov5/` directory should already be present (it's part of the repository). Confirm:

```bash
ls yolov5/train.py yolov5/detect.py yolov5/models/yolov5s.yaml
```

---

### Step 5 — Run the Application

```bash
python app.py
# Flask starts at http://0.0.0.0:8080
```

If you want to train first before predicting:
1. Open `http://localhost:8080/train` — triggers the full training pipeline
2. Wait for training to complete (downloads data, trains, saves `yolov5/my_model.pt`)
3. Then use `http://localhost:8080/` to upload images for detection

> **Note:** The model file `yolov5/my_model.pt` must exist before predictions will work. Either run training first, or place a pre-trained `my_model.pt` in the `yolov5/` directory.

---

### Step 6 — Test the Prediction API

```bash
# Encode a test image to base64
python -c "
import base64
with open('test_image.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')
print(b64[:50])  # Print first 50 chars to verify
"

# Send to predict endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"<your_base64_string>\"}"
```

---

## 11. Running the Application

### Local Run

```bash
python app.py
# http://localhost:8080
```

### Docker Run

```bash
# Build
docker build -t dmis:latest .

# Run
docker run -d -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=<your_key> \
  -e AWS_SECRET_ACCESS_KEY=<your_secret> \
  -e AWS_REGION=us-east-1 \
  dmis:latest
```

### Trigger Training via Browser

Visit `http://localhost:8080/train` — this runs the full 3-stage pipeline and saves the trained model to `yolov5/my_model.pt`.

### Live Webcam Detection

Visit `http://localhost:8080/live` — requires a connected webcam. For Docker deployments, pass `--device /dev/video0` to expose the camera.

---

## 12. CI/CD & Cloud Deployment

Every push to `main` triggers a 3-job GitHub Actions pipeline:

```
Developer  ──► git push origin main
                      │
             GitHub Actions triggered
                      │
       ┌──────────────┼──────────────────────────────────┐
       │              │                                   │
  Job 1: CI      Job 2: CD (Build)             Job 3: Deploy
  (ubuntu-latest) (ubuntu-latest)              (self-hosted EC2)
       │              │                                   │
  Lint code       Configure AWS creds          Checkout code
  Unit tests      Login to ECR                 Configure AWS creds
                  docker build                 Login to ECR
                  docker push → ECR            docker pull from ECR
                  tag: latest                  docker run -d -p 8080:8080
                                               docker system prune -f
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ECR_LOGIN_URI` | ECR registry URI (e.g. `566373416292.dkr.ecr.ap-south-1.amazonaws.com`) |
| `ECR_REPOSITORY_NAME` | ECR repository name |

### IAM Policies Required

```
AmazonEC2ContainerRegistryFullAccess
AmazonEC2FullAccess
```

### EC2 Setup for Self-Hosted Runner

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker ubuntu && newgrp docker

# Register self-hosted runner
# GitHub → Settings → Actions → Runners → New self-hosted runner → Linux
# Follow the provided commands on your EC2 instance
```

### Dockerfile Summary

```dockerfile
FROM python:3.7-slim-buster
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
```

> **Why `ffmpeg libsm6 libxext6`?** These system libraries are required by OpenCV (`opencv-python`) for image processing in a headless Linux environment. Without them, `import cv2` fails in the container.

---

## 13. Business Applications & Other Domains

### Primary Use Case — Waste Detection & Environmental Monitoring

| User | Value Delivered |
|------|----------------|
| **Recycling facilities** | Automated conveyor belt sorting — identify and route materials by category in real-time |
| **Municipal services** | Deploy on street cameras to detect illegal dumping events automatically |
| **Waste management companies** | Audit waste streams — measure category composition over time |
| **Environmental NGOs** | Drone or mobile surveys of landfill sites and polluted areas |
| **Smart bins** | Camera-equipped bins that identify and log what is deposited |
| **Industrial facilities** | Monitor compliance with waste disposal regulations |

### Adjacent Domains (Same Object Detection Pattern)

| Domain | Analogous System | Key Adaptation |
|--------|-----------------|----------------|
| **Manufacturing QA** | Defect detection on production lines | Replace waste classes with defect types (scratch, crack, discoloration) |
| **Agriculture** | Crop disease detection from field images | Replace waste with disease/pest categories |
| **Retail** | Shelf inventory and misplaced item detection | Replace waste with product SKU classes |
| **Medical imaging** | Polyp / lesion detection in endoscopy | Replace waste with clinical anomaly classes; stricter validation |
| **Traffic & safety** | PPE compliance detection on construction sites | Detect presence/absence of helmet, vest, gloves |
| **Logistics** | Damaged parcel detection in sorting facilities | Replace waste with damage severity categories |
| **Security** | Abandoned object detection in public spaces | Single-class: "unattended bag" |

---

## 14. How to Improve This Project

### 🧠 Model & Training Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Increase epochs** | 🔴 High | `MODEL_TRAINER_NO_EPOCHS = 1` is a demonstration setting — production training needs 50–100+ epochs for meaningful mAP |
| **Evaluate with mAP** | 🔴 High | Add `yolov5/val.py` run after training to log `mAP@0.5` and `mAP@0.5:0.95` metrics — currently no evaluation step |
| **Larger YOLOv5 variant** | 🟡 Medium | Try YOLOv5m or YOLOv5l for better accuracy if hardware allows; tradeoff is inference speed |
| **Data augmentation tuning** | 🟡 Medium | Edit `yolov5/data/hyps/hyp.scratch-low.yaml` to tune mosaic, mixup, colour jitter for waste imagery |
| **More training data** | 🟡 Medium | Waste detection quality is strongly data-driven — expand dataset with more images per class |
| **Upgrade to YOLOv8/v11** | 🟢 Low | Ultralytics has released newer versions with better accuracy and Python API (not just CLI) |

### 🏗️ Engineering & MLOps Improvements

| Area | Recommendation |
|------|---------------|
| **Replace `os.system` calls** | Use Python subprocess with proper error handling; `os.system` returns exit code only — failures are silent |
| **MLflow experiment tracking** | Log mAP, training loss, and hyperparameters per training run |
| **Async training** | Running `train.py` synchronously blocks the Flask server; use Celery/threading for background training |
| **Add training status endpoint** | `GET /train_status` to poll whether training is running, completed, or failed |
| **Model versioning** | Version `my_model.pt` by timestamp or run ID — avoid silently overwriting the production model |
| **Unit tests** | Test `decodeImage`, `encodeImageIntoBase64`, pipeline stage logic, and route responses |
| **Add `/health` endpoint** | Check Flask is running and `my_model.pt` exists — useful for load balancer health checks |
| **Streamline CI lint job** | Jobs 1 CI currently just echoes placeholders — add actual linting (`flake8`) and smoke tests |

### 📦 Product Improvements

- Return **structured JSON** alongside the annotated image — list of detected objects with class name, confidence, and bounding box coordinates
- Add **confidence threshold control** — let the user set `--conf` via a slider in the UI
- Support **video file upload** in addition to single images
- Add a **class filter** — let the user focus detection on specific waste categories
- Display **detection statistics** — count of items per class detected in the image

---

## 15. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `ModuleNotFoundError: wasteDetection` | Run `pip install -r requirements.txt` which installs the package via `-e .` |
| `FileNotFoundError: yolov5/my_model.pt` | Run training first: visit `http://localhost:8080/train` or place `my_model.pt` in `yolov5/` |
| `ImportError: libGL.so.1` or `libSM.so.6` | Install OpenCV system deps: `apt-get install -y ffmpeg libsm6 libxext6` (already in Dockerfile) |
| `gdown` download fails | Check Google Drive link is set to "Anyone with the link can view" — `DATA_DOWNLOAD_URL` in constants |
| `Validation status: False` | Dataset ZIP does not contain `train/`, `valid/`, and `data.yaml` — check the data source |
| Prediction returns `"Invalid input"` | Image base64 string is malformed — ensure `data:image/jpeg;base64,` prefix is stripped before sending |
| Live webcam shows nothing | Webcam not available or not passed to Docker container; add `--device /dev/video0` to `docker run` |
| Port 8080 in use | `lsof -ti:8080 \| xargs kill -9` or change `APP_PORT` in `wasteDetection/constant/application.py` |
| GitHub Actions deploy fails | Confirm EC2 self-hosted runner is online and Docker is running on the EC2 instance |
| Training takes too long | Reduce dataset size, lower batch size, or reduce `--img` to 320 — or run on GPU instance |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **YOLOv5** | You Only Look Once version 5 — a single-pass object detection architecture that predicts bounding boxes and class probabilities in one forward pass |
| **YOLOv5s** | The small variant of YOLOv5 — fewer parameters for faster inference; suitable for CPU deployment |
| **Object Detection** | Computer vision task: locate objects (bounding boxes) and classify them within an image |
| **Bounding Box** | Rectangle (x, y, width, height) drawn around a detected object |
| **nc** | Number of classes — the total number of distinct object categories the model can detect |
| **Transfer Learning** | Using weights pretrained on a large dataset (COCO) as a starting point, then fine-tuning on the target dataset |
| **mAP** | Mean Average Precision — the standard evaluation metric for object detection; higher is better |
| **mAP@0.5** | mAP at IoU threshold 0.5 — a detection is counted as correct if bounding box overlap ≥ 50% |
| **IoU** | Intersection over Union — ratio of overlap between predicted and ground truth bounding boxes |
| **Confidence threshold** | Minimum score (0–1) a detection must reach to be included in results; set to 0.5 here |
| **YOLO format** | Label format: one `.txt` file per image, each line = `class_id cx cy width height` (normalised) |
| **data.yaml** | YOLO dataset config file specifying class names, number of classes, and paths to train/val sets |
| **custom_yolov5s.yaml** | Model architecture config patched with the correct `nc` for the waste dataset |
| **best.pt** | PyTorch checkpoint saved at the training epoch with the best validation mAP |
| **gdown** | Python library for downloading files from Google Drive by share link |
| **SPPF** | Spatial Pyramid Pooling Fast — aggregates features at multiple scales within one layer |
| **CSP / C3** | Cross Stage Partial Network module — improves gradient flow and reduces computation |
| **PANet** | Path Aggregation Network — fuses features from different backbone scales in the detector head |
| **Base64** | Binary-to-text encoding — used here to transport image data over JSON API requests |
| **AppException** | Custom exception class that captures the Python filename and line number where an error occurred |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---

