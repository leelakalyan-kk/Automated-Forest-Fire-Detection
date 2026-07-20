# 🔥 Automated Forest Fire Detection System

A Streamlit web application for detecting wildfire presence in images using deep learning models (CNN, ResNet50, InceptionV3).

## Overview

This system uses trained neural networks to classify images as either containing fire or no fire. It provides:
- Multiple model options for inference
- Real-time confidence scores
- Fire alarm audio alert
- Emergency safety guidelines

## ⚠️ Important: Setup Required

**This repository contains the app code only.** The trained model files are NOT included in the repository because they are large (80-100 MB each).

### Models NOT Included (Must be added locally):
- `building_model.keras` (~80 MB)
- `resnet50_forest_fire_detection_model.keras` (~100 MB)
- `inceptionv3_forest_fire_detection_model.keras` (~95 MB)

**The app will NOT run without these model files.**

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/leelakalyan-kk/Automated-Forest-Fire-Detection.git
cd Automated-Forest-Fire-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
(Or manually install: `streamlit tensorflow pillow numpy matplotlib`)

### 3. Add Model Files
You must obtain the trained model files and place them in the project root directory:
```
Automated-Forest-Fire-Detection/
├── app.py
├── building_model.keras
├── resnet50_forest_fire_detection_model.keras
├── inceptionv3_forest_fire_detection_model.keras
└── alarm.mpeg
```

**Where to get the models:**
- Run the training notebook `Try this.ipynb` to automatically generate and save all models
- The notebook trains three models and **automatically saves them** in both `.h5` and `.keras` formats
- Or contact the project maintainer for pre-trained model download links

**Training the Models Automatically:**
To generate the trained models yourself, run the Jupyter notebook:
```bash
jupyter notebook Try this.ipynb
```
The notebook will:
1. Load the wildfire dataset from `the_wildfire_dataset_2n_version/`
2. Train three models (Baseline CNN, ResNet50, InceptionV3)
3. **Automatically save the trained models** to the project root:
   - `building_model.keras` / `building_model.h5`
   - `resnet50_forest_fire_detection_model.keras` / `resnet50_forest_fire_detection_model.h5`
   - `inceptionv3_forest_fire_detection_model.keras` / `inceptionv3_forest_fire_detection_model.h5`

### 4. Run the App
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Usage

1. Select a model from the sidebar
2. Toggle the fire alarm (optional)
3. Upload an image (JPG, PNG, JPEG)
4. View the prediction with confidence score
5. If fire is detected, emergency safety guidelines are displayed

## Model Options

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Baseline CNN | Custom CNN | — |
| ResNet50 | Transfer Learning | — |
| InceptionV3 | Transfer Learning | — |

## Files Included
- `app.py` - Streamlit application
- `Try this.ipynb` - Training notebook (reference)
- `alarm.mpeg` - Fire alarm audio
- Sample images for testing

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Pillow
- NumPy
- Matplotlib

## Deploy on Render

This project can be deployed on Render as a Streamlit web service.

### Important
- The app still needs the trained model files (`building_model.keras`, `resnet50_forest_fire_detection_model.keras`, `inceptionv3_forest_fire_detection_model.keras`) at runtime.
- If those files are not present in the Render instance, the app will show a model-not-found error.

### Automatic model download on Render
The app can download model files at startup if you provide direct file URLs as environment variables:
- `BUILDING_MODEL_URL`
- `RESNET50_MODEL_URL`
- `INCEPTIONV3_MODEL_URL`

Each URL must point directly to a downloadable `.keras` file.

### Render setup
1. Push this repository to GitHub.
2. Sign in to Render and create a new **Web Service**.
3. Connect the GitHub repository.
4. Use the included `render.yaml` or set these manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
5. Add the environment variables for the model URLs in Render.
6. The app will download the models automatically the first time it starts.

### If you want the app to run automatically on Render
You must provide the model files through one of these options:
- Upload them into the repo if they are under the GitHub size limit
- Store them in cloud storage and download them at startup
- Use Git LFS for the model artifacts
