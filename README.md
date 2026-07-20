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
- If you have the original training notebooks, regenerate them
- Or contact the project maintainer for model download links

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
