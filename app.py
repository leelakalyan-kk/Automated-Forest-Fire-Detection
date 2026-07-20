import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import base64
from urllib.request import urlretrieve

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="🔥 Forest Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

IMG_SIZE = (160, 160)
CLASS_NAMES = ["Fire", "No Fire"]


def ensure_model_file(model_path, model_url_env_var):
    if os.path.exists(model_path):
        return model_path

    model_url = os.getenv(model_url_env_var, "").strip()
    if not model_url:
        return None

    try:
        st.info(f"Downloading model for {os.path.basename(model_path)}...")
        urlretrieve(model_url, model_path)
        return model_path
    except Exception as error:
        st.error(f"❌ Failed to download {model_path}: {error}")
        return None


# -------------------------------------------------------
# LOAD MODEL (USING H5 FILES)
# -------------------------------------------------------
@st.cache_resource
def load_selected_model(model_name):

    model_paths = {
        "Baseline CNN (Building Model)": "building_model.keras",
        "ResNet50 (Transfer Learning)": "resnet50_forest_fire_detection_model.keras",
        "InceptionV3 (Transfer Learning)": "inceptionv3_forest_fire_detection_model.keras"
    }

    model_urls = {
        "Baseline CNN (Building Model)": "BUILDING_MODEL_URL",
        "ResNet50 (Transfer Learning)": "RESNET50_MODEL_URL",
        "InceptionV3 (Transfer Learning)": "INCEPTIONV3_MODEL_URL"
    }

    model_path = model_paths.get(model_name)
    model_url_env_var = model_urls.get(model_name)

    if model_path and model_url_env_var:
        model_path = ensure_model_file(model_path, model_url_env_var)

    if not model_path or not os.path.exists(model_path):
        st.error(
            f"❌ Model file not found: {model_path}. "
            f"Set the matching environment variable ({model_url_env_var}) to a direct download URL."
        )
        st.stop()

    try:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


# -------------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# -------------------------------------------------------
# ALARM FUNCTION
# -------------------------------------------------------
def play_alarm():
    alarm_file = "alarm.mpeg"

    if os.path.exists(alarm_file):
        with open(alarm_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()

        audio_html = f"""
            <audio autoplay>
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            </audio>
        """

        st.markdown(audio_html, unsafe_allow_html=True)
    # Sound file not found - alarm will be silent


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Baseline CNN (Building Model)",
        "ResNet50 (Transfer Learning)",
        "InceptionV3 (Transfer Learning)"
    ]
)

alarm_enabled = st.sidebar.toggle("🚨 Enable Fire Alarm", value=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Settings:**  
Image Size: 160x160  
Classes: Fire / No Fire  
Model Format: .keras  

**Decision Thresholds:**  
• Fire: ≥ 55% confidence  
• No Fire: ≥ 55% confidence  
• Uncertain: 45-55% range  
""")


# -------------------------------------------------------
# MAIN UI
# -------------------------------------------------------
st.title("🔥 Forest Fire Detection System")
st.markdown("### Upload an image to detect wildfire presence")

model = load_selected_model(selected_model_name)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(processed_image, verbose=0)

    # Sigmoid output corresponds to class index 1 from training.
    # Your dataset class order is typically: ["fire", "nofire"],
    # so model output is probability of "No Fire".
    no_fire_probability = float(prediction[0][0])
    fire_probability = 1.0 - no_fire_probability

    # Determine prediction category with uncertainty threshold
    UNCERTAINTY_LOWER = 0.45
    UNCERTAINTY_UPPER = 0.55
    
    if fire_probability >= UNCERTAINTY_UPPER:
        predicted_class = "Fire"
        confidence = fire_probability
        is_uncertain = False
    elif fire_probability <= UNCERTAINTY_LOWER:
        predicted_class = "No Fire"
        confidence = no_fire_probability
        is_uncertain = False
    else:
        predicted_class = "Uncertain"
        confidence = max(fire_probability, no_fire_probability)
        is_uncertain = True

    probabilities = [fire_probability, no_fire_probability]  # [Fire, No Fire]

    with col2:
        st.subheader("🔎 Prediction Result")

        if predicted_class.lower() == "fire":
            st.error("🔥 FIRE DETECTED")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            if alarm_enabled:
                play_alarm()
            
            # Safety Steps
            st.markdown("---")
            st.subheader("🚨 IMMEDIATE SAFETY STEPS")
            st.warning("""
            **TAKE ACTION NOW:**
            
            1. 📞 **Alert Emergency Services**
               - Call Fire Department: 911 / 101
               - Report exact location coordinates
               
            2. 🔊 **Notify Authorities**
               - Contact Forest Department
               - Alert nearby communities
               
            3. 👥 **Evacuate Area**
               - Move to safe distance upwind
               - Follow designated evacuation routes
               - Do NOT attempt to fight large fires
               
            4. 📸 **Document & Monitor**
               - Take photos/videos if safe
               - Track fire spread direction
               - Update authorities regularly
               
            5. ⚠️ **Safety Precautions**
               - Stay low to avoid smoke
               - Cover nose/mouth with wet cloth
               - Do not return until cleared by authorities
            """)

        elif predicted_class.lower() == "uncertain":
            st.warning("⚠️ UNCERTAIN RESULT")
            st.info(f"Model confidence is too low ({confidence * 100:.2f}%). "
                   f"The model cannot reliably classify this image. "
                   f"Please review manually or try a clearer image.")
            
        else:
            st.success("🌲 NO FIRE DETECTED")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

    # -------------------------------------------------------
    # Probability Chart
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Prediction Probabilities")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ff6b6b' if CLASS_NAMES[i] == "Fire" else '#51cf66' for i in range(len(CLASS_NAMES))]
    bars = ax.bar(CLASS_NAMES, probabilities, color=colors, alpha=0.7)
    
    # Add uncertainty threshold lines
    ax.axhline(y=0.60, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Uncertainty Zone')
    ax.axhline(y=0.40, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Model Confidence Distribution", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

    st.pyplot(fig)


# -------------------------------------------------------
# MODEL INFO SECTION
# -------------------------------------------------------
st.markdown("---")
st.header("📈 Model Architecture")

if selected_model_name == "Baseline CNN (Building Model)":
    st.write("""
    • Custom CNN  
    • Conv2D + MaxPooling  
    • Dropout  
    • Dense layers  
    • Sigmoid output  
    """)

elif selected_model_name == "ResNet50 (Transfer Learning)":
    st.write("""
    • Pretrained ResNet50  
    • Transfer Learning  
    • GlobalAveragePooling  
    • Dense layers  
    • Sigmoid output  
    """)

elif selected_model_name == "InceptionV3 (Transfer Learning)":
    st.write("""
    • Pretrained InceptionV3  
    • Transfer Learning  
    • GlobalAveragePooling  
    • Dense layers  
    • Sigmoid output  
    """)


# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("---")
st.caption("Built with TensorFlow + Streamlit | Forest Fire Detection Project")