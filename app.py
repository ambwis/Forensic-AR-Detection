import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Forensic AR Investigation", layout="wide")

st.title("🔍 Digital Forensic Investigation using AR")
st.write("Upload an image to detect forensic evidence (Bloodstains, Devices, Weapons, etc.)")

# Load your model (best.pt)
@st.cache_resource
def load_model():
    model = YOLO('best.pt')  # Make sure best.pt is in the same folder
    return model

model = load_model()

# Image Upload Logic
uploaded_file = st.file_uploader("Choose a forensic image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Run Detection
    results = model(img_array)
    
    # Show Results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Detection Results', use_column_width=True)
    
    # Show counts of detected objects
    st.subheader("Evidence Found:")
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            st.write(f"✅ Detected: **{label}**")
