import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image

st.title("Forensic Investigation - AR Detection")

# Model load karen
model = YOLO('weights/best.pt') 

# Image upload ka option
source = st.radio("Select Source", ("Image Upload", "Live Stream (IP Webcam)"))

if source == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        results = model(img) # Detection
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Detected Objects', use_column_width=True)

# Note: Live IP Webcam ke liye alag code (webrtc) chahiye hoga