import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# =========================
# LOAD MODEL
# =========================
model = YOLO("best.pt")

# =========================
# UI TITLE
# =========================
st.title("🚗 Driver Drowsiness Detection")
st.write("Upload gambar untuk mendeteksi mata terbuka atau tertutup")

# =========================
# SIDEBAR SETTING
# =========================
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    key="confidence_slider"
)

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"],
    key="image_uploader"
)

# =========================
# PROSES JIKA FILE ADA
# =========================
if uploaded_file is not None:

    # Baca gambar dan pastikan RGB
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Prediksi YOLO
    results = model.predict(image_np, conf=confidence)

    # Ambil hasil dengan bounding box
    annotated_frame = results[0].plot()

    # Tampilkan hasil
    st.image(annotated_frame, caption="Detection Result", use_container_width=True)

# =========================
# WEBCAM MODE
# =========================

st.title("Drowsiness Detection - Realtime Webcam")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.25)

        drowsy_detected = False

        if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
            for cls in results[0].boxes.cls:
                class_name = model.names[int(cls)]
                if class_name:
                    drowsy_detected = True

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

